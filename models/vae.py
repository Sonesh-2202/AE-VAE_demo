"""
Variational Autoencoder (VAE)
==============================
Unlike a standard autoencoder, a VAE learns a *probabilistic* latent space.
Instead of mapping inputs to a single latent vector, the encoder outputs the
parameters of a Gaussian distribution (mean μ and log-variance log σ²).

Key Concepts:
  1. Encoder outputs μ and log σ² (two vectors of size latent_dim).
  2. Reparameterization Trick: z = μ + σ · ε, where ε ~ N(0, I).
     This allows gradients to flow through the sampling step.
  3. Loss = Reconstruction Loss (BCE) + KL Divergence.
     KL(q(z|x) || p(z)) pushes the learned distribution toward N(0, I).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class VAE(nn.Module):
    """
    Variational Autoencoder with convolutional encoder and decoder.

    The encoder produces μ and log σ² for each input, and the decoder
    reconstructs the input from a sampled latent vector z.

    Usage:
        model = VAE(latent_dim=16)
        x_hat, mu, log_var = model(images)
        loss = model.loss_function(x_hat, images, mu, log_var)
    """

    def __init__(self, latent_dim: int = 16):
        super().__init__()
        self.latent_dim = latent_dim

        # ── Encoder ──────────────────────────────────────────────────────
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # After 3 stride-2 convolutions on 28×28: spatial size = 4×4
        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(128 * 4 * 4, latent_dim)       # Mean μ
        self.fc_log_var = nn.Linear(128 * 4 * 4, latent_dim)  # Log-variance log σ²

        # ── Decoder ──────────────────────────────────────────────────────
        self.fc_decode = nn.Linear(latent_dim, 128 * 4 * 4)
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(128, 4, 4))
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=0),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input images into distribution parameters.

        Args:
            x: Input images (batch, 1, 28, 28)
        Returns:
            mu:      Mean of the latent Gaussian (batch, latent_dim)
            log_var: Log-variance of the latent Gaussian (batch, latent_dim)
        """
        h = self.encoder_conv(x)
        h = self.flatten(h)
        mu = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        return mu, log_var

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization Trick:
            z = μ + σ · ε,   where ε ~ N(0, I)

        This makes the sampling differentiable — gradients flow through μ and σ,
        not through the random noise ε.

        Args:
            mu:      Mean (batch, latent_dim)
            log_var: Log-variance (batch, latent_dim)
        Returns:
            z: Sampled latent vector (batch, latent_dim)
        """
        std = torch.exp(0.5 * log_var)  # σ = exp(0.5 · log σ²)
        eps = torch.randn_like(std)      # ε ~ N(0, I)
        z = mu + std * eps
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode a latent vector back into an image.

        Args:
            z: Latent vector (batch, latent_dim)
        Returns:
            Reconstructed images (batch, 1, 28, 28)
        """
        h = self.fc_decode(z)
        h = nn.LeakyReLU(0.2)(h)
        h = self.unflatten(h)
        x_hat = self.decoder_conv(h)
        return x_hat

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass: encode → sample → decode.

        Returns:
            x_hat:   Reconstructed images (batch, 1, 28, 28)
            mu:      Latent mean (batch, latent_dim)
            log_var: Latent log-variance (batch, latent_dim)
        """
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_hat = self.decode(z)
        return x_hat, mu, log_var

    @staticmethod
    def loss_function(
        x_hat: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        log_var: torch.Tensor,
        kl_weight: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        VAE Loss = Reconstruction Loss + β · KL Divergence

        Reconstruction Loss (BCE):
            Measures how well the decoder output matches the original input,
            pixel by pixel, using binary cross-entropy.

        KL Divergence:
            KL(q(z|x) || p(z)) = -0.5 · Σ(1 + log σ² - μ² - σ²)
            Regularizes the learned latent distribution toward N(0, I).

        Args:
            x_hat:     Reconstructed images
            x:         Original images
            mu:        Latent means
            log_var:   Latent log-variances
            kl_weight: β weighting factor for KL term (default 1.0)

        Returns:
            total_loss:  Combined loss
            recon_loss:  Reconstruction component
            kl_loss:     KL divergence component
        """
        # Reconstruction loss (sum over pixels, mean over batch)
        recon_loss = F.binary_cross_entropy(x_hat, x, reduction="sum") / x.size(0)

        # KL divergence (analytically computed for Gaussian)
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / x.size(0)

        total_loss = recon_loss + kl_weight * kl_loss
        return total_loss, recon_loss, kl_loss
