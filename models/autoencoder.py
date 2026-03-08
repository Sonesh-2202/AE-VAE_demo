"""
Standard Convolutional Autoencoder
===================================
Architecture:
  Encoder: Input(1,28,28) → Conv → Conv → Flatten → Linear → Latent (z)
  Decoder: Latent (z) → Linear → Unflatten → ConvTranspose → ConvTranspose → Output(1,28,28)

The autoencoder learns a compressed (latent) representation of the input data
by minimizing the reconstruction error between input and output.
"""

import torch
import torch.nn as nn


class Encoder(nn.Module):
    """
    Encodes a 28×28 grayscale image into a latent vector of size `latent_dim`.

    Architecture:
        Conv2d(1→32, 3×3, stride=2, pad=1)  → (32, 14, 14)
        Conv2d(32→64, 3×3, stride=2, pad=1) → (64, 7, 7)
        Flatten                               → (64*7*7 = 3136)
        Linear(3136 → latent_dim)             → (latent_dim)
    """

    def __init__(self, latent_dim: int = 16):
        super().__init__()
        self.conv_layers = nn.Sequential(
            # First convolutional block
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # Second convolutional block
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64 * 7 * 7, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input images of shape (batch, 1, 28, 28)
        Returns:
            Latent representation of shape (batch, latent_dim)
        """
        x = self.conv_layers(x)
        x = self.flatten(x)
        z = self.fc(x)
        return z


class Decoder(nn.Module):
    """
    Decodes a latent vector back into a 28×28 grayscale image.

    Architecture:
        Linear(latent_dim → 3136)             → (3136)
        Unflatten                             → (64, 7, 7)
        ConvTranspose2d(64→32, 3×3, stride=2) → (32, 14, 14)  (with output_padding=1)
        ConvTranspose2d(32→1, 3×3, stride=2)  → (1, 28, 28)   (with output_padding=1)
        Sigmoid                               → pixel values in [0, 1]
    """

    def __init__(self, latent_dim: int = 16):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 64 * 7 * 7)
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(64, 7, 7))
        self.conv_layers = nn.Sequential(
            # First transposed convolutional block
            nn.ConvTranspose2d(
                in_channels=64,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # Second transposed convolutional block → reconstruct image
            nn.ConvTranspose2d(
                in_channels=32,
                out_channels=1,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.Sigmoid(),  # Ensure output pixels are in [0, 1]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Latent vector of shape (batch, latent_dim)
        Returns:
            Reconstructed images of shape (batch, 1, 28, 28)
        """
        x = self.fc(z)
        x = nn.ReLU()(x)
        x = self.unflatten(x)
        x = self.conv_layers(x)
        return x


class Autoencoder(nn.Module):
    """
    Full Autoencoder = Encoder + Decoder.

    The model learns to compress inputs into a low-dimensional latent space
    and then reconstruct them, minimizing mean squared error (MSE).

    Usage:
        model = Autoencoder(latent_dim=16)
        reconstructed = model(images)          # end-to-end forward
        z = model.encode(images)               # just encode
        generated = model.decode(z)            # just decode
    """

    def __init__(self, latent_dim: int = 16):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Compress an image batch into latent vectors."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct images from latent vectors."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward pass: encode then decode."""
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat
