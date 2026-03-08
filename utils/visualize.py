"""
Visualization Utilities
========================
Functions for plotting training curves, reconstructions, latent spaces,
and generated samples. All plots are saved to the `outputs/` directory.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional


# Ensure output directory exists
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_training_loss(
    losses: List[float],
    title: str = "Training Loss",
    filename: str = "training_loss.png",
    kl_losses: Optional[List[float]] = None,
    recon_losses: Optional[List[float]] = None,
) -> None:
    """
    Plot the training loss curve over epochs.

    Args:
        losses:       List of total loss values per epoch
        title:        Plot title
        filename:     Output filename
        kl_losses:    (Optional) KL divergence losses per epoch (VAE only)
        recon_losses: (Optional) Reconstruction losses per epoch (VAE only)
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    epochs = range(1, len(losses) + 1)
    ax.plot(epochs, losses, "b-o", linewidth=2, markersize=4, label="Total Loss")

    if recon_losses is not None:
        ax.plot(epochs, recon_losses, "g--s", linewidth=1.5, markersize=3, label="Reconstruction Loss")
    if kl_losses is not None:
        ax.plot(epochs, kl_losses, "r--^", linewidth=1.5, markersize=3, label="KL Divergence")

    ax.set_xlabel("Epoch", fontsize=13)
    ax.set_ylabel("Loss", fontsize=13)
    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Loss plot saved to {save_path}")


def plot_reconstructions(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    n_images: int = 10,
    title: str = "Reconstructions",
    filename: str = "reconstructions.png",
    is_vae: bool = False,
) -> None:
    """
    Show original images (top row) and their reconstructions (bottom row).

    Args:
        model:       Trained autoencoder model
        data_loader: DataLoader to sample images from
        device:      torch device (cpu / cuda)
        n_images:    Number of image pairs to display
        title:       Plot title
        filename:    Output filename
        is_vae:      If True, model returns (x_hat, mu, log_var)
    """
    model.eval()
    images, _ = next(iter(data_loader))
    images = images[:n_images].to(device)

    with torch.no_grad():
        if is_vae:
            recon, _, _ = model(images)
        else:
            recon = model(images)

    images = images.cpu().numpy()
    recon = recon.cpu().numpy()

    fig, axes = plt.subplots(2, n_images, figsize=(n_images * 1.5, 3.5))
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)

    for i in range(n_images):
        # Original
        axes[0, i].imshow(images[i].squeeze(), cmap="gray")
        axes[0, i].axis("off")
        if i == 0:
            axes[0, i].set_title("Original", fontsize=10)

        # Reconstruction
        axes[1, i].imshow(recon[i].squeeze(), cmap="gray")
        axes[1, i].axis("off")
        if i == 0:
            axes[1, i].set_title("Reconstructed", fontsize=10)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Reconstruction plot saved to {save_path}")


def plot_latent_space(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    title: str = "Latent Space",
    filename: str = "latent_space.png",
    is_vae: bool = False,
    max_samples: int = 5000,
) -> None:
    """
    Scatter plot of the first 2 dimensions of the latent space,
    colored by digit class.

    Args:
        model:       Trained model
        data_loader: DataLoader with labeled data
        device:      torch device
        title:       Plot title
        filename:    Output filename
        is_vae:      If True, encoder returns (mu, log_var)
        max_samples: Maximum number of points to plot
    """
    model.eval()
    all_z = []
    all_labels = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            if is_vae:
                mu, _ = model.encode(images)
                z = mu  # Use the mean for visualization
            else:
                z = model.encode(images)
            all_z.append(z.cpu())
            all_labels.append(labels)

            if sum(z.size(0) for z in all_z) >= max_samples:
                break

    all_z = torch.cat(all_z, dim=0)[:max_samples].numpy()
    all_labels = torch.cat(all_labels, dim=0)[:max_samples].numpy()

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        all_z[:, 0],
        all_z[:, 1],
        c=all_labels,
        cmap="tab10",
        s=8,
        alpha=0.6,
        edgecolors="none",
    )
    cbar = plt.colorbar(scatter, ax=ax, ticks=range(10))
    cbar.set_label("Digit Class", fontsize=12)
    ax.set_xlabel("Latent Dimension 1", fontsize=13)
    ax.set_ylabel("Latent Dimension 2", fontsize=13)
    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Latent space plot saved to {save_path}")


def plot_generated_samples(
    model: torch.nn.Module,
    device: torch.device,
    latent_dim: int,
    n_samples: int = 64,
    title: str = "Generated Samples",
    filename: str = "generated_samples.png",
) -> None:
    """
    Generate new images by sampling random latent vectors from N(0, I)
    and decoding them.

    Args:
        model:      Trained model with a `decode()` method
        device:     torch device
        latent_dim: Size of the latent space
        n_samples:  Number of images to generate (should be a perfect square)
        title:      Plot title
        filename:   Output filename
    """
    model.eval()
    n_row = int(np.sqrt(n_samples))

    # Sample from standard normal
    z = torch.randn(n_samples, latent_dim).to(device)

    with torch.no_grad():
        generated = model.decode(z).cpu().numpy()

    fig, axes = plt.subplots(n_row, n_row, figsize=(n_row * 1.2, n_row * 1.2))
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)

    for i in range(n_row):
        for j in range(n_row):
            idx = i * n_row + j
            axes[i, j].imshow(generated[idx].squeeze(), cmap="gray")
            axes[i, j].axis("off")

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Generated samples saved to {save_path}")
