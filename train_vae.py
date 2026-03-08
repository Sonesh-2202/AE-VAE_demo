"""
Train a Variational Autoencoder (VAE) on MNIST
================================================
This script trains the VAE, tracks reconstruction and KL divergence
losses separately, and produces visualizations including generated
samples from the learned latent distribution.

Usage:
    python train_vae.py

All outputs (plots, saved model) go into the `outputs/` directory.
"""

import os
import sys
import time
import torch
import torch.optim as optim

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from models.vae import VAE
from utils.data_loader import get_mnist_loaders
from utils.visualize import (
    plot_training_loss,
    plot_reconstructions,
    plot_latent_space,
    plot_generated_samples,
)


# ╔══════════════════════════════════════════════════════════════════╗
# ║                      HYPERPARAMETERS                            ║
# ╚══════════════════════════════════════════════════════════════════╝
LATENT_DIM = 2           # 2D latent space for visualization & demo
LEARNING_RATE = 1e-3     # Adam optimizer learning rate
BATCH_SIZE = 128         # Mini-batch size
EPOCHS = 50              # More epochs for better quality
KL_WEIGHT = 0.5          # β < 1 → prioritize reconstruction quality
                         # (standard β=1 trades quality for structure)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── GPU Optimization ───────────────────────────────────────────────
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def train_one_epoch(
    model: VAE,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    kl_weight: float = 1.0,
) -> tuple:
    """Train the VAE for one epoch. Returns (total, recon, kl) losses."""
    model.train()
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0

    for images, _ in loader:
        images = images.to(device, non_blocking=True)

        # Forward pass
        x_hat, mu, log_var = model(images)

        # Compute loss
        loss, recon_loss, kl_loss = model.loss_function(
            x_hat, images, mu, log_var, kl_weight=kl_weight
        )

        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl_loss.item()

    n = len(loader)
    return total_loss / n, total_recon / n, total_kl / n


def evaluate(
    model: VAE,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    kl_weight: float = 1.0,
) -> tuple:
    """Evaluate VAE on test set. Returns (total, recon, kl) losses."""
    model.eval()
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device, non_blocking=True)
            x_hat, mu, log_var = model(images)
            loss, recon_loss, kl_loss = model.loss_function(
                x_hat, images, mu, log_var, kl_weight=kl_weight
            )
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()

    n = len(loader)
    return total_loss / n, total_recon / n, total_kl / n


def main():
    print("=" * 60)
    print("  Variational Autoencoder (VAE) — MNIST Training")
    print("=" * 60)
    print(f"  Device      : {DEVICE}")
    if torch.cuda.is_available():
        print(f"  GPU         : {torch.cuda.get_device_name(0)}")
        vram_bytes = torch.cuda.get_device_properties(0).total_memory
        print(f"  VRAM        : {vram_bytes / 1024**3:.1f} GB")
    print(f"  Latent Dim  : {LATENT_DIM}")
    print(f"  Batch Size  : {BATCH_SIZE}")
    print(f"  Epochs      : {EPOCHS}")
    print(f"  LR          : {LEARNING_RATE}")
    print(f"  KL Weight β : {KL_WEIGHT}")
    print("=" * 60)

    # ── Data ────────────────────────────────────────────────────────
    print("\n📁 Loading MNIST dataset...")
    train_loader, test_loader = get_mnist_loaders(
        batch_size=BATCH_SIZE, data_dir="./data"
    )
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Test batches:  {len(test_loader)}")

    # ── Model & Optimizer ───────────────────────────────────────────
    model = VAE(latent_dim=LATENT_DIM).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Learning rate scheduler — reduce LR when loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=7
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n🧠 Model Parameters: {total_params:,} total, {trainable_params:,} trainable")

    # ── Training Loop ───────────────────────────────────────────────
    print("\n🚀 Starting training...\n")
    train_losses = []
    recon_losses = []
    kl_losses = []
    best_test_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        start_time = time.time()

        train_loss, recon_loss, kl_loss = train_one_epoch(
            model, train_loader, optimizer, DEVICE, kl_weight=KL_WEIGHT
        )
        test_loss, test_recon, test_kl = evaluate(
            model, test_loader, DEVICE, kl_weight=KL_WEIGHT
        )

        train_losses.append(train_loss)
        recon_losses.append(recon_loss)
        kl_losses.append(kl_loss)

        # Step the scheduler
        scheduler.step(test_loss)

        # Save best model
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "vae.pth"))

        elapsed = time.time() - start_time
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"  Epoch [{epoch:02d}/{EPOCHS}]  "
            f"Total: {train_loss:.2f}  "
            f"Recon: {recon_loss:.2f}  "
            f"KL: {kl_loss:.2f}  "
            f"Test: {test_loss:.2f}  "
            f"LR: {current_lr:.1e}  "
            f"Time: {elapsed:.1f}s"
            + (" ★" if test_loss <= best_test_loss else "")
        )

    # ── Load Best Model ────────────────────────────────────────────
    model_path = os.path.join(OUTPUT_DIR, "vae.pth")
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    print(f"\n💾 Best model loaded (test loss: {best_test_loss:.2f})")

    # ── Visualizations ──────────────────────────────────────────────
    print("\n📊 Generating visualizations...")

    plot_training_loss(
        losses=train_losses,
        title="VAE — Training Loss Breakdown",
        filename="vae_training_loss.png",
        kl_losses=kl_losses,
        recon_losses=recon_losses,
    )

    plot_reconstructions(
        model=model,
        data_loader=test_loader,
        device=DEVICE,
        title="VAE — Reconstructions",
        filename="vae_reconstructions.png",
        is_vae=True,
    )

    plot_latent_space(
        model=model,
        data_loader=test_loader,
        device=DEVICE,
        title="VAE — 2D Latent Space",
        filename="vae_latent_space.png",
        is_vae=True,
    )

    plot_generated_samples(
        model=model,
        device=DEVICE,
        latent_dim=LATENT_DIM,
        n_samples=64,
        title="VAE — Generated Samples from N(0,I)",
        filename="vae_generated_samples.png",
    )

    # ── Latent Space Manifold ──────────────────────────────────────
    print("\n🎨 Generating latent space grid (2D manifold)...")
    generate_manifold(model, DEVICE, filename="vae_manifold.png")

    print("\n✅ All done! Check the 'outputs/' directory for results.")


def generate_manifold(
    model: VAE,
    device: torch.device,
    n: int = 20,
    digit_size: int = 28,
    filename: str = "vae_manifold.png",
) -> None:
    """Generate a 2D manifold grid of decoded digits."""
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import norm

    model.eval()
    figure = np.zeros((digit_size * n, digit_size * n))

    grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
    grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = torch.tensor([[xi, yi]], dtype=torch.float32).to(device)
            with torch.no_grad():
                x_decoded = model.decode(z_sample).cpu().numpy()
            digit = x_decoded[0].squeeze()
            figure[
                i * digit_size : (i + 1) * digit_size,
                j * digit_size : (j + 1) * digit_size,
            ] = digit

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(figure, cmap="gray")
    ax.set_title("VAE — 2D Latent Space Manifold", fontsize=16, fontweight="bold")
    ax.axis("off")

    save_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Manifold grid saved to {save_path}")


if __name__ == "__main__":
    main()
