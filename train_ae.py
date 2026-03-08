"""
Train a Standard Autoencoder on MNIST
=======================================
This script trains the convolutional autoencoder, logs loss per epoch,
and produces visualizations of reconstructions, latent space, and
generated samples.

Usage:
    python train_ae.py

All outputs (plots, saved model) go into the `outputs/` directory.
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from models.autoencoder import Autoencoder
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
LATENT_DIM = 16          # Size of the bottleneck / latent vector
LEARNING_RATE = 1e-3     # Adam optimizer learning rate
BATCH_SIZE = 128         # Mini-batch size
EPOCHS = 50              # Number of training epochs
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── GPU Optimization ───────────────────────────────────────────────
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True   # Auto-tune conv algorithms
    torch.backends.cudnn.deterministic = False


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Train the model for one epoch. Returns average loss."""
    model.train()
    total_loss = 0.0

    for images, _ in loader:
        images = images.to(device, non_blocking=True)

        reconstructed = model(images)
        loss = criterion(reconstructed, images)

        optimizer.zero_grad(set_to_none=True)  # Slightly faster than zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Evaluate model on test set. Returns average loss."""
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device, non_blocking=True)
            reconstructed = model(images)
            loss = criterion(reconstructed, images)
            total_loss += loss.item()

    return total_loss / len(loader)


def main():
    print("=" * 60)
    print("  Standard Autoencoder — MNIST Training")
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
    print("=" * 60)

    # ── Data ────────────────────────────────────────────────────────
    print("\n📁 Loading MNIST dataset...")
    train_loader, test_loader = get_mnist_loaders(
        batch_size=BATCH_SIZE, data_dir="./data"
    )
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Test batches:  {len(test_loader)}")

    # ── Model, Loss, Optimizer ──────────────────────────────────────
    model = Autoencoder(latent_dim=LATENT_DIM).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Learning rate scheduler — reduce LR when loss stops decreasing
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n🧠 Model Parameters: {total_params:,} total, {trainable_params:,} trainable")

    # ── Training Loop ───────────────────────────────────────────────
    print("\n🚀 Starting training...\n")
    train_losses = []
    test_losses = []
    best_test_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        start_time = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        test_loss = evaluate(model, test_loader, criterion, DEVICE)

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        # Step the scheduler
        scheduler.step(test_loss)

        # Save best model
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "autoencoder.pth"))

        elapsed = time.time() - start_time
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"  Epoch [{epoch:02d}/{EPOCHS}]  "
            f"Train: {train_loss:.6f}  "
            f"Test: {test_loss:.6f}  "
            f"LR: {current_lr:.1e}  "
            f"Time: {elapsed:.1f}s"
            + (" ★" if test_loss <= best_test_loss else "")
        )

    # ── Load Best Model ────────────────────────────────────────────
    model_path = os.path.join(OUTPUT_DIR, "autoencoder.pth")
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    print(f"\n💾 Best model loaded (test loss: {best_test_loss:.6f})")

    # ── Visualizations ──────────────────────────────────────────────
    print("\n📊 Generating visualizations...")

    plot_training_loss(
        losses=train_losses,
        title="Standard AE — Training Loss",
        filename="ae_training_loss.png",
    )

    plot_reconstructions(
        model=model,
        data_loader=test_loader,
        device=DEVICE,
        title="Standard AE — Reconstructions",
        filename="ae_reconstructions.png",
        is_vae=False,
    )

    plot_latent_space(
        model=model,
        data_loader=test_loader,
        device=DEVICE,
        title="Standard AE — Latent Space (first 2 dims)",
        filename="ae_latent_space.png",
        is_vae=False,
    )

    plot_generated_samples(
        model=model,
        device=DEVICE,
        latent_dim=LATENT_DIM,
        title="Standard AE — Generated Samples",
        filename="ae_generated_samples.png",
    )

    print("\n✅ All done! Check the 'outputs/' directory for results.")


if __name__ == "__main__":
    main()
