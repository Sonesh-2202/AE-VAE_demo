# Code Walkthrough — Line by Line

> Every file in this project explained, line by line, so you understand **exactly** what each piece of code does.

---

## Table of Contents

1. [models/autoencoder.py — Standard Autoencoder](#1-modelsautoencoderpy)
2. [models/vae.py — Variational Autoencoder](#2-modelsvaepy)
3. [utils/data_loader.py — Data Loading](#3-utilsdata_loaderpy)
4. [utils/visualize.py — Visualization](#4-utilsvisualizepy)
5. [train_ae.py — AE Training Script](#5-train_aepy)
6. [train_vae.py — VAE Training Script](#6-train_vaepy)
7. [export_onnx.py — ONNX Export for Browser](#7-export_onnxpy)
8. [demo/app.js — Browser Inference](#8-demoappjs)

---

## 1. `models/autoencoder.py`

### Imports

```python
import torch          # Core PyTorch library — tensors, autograd, GPU support
import torch.nn as nn # Neural network module — layers, loss functions, containers
```

### Encoder Class

```python
class Encoder(nn.Module):
    # Inherits from nn.Module — the base class for ALL neural networks in PyTorch
    # nn.Module gives us: parameter tracking, GPU movement, save/load, etc.

    def __init__(self, latent_dim: int = 16):
        super().__init__()
        # super().__init__() calls nn.Module's constructor
        # This registers all layers so PyTorch can track their parameters

        self.conv_layers = nn.Sequential(
            # nn.Sequential: a container that runs layers in order, one after another

            nn.Conv2d(
                in_channels=1,      # Input has 1 channel (grayscale image)
                out_channels=32,    # Produce 32 feature maps (32 different "filters")
                kernel_size=3,      # Each filter is 3×3 pixels
                stride=2,           # Move the filter 2 pixels at a time → halves the image
                padding=1,          # Add 1 pixel of zeros around the border → keeps math clean
            ),
            # Input:  (batch, 1, 28, 28)
            # Output: (batch, 32, 14, 14)  ← 28/2 = 14

            nn.BatchNorm2d(32),
            # Normalizes each of the 32 feature maps to have mean≈0, std≈1
            # Why? Prevents internal covariate shift → faster, more stable training

            nn.ReLU(inplace=True),
            # ReLU(x) = max(0, x) — kills all negative values
            # inplace=True: modifies tensor directly instead of creating a copy (saves memory)

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            # Input:  (batch, 32, 14, 14)
            # Output: (batch, 64, 7, 7)  ← 14/2 = 7

            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.flatten = nn.Flatten()
        # Reshape from (batch, 64, 7, 7) → (batch, 64*7*7) = (batch, 3136)
        # Flattens the 3D feature maps into a 1D vector for the linear layer

        self.fc = nn.Linear(64 * 7 * 7, latent_dim)
        # Fully connected layer: 3136 inputs → latent_dim outputs
        # This is the actual "compression" — going from 3136 numbers to just 16

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # forward() defines what happens when you call encoder(images)
        # PyTorch automatically calls this when you do: output = model(input)

        x = self.conv_layers(x)   # (batch, 1, 28, 28) → (batch, 64, 7, 7)
        x = self.flatten(x)       # (batch, 64, 7, 7)  → (batch, 3136)
        z = self.fc(x)            # (batch, 3136)       → (batch, latent_dim)
        return z                  # This IS the compressed representation
```

### Decoder Class

```python
class Decoder(nn.Module):

    def __init__(self, latent_dim: int = 16):
        super().__init__()

        self.fc = nn.Linear(latent_dim, 64 * 7 * 7)
        # Expand from latent_dim back to 3136
        # This is the reverse of the encoder's final compression

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(64, 7, 7))
        # Reshape from (batch, 3136) → (batch, 64, 7, 7)
        # dim=1 means "reshape dimension 1" (not the batch dimension)

        self.conv_layers = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=64,
                out_channels=32,
                kernel_size=3,
                stride=2,           # Doubles spatial size (opposite of Conv2d stride=2)
                padding=1,
                output_padding=1,   # Needed to get exact size: 7*2 = 14
            ),
            # Input:  (batch, 64, 7, 7)
            # Output: (batch, 32, 14, 14)  ← ConvTranspose DOUBLES the size

            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            # Input:  (batch, 32, 14, 14)
            # Output: (batch, 1, 28, 28)  ← Back to original image size!

            nn.Sigmoid(),
            # Sigmoid(x) = 1 / (1 + e^(-x))
            # Squashes output to [0, 1] range — same range as our input pixels
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z)              # (batch, latent_dim) → (batch, 3136)
        x = nn.ReLU()(x)            # Apply ReLU activation
        x = self.unflatten(x)       # (batch, 3136) → (batch, 64, 7, 7)
        x = self.conv_layers(x)     # (batch, 64, 7, 7) → (batch, 1, 28, 28)
        return x                    # Reconstructed image!
```

### Autoencoder Wrapper

```python
class Autoencoder(nn.Module):
    def __init__(self, latent_dim: int = 16):
        super().__init__()
        self.encoder = Encoder(latent_dim)  # Create the encoder
        self.decoder = Decoder(latent_dim)  # Create the decoder

    def encode(self, x):
        return self.encoder(x)    # Just expose encoder as a method

    def decode(self, z):
        return self.decoder(z)    # Just expose decoder as a method

    def forward(self, x):
        z = self.encode(x)       # Step 1: Compress input to latent vector
        x_hat = self.decode(z)   # Step 2: Reconstruct from latent vector
        return x_hat             # Return the reconstruction
```

---

## 2. `models/vae.py`

### Imports

```python
import torch
import torch.nn as nn
import torch.nn.functional as F    # Functional API — stateless operations like BCE loss
from typing import Tuple           # Type hints for multiple return values
```

### VAE Encoder

```python
self.encoder_conv = nn.Sequential(
    nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),    # (1,28,28)→(32,14,14)
    nn.BatchNorm2d(32),
    nn.LeakyReLU(0.2, inplace=True),
    # LeakyReLU: like ReLU but allows small negative values (slope=0.2)
    # LeakyReLU(x) = x if x>0, else 0.2*x
    # Why? Prevents "dead neurons" that ReLU can cause

    nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),   # (32,14,14)→(64,7,7)
    nn.BatchNorm2d(64),
    nn.LeakyReLU(0.2, inplace=True),

    nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # (64,7,7)→(128,4,4)
    nn.BatchNorm2d(128),
    nn.LeakyReLU(0.2, inplace=True),
    # VAE has 3 conv layers (deeper than AE's 2) for better feature extraction
)

# THE KEY DIFFERENCE FROM STANDARD AE:
# Instead of ONE output vector, we get TWO:
self.fc_mu = nn.Linear(128 * 4 * 4, latent_dim)
# fc_mu outputs μ (mean) — "where is the center of the distribution?"

self.fc_log_var = nn.Linear(128 * 4 * 4, latent_dim)
# fc_log_var outputs log(σ²) — "how spread out is the distribution?"
# We predict LOG variance because variance must be positive,
# and exp(anything) is always positive
```

### Reparameterization Trick

```python
def reparameterize(self, mu, log_var):
    std = torch.exp(0.5 * log_var)
    # Convert log(σ²) → σ
    # Math: σ = √(σ²) = √(exp(log(σ²))) = exp(0.5 * log(σ²))

    eps = torch.randn_like(std)
    # Sample random noise ε from N(0, 1)
    # randn_like: same shape and device as std, filled with random normals

    z = mu + std * eps
    # z = μ + σ·ε
    # This IS the reparameterization trick!
    # Gradients flow through μ and σ (learnable), NOT through ε (random)

    return z
```

### VAE Loss Function

```python
@staticmethod  # No need for self — it's a pure function
def loss_function(x_hat, x, mu, log_var, kl_weight=1.0):

    recon_loss = F.binary_cross_entropy(x_hat, x, reduction="sum") / x.size(0)
    # Binary Cross-Entropy:  -Σ[x·log(x̂) + (1-x)·log(1-x̂)]
    # reduction="sum": sum all pixel losses (not average)
    # / x.size(0): divide by batch size to get per-sample loss
    # Measures: "How well did we rebuild the input?"

    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / x.size(0)
    # KL Divergence:  -½ · Σ(1 + log(σ²) - μ² - σ²)
    # mu.pow(2):   μ²  (element-wise square)
    # log_var.exp(): exp(log(σ²)) = σ²
    # Measures: "How far is the learned distribution from N(0,1)?"

    total_loss = recon_loss + kl_weight * kl_loss
    # Total = Reconstruction + β × KL
    # kl_weight (β) controls the trade-off between quality and regularity
    # β = 0.5 in our setup: prioritizes sharper reconstructions

    return total_loss, recon_loss, kl_loss
    # Return all three so we can track them separately during training
```

---

## 3. `utils/data_loader.py`

```python
from torchvision import datasets, transforms
# datasets: provides standard ML datasets (MNIST, CIFAR, etc.)
# transforms: preprocessing operations (convert to tensor, normalize, etc.)

def get_mnist_loaders(batch_size=128, data_dir="./data", num_workers=2):

    transform = transforms.Compose([
        transforms.ToTensor(),
        # Converts PIL Image (0-255, H×W×C) → PyTorch Tensor (0.0-1.0, C×H×W)
        # This is ALL we need — MNIST pixels become floats in [0, 1]
    ])

    train_dataset = datasets.MNIST(
        root=data_dir,       # Where to store the files
        train=True,          # Use the 60,000 training images
        download=True,       # Download from the internet if not found locally
        transform=transform, # Apply our ToTensor transform
    )

    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,         # Use the 10,000 test images
        download=True,
        transform=transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,  # How many images per batch (128)
        shuffle=True,           # Randomize order each epoch → better training
        num_workers=num_workers, # Parallel data loading threads
        pin_memory=torch.cuda.is_available(),
        # pin_memory=True: keeps data in CPU "pinned" memory → faster GPU transfer
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,         # Don't shuffle test data — we want consistent results
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, test_loader
```

---

## 4. `utils/visualize.py`

### Plot Reconstructions (Key Function)

```python
def plot_reconstructions(model, data_loader, device, n_images=10, ...):
    model.eval()
    # Switch model to evaluation mode:
    # - BatchNorm uses running stats (not batch stats)
    # - Dropout is disabled
    # Always do this before inference!

    images, _ = next(iter(data_loader))
    # next(iter(...)): grab the first batch from the DataLoader
    # images: the pixel data (batch, 1, 28, 28)
    # _: the labels (0-9) — we don't need them here

    images = images[:n_images].to(device)
    # Take only the first n_images from the batch
    # .to(device): move to GPU if available

    with torch.no_grad():
        # Disable gradient computation — we're not training, just predicting
        # This saves memory and speeds things up
        if is_vae:
            recon, _, _ = model(images)  # VAE returns (x_hat, mu, log_var)
        else:
            recon = model(images)         # AE returns just x_hat

    # Convert to numpy for matplotlib
    images = images.cpu().numpy()   # .cpu(): move from GPU back to CPU
    recon = recon.cpu().numpy()     # .numpy(): convert tensor to numpy array

    fig, axes = plt.subplots(2, n_images, figsize=(n_images * 1.5, 3.5))
    # Create a grid: 2 rows (original + reconstructed) × n_images columns

    for i in range(n_images):
        axes[0, i].imshow(images[i].squeeze(), cmap="gray")
        # .squeeze(): remove the channel dim (1,28,28)→(28,28)
        # cmap="gray": display as grayscale

        axes[1, i].imshow(recon[i].squeeze(), cmap="gray")
        axes[0, i].axis("off")  # Hide axis ticks
        axes[1, i].axis("off")
```

### Plot Latent Space (Key Function)

```python
def plot_latent_space(model, data_loader, device, ...):
    model.eval()
    all_z = []       # Collect all latent vectors
    all_labels = []  # Collect all digit labels

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            if is_vae:
                mu, _ = model.encode(images)
                z = mu  # Use mean (not sampled z) for cleaner visualization
            else:
                z = model.encode(images)
            all_z.append(z.cpu())
            all_labels.append(labels)

    # torch.cat: concatenate all batches into one big tensor
    all_z = torch.cat(all_z, dim=0)[:max_samples].numpy()
    all_labels = torch.cat(all_labels, dim=0)[:max_samples].numpy()

    scatter = ax.scatter(
        all_z[:, 0],     # First latent dimension → x-axis
        all_z[:, 1],     # Second latent dimension → y-axis
        c=all_labels,    # Color by digit class (0-9)
        cmap="tab10",    # Use 10-color colormap
        s=8,             # Small dot size
        alpha=0.6,       # Semi-transparent
    )
```

---

## 5. `train_ae.py`

### Hyperparameters

```python
LATENT_DIM = 16          # Bottleneck size — compress 784 pixels into 16 numbers
LEARNING_RATE = 1e-3     # How fast to update weights (0.001)
BATCH_SIZE = 128         # Process 128 images at once
EPOCHS = 50              # Full passes through the dataset (increased from 20)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Use GPU if available, otherwise CPU
```

### GPU Optimization

```python
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    # Auto-tune the best CUDA convolution algorithm for your hardware
    # Makes first epoch slightly slower but all subsequent epochs faster

    torch.backends.cudnn.deterministic = False
    # Allow non-deterministic algorithms for maximum speed
```

### Training Loop (with GPU optimizations)

```python
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        # non_blocking=True: transfers data to GPU asynchronously
        # CPU continues working while data is being transferred → overlap

        reconstructed = model(images)
        loss = criterion(reconstructed, images)

        optimizer.zero_grad(set_to_none=True)
        # set_to_none=True: sets gradients to None instead of zero
        # Slightly faster because it avoids a memory allocation

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)
```

### Main Flow (with LR Scheduler + Best Model Saving)

```python
def main():
    model = Autoencoder(latent_dim=LATENT_DIM).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Learning rate scheduler — automatically reduces LR when loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    # mode="min": we want loss to go DOWN
    # factor=0.5: when triggered, multiply LR by 0.5 (halve it)
    # patience=5: wait 5 epochs of no improvement before reducing

    best_test_loss = float("inf")  # Track the best model

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        test_loss = evaluate(model, test_loader, criterion, DEVICE)

        scheduler.step(test_loss)  # Let scheduler see the test loss

        # Save best model — only keep the one with lowest test loss
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model.state_dict(), "outputs/autoencoder.pth")

    # After all 50 epochs, load the BEST checkpoint (not the last!)
    model.load_state_dict(torch.load("outputs/autoencoder.pth",
                                      map_location=DEVICE, weights_only=True))
    # weights_only=True: security measure — prevents arbitrary code execution
    # map_location: ensures weights load to the correct device (CPU/GPU)
```

---

## 6. `train_vae.py`

### Key Differences from `train_ae.py`

```python
LATENT_DIM = 2
# Only 2 dimensions! This lets us plot the entire latent space in 2D
# The VAE still works because KL regularization makes it use the space efficiently

KL_WEIGHT = 0.5  # (changed from 1.0 for better reconstruction quality)
# β parameter — controls trade-off:
#   β < 1: BETTER reconstructions, less structured latent space    ← OUR CHOICE
#   β = 1: standard VAE — balanced
#   β > 1: more disentangled latent space, blurrier outputs (β-VAE)

# With only 2 latent dims, we need β < 1 to prioritize reconstruction
# because compression is extreme (784 → 2 = 392:1 ratio)
```

### VAE Training Loop

```python
x_hat, mu, log_var = model(images)
# VAE returns THREE things:
#   x_hat:   the reconstructed image
#   mu:      the mean of the latent distribution
#   log_var: the log-variance of the latent distribution

loss, recon_loss, kl_loss = model.loss_function(
    x_hat, images, mu, log_var, kl_weight=KL_WEIGHT
)
# Loss = Reconstruction + β × KL Divergence
# We track all three separately to monitor training
```

### Manifold Generation

```python
def generate_manifold(model, device, n=20, digit_size=28):
    from scipy.stats import norm

    grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
    # norm.ppf: "percent point function" = inverse of the normal CDF
    # Maps uniform [0.05, 0.95] → Gaussian values [-1.64, 1.64]
    # This ensures we sample evenly across the latent distribution

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = torch.tensor([[xi, yi]])
            # Create a 2D latent vector at position (xi, yi)

            x_decoded = model.decode(z_sample)
            # Ask the decoder: "what digit lives at this latent position?"

            # Place the decoded 28×28 digit tile into the big grid
            figure[i*28:(i+1)*28, j*28:(j+1)*28] = digit
    # Result: a 560×560 image showing how digits morph across the latent space
```

---

## 7. `export_onnx.py`

This script converts trained PyTorch models (`.pth`) into ONNX format (`.onnx`) for running in the browser.

### Why ONNX?

```python
# PyTorch models can't run directly in a browser.
# ONNX (Open Neural Network Exchange) is a universal format that
# can be loaded by ONNX Runtime Web (a JavaScript library).
# The conversion: PyTorch (.pth) → ONNX (.onnx) → Browser (via ort-web)
```

### Key Export Steps

```python
dummy_input = torch.randn(1, 1, 28, 28)
# Create a fake input — ONNX needs to trace through the model once
# to capture the computation graph

torch.onnx.export(
    model,                         # The trained model
    dummy_input,                   # Example input for tracing
    "demo/models/autoencoder.onnx",  # Output path
    opset_version=13,              # ONNX operation set version
    input_names=["input"],         # Name for the input tensor
    output_names=["output"],       # Name for the output tensor
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    # dynamic_axes: allows variable batch size (important for browser)
)
```

### Ensuring Browser Compatibility

```python
import onnx
model_proto = onnx.load("demo/models/autoencoder.onnx")
model_proto.ir_version = 8  # Force IR version 8 for maximum compatibility

onnx.save_model(model_proto, "demo/models/autoencoder.onnx",
                save_as_external_data=False)
# save_as_external_data=False: embed ALL weights inside the .onnx file
# Without this, weights go into a separate .data file that browsers can't find
```

---

## 8. `demo/app.js`

The JavaScript file that runs the models in your browser.

### Loading Models

```javascript
async function loadModels() {
    // ort = ONNX Runtime Web (loaded from CDN in index.html)
    ort.env.wasm.numThreads = 1;  // Single-thread for compatibility

    // Load model as ArrayBuffer (binary data) via fetch
    const response = await fetch('models/autoencoder.onnx');
    const buffer = await response.arrayBuffer();
    // Why ArrayBuffer? More reliable than URL-based loading in browsers

    aeSession = await ort.InferenceSession.create(buffer, {
        executionProviders: ['wasm']  // Use WebAssembly backend
    });
    // Now aeSession can run the model on any input!
}
```

### Running Inference

```javascript
async function runInference(imageData) {
    // 1. Preprocess: canvas pixels → Float32Array → ONNX tensor
    const floatData = new Float32Array(28 * 28);
    for (let i = 0; i < 28 * 28; i++) {
        floatData[i] = imageData[i * 4] / 255.0;
        // Canvas gives RGBA (4 channels) — we only need R channel
        // Divide by 255 to normalize to [0, 1] like our training data
    }

    const inputTensor = new ort.Tensor('float32', floatData, [1, 1, 28, 28]);
    // Shape: [batch=1, channels=1, height=28, width=28]

    // 2. Run model
    const results = await aeSession.run({ input: inputTensor });
    // aeSession.run() executes the neural network in WebAssembly
    // Returns a dictionary of output tensors

    // 3. Extract output and display
    const output = results.output.data;  // Float32Array of 784 values
    displayOutputOnCanvas(output, outputCanvas);
}
```

### Latent Space Explorer

```javascript
latentCanvas.addEventListener('mousedown', async (e) => {
    // Map mouse position to latent coordinates [-3, 3]
    const z1 = (x / width) * 6 - 3;   // Mouse X → latent dim 1
    const z2 = (y / height) * 6 - 3;  // Mouse Y → latent dim 2

    const latentTensor = new ort.Tensor('float32', [z1, z2], [1, 2]);
    // Create a 2D latent vector from mouse position

    const results = await vaeDecoderSession.run({ input: latentTensor });
    // Run ONLY the decoder — generates a digit from the latent point
    // No need for the encoder here, we're creating from scratch!
});
```

---

## Quick Reference — What Each File Does

| File | Input | Output | Purpose |
|------|-------|--------|---------|
| `autoencoder.py` | Image (1,28,28) | Image (1,28,28) | Compress and reconstruct |
| `vae.py` | Image (1,28,28) | Image + μ + log σ² | Probabilistic compress/reconstruct |
| `data_loader.py` | Nothing | DataLoader objects | Download & serve MNIST |
| `visualize.py` | Model + data | PNG images | Create all plots |
| `train_ae.py` | MNIST data | Model + plots | Train the standard AE (50 epochs, LR scheduler, GPU-optimized) |
| `train_vae.py` | MNIST data | Model + plots + manifold | Train the VAE (50 epochs, KL=0.5, LR scheduler, GPU-optimized) |
| `export_onnx.py` | `.pth` weights | `.onnx` models | Convert for browser use |
| `demo/app.js` | User input (draw/upload) | Reconstructed digits | Run models in browser via ONNX Runtime Web |
