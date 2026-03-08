# How to Run This Project — Step-by-Step Execution Guide

> Follow these steps **exactly** and you'll have fully trained autoencoders with beautiful output visualizations and an interactive browser demo.

---

## Prerequisites

Before you start, make sure you have these installed:

| Software | Version | Check Command |
|----------|---------|---------------|
| **Python** | 3.8 or higher | `python --version` |
| **pip** | Any recent | `pip --version` |
| **GPU (optional)** | NVIDIA with CUDA | `python -c "import torch; print(torch.cuda.is_available())"` |

> **No GPU?** No problem! Everything works on CPU too — it just takes longer (~15 min per model instead of ~7 min).

---

## Step 1: Open Terminal in the Project Folder

### On Windows
1. Open **File Explorer**
2. Navigate to `Documents > Antigravity Project > AE&VAE`
3. Click the address bar, type `cmd` or `powershell`, press **Enter**

Or use VS Code:
1. Open the `AE&VAE` folder in VS Code
2. Press `` Ctrl + ` `` to open the integrated terminal

### On Mac/Linux
```bash
cd ~/path/to/AE\&VAE
```

---

## Step 2: Install Dependencies

Run this command **once** — it downloads all required libraries:

```bash
pip install -r requirements.txt
```

**What gets installed:**

| Package | What It Does |
|---------|-------------|
| `torch` | PyTorch — the deep learning framework |
| `torchvision` | Provides the MNIST dataset + image utilities |
| `matplotlib` | Creates all the plots and visualizations |
| `numpy` | Numerical operations on arrays |
| `scipy` | Scientific computing (used for the manifold grid) |

### Verify Installation

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

**Expected output:**
```
PyTorch 2.x.x
CUDA: True    ← (or False if no GPU — that's fine!)
```

### For GPU Users (Optional — Much Faster)

If you have an NVIDIA GPU but `CUDA: False`, install the CUDA-enabled PyTorch:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## Step 3: Train the Standard Autoencoder

```bash
python train_ae.py
```

### What You'll See

```
============================================================
  Standard Autoencoder — MNIST Training
============================================================
  Device      : cuda
  GPU         : NVIDIA GeForce RTX 3060      ← Your GPU name
  VRAM        : 12.0 GB                      ← Your GPU memory
  Latent Dim  : 16
  Batch Size  : 128
  Epochs      : 50
  LR          : 0.001
============================================================

📁 Loading MNIST dataset...
   Train batches: 469
   Test batches:  79

🧠 Model Parameters: 141,329 total, 141,329 trainable

🚀 Starting training...

  Epoch [01/50]  Train: 0.048312  Test: 0.027845  LR: 1.0e-03  Time: 8.1s ★
  Epoch [02/50]  Train: 0.024123  Test: 0.020156  LR: 1.0e-03  Time: 7.9s ★
  ...
  Epoch [50/50]  Train: 0.005234  Test: 0.005134  LR: 5.0e-04  Time: 8.2s

💾 Best model loaded (test loss: 0.005001)

📊 Generating visualizations...
  ✓ Loss plot saved to outputs/ae_training_loss.png
  ✓ Reconstruction plot saved to outputs/ae_reconstructions.png
  ✓ Latent space plot saved to outputs/ae_latent_space.png
  ✓ Generated samples saved to outputs/ae_generated_samples.png

✅ All done! Check the 'outputs/' directory for results.
```

**What the symbols mean:**
- `★` = New best model saved (lowest test loss so far)
- `LR` = Current learning rate (automatically decreases over time)

### How Long Does It Take?

| Hardware | 50 Epochs |
|----------|-----------|
| NVIDIA GPU (any) | 5–8 minutes |
| CPU only | 15–30 minutes |

---

## Step 4: Train the Variational Autoencoder (VAE)

```bash
python train_vae.py
```

### What You'll See

```
============================================================
  Variational Autoencoder (VAE) — MNIST Training
============================================================
  Device      : cuda
  GPU         : NVIDIA GeForce RTX 3060
  VRAM        : 12.0 GB
  Latent Dim  : 2
  Batch Size  : 128
  Epochs      : 50
  LR          : 0.001
  KL Weight β : 0.5
============================================================

🧠 Model Parameters: 200,197 total, 200,197 trainable

🚀 Starting training...

  Epoch [01/50]  Total: 185.42  Recon: 181.12  KL: 8.60  Test: 170.53  LR: 1.0e-03  Time: 8.4s ★
  ...
  Epoch [50/50]  Total: 103.85  Recon: 97.11  KL: 13.48  Test: 104.12  LR: 5.0e-04  Time: 8.2s

💾 Best model loaded (test loss: 103.22)

📊 Generating visualizations...
🎨 Generating latent space grid (2D manifold)...

✅ All done!
```

**VAE loss breakdown:**
- **Total** = Recon + β·KL (the combined loss)
- **Recon** = How well it reconstructs the input (lower = sharper images)
- **KL** = How close the latent space is to N(0,I) (higher = more structured for generation)
- **β = 0.5** means we prioritize reconstruction quality over perfect latent structure

---

## Step 5: View Your Results

### Where Are the Files?

All outputs are saved in the `outputs/` folder:

```
outputs/
├── ae_training_loss.png       ← Loss curve
├── ae_reconstructions.png     ← Original vs rebuilt digits
├── ae_latent_space.png        ← 2D scatter of latent codes
├── ae_generated_samples.png   ← Random samples from the AE
├── autoencoder.pth            ← Best AE model weights
├── vae_training_loss.png      ← VAE loss with Recon + KL breakdown
├── vae_reconstructions.png    ← VAE original vs rebuilt digits
├── vae_latent_space.png       ← 2D latent space colored by digit
├── vae_generated_samples.png  ← New digits from random sampling
├── vae_manifold.png           ← 2D grid showing digit transitions
└── vae.pth                    ← Best VAE model weights
```

### How to Open the Images

**Option A — File Explorer:**
Navigate to `AE&VAE/outputs/` and double-click any `.png` file.

**Option B — VS Code:**
Click any `.png` file in the Explorer sidebar — it opens directly in VS Code.

---

## Step 6: Understanding Each Output

### `ae_training_loss.png` / `vae_training_loss.png`
**What it shows:** How the loss decreases over training epochs.
**What to look for:**
- ✅ Loss goes **down** smoothly → training is working
- ❌ Loss goes **up** or oscillates wildly → learning rate too high
- ❌ Loss **plateaus** very early → model might be too small

### `ae_reconstructions.png` / `vae_reconstructions.png`
**What it shows:** Top row = original MNIST digits, Bottom row = model's reconstruction.
**What to look for:**
- ✅ Bottom row looks like the top row → model learned to compress and reconstruct
- ❌ Bottom row is all blurry → train longer or increase latent dim

### `ae_latent_space.png` / `vae_latent_space.png`
**What it shows:** Each dot is one MNIST image plotted at its latent space position, colored by digit (0-9).
**What to look for:**
- ✅ Same-colored dots cluster together → model learned meaningful features
- ✅ (VAE) Clusters are round and well-separated → KL regularization is working
- ❌ All dots are on top of each other → latent dim might be too small

### `vae_generated_samples.png`
**What it shows:** Brand new digits that the model invented by sampling random latent vectors.
**What to look for:**
- ✅ They look like real digits → the VAE has learned the data distribution
- ❌ They look like noise → train longer or adjust KL weight

### `vae_manifold.png`
**What it shows:** A 20×20 grid where each position maps to a different point in the 2D latent space.
**What to look for:**
- ✅ Digits smoothly morph into each other → the latent space is continuous
- ✅ Different digits appear in different regions → the space is organized

---

## Step 7: Run the Interactive Web Demo

After training, you can launch a browser-based demo:

### Export Trained Models to ONNX

```bash
python export_onnx.py
```

This converts your `.pth` checkpoints to `.onnx` files in `demo/models/`:
```
✓ autoencoder.onnx (591 KB)
✓ ae_encoder.onnx (289 KB)
✓ vae.onnx (815 KB)
✓ vae_decoder.onnx (405 KB)
```

### Launch the Demo Server

```bash
python -m http.server 8080 --directory demo
```

Open **http://localhost:8080** in your browser.

### What the Demo Includes

| Feature | Description |
|---------|-------------|
| ✏️ Draw a digit | Draw on the canvas → see AE and VAE reconstruct it |
| 📁 Upload an image | Load any image → both models process it |
| 🔮 Latent space | Click on a 2D grid → generate new digits from the VAE |
| 📐 Architecture | Diagrams showing both model architectures |

> **Note**: The web demo runs models in your browser using ONNX Runtime Web. These are the same models you trained — no server needed!

---

## Step 8: Customizing the Training

### Change Hyperparameters

Edit the top of `train_ae.py` or `train_vae.py`:

```python
# In train_ae.py:
LATENT_DIM = 32      # Try 2, 8, 32, 64, 128
LEARNING_RATE = 5e-4  # Try 1e-4, 5e-4, 1e-3, 5e-3
BATCH_SIZE = 256      # Try 64, 128, 256, 512
EPOCHS = 100          # More epochs = better results

# In train_vae.py (additional):
KL_WEIGHT = 0.5       # Try 0.1, 0.5, 1.0, 2.0, 5.0
```

### Experiments to Try

| Experiment | What to Change | What You'll Learn |
|-----------|----------------|-------------------|
| **Shallow latent** | `LATENT_DIM = 2` in AE | How compression affects quality |
| **Deep latent** | `LATENT_DIM = 128` in AE | Diminishing returns of more capacity |
| **β-VAE** | `KL_WEIGHT = 5.0` in VAE | How β affects disentanglement |
| **Long training** | `EPOCHS = 100` | How much quality improves over time |
| **Small batches** | `BATCH_SIZE = 32` | Noisier gradients, sometimes finds better minima |

---

## Step 9: Loading a Saved Model

After training, you can reload your model without retraining:

```python
import torch
from models.autoencoder import Autoencoder
from models.vae import VAE

# Load Standard AE
ae = Autoencoder(latent_dim=16)
ae.load_state_dict(torch.load("outputs/autoencoder.pth", map_location="cpu", weights_only=True))
ae.eval()

# Load VAE
vae = VAE(latent_dim=2)
vae.load_state_dict(torch.load("outputs/vae.pth", map_location="cpu", weights_only=True))
vae.eval()

# Use them:
sample_image = torch.randn(1, 1, 28, 28)

with torch.no_grad():
    ae_output = ae(sample_image)                    # AE reconstruction
    vae_output, mu, log_var = vae(sample_image)     # VAE reconstruction

    # Generate new images from VAE:
    z = torch.randn(10, 2)          # 10 random 2D latent vectors
    generated = vae.decode(z)       # Decode them into digit images!
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: No module named 'torch'` | Run `pip install -r requirements.txt` |
| `CUDA out of memory` | Reduce `BATCH_SIZE` to 64 or 32 |
| `FileNotFoundError: ./data/MNIST/...` | Ensure internet access on first run (auto-downloads MNIST) |
| Very slow training (no GPU) | Normal on CPU. Reduce `EPOCHS` to 10 for a quick test |
| All generated images look the same | Train for more epochs, or reduce `KL_WEIGHT` |
| Web demo says "Failed to load models" | Run `python export_onnx.py` first — ONNX files must exist in `demo/models/` |
| `No GPU detected` | Install CUDA PyTorch: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121` |
