# 🧠 Autoencoder & Variational Autoencoder — PyTorch

> A complete, educational deep learning project: train neural networks that learn to **compress** and **reconstruct** handwritten digits, then explore what they've learned through an interactive browser demo.

<p align="center">
  <strong>🔵 Standard Autoencoder (AE)</strong> · <strong>🟣 Variational Autoencoder (VAE)</strong> · <strong>🌐 Interactive Web Demo</strong>
</p>

---

## ✨ What This Project Does

| Step | What Happens |
|------|-------------|
| **1. Train** | Two neural networks learn to compress 28×28 digit images into tiny latent vectors (16 numbers for AE, just 2 for VAE) |
| **2. Reconstruct** | From those few numbers, they rebuild the original image — compare how each model does it |
| **3. Generate** | The VAE can create brand-new digits by sampling random points in its learned 2D latent space |
| **4. Explore** | An interactive browser demo lets you draw digits and watch both models work in real-time |

### 🖥️ Full Training (Run on Your PC)

The real power is running training locally on your machine. You'll get:
- **High-quality trained models** using your GPU (CUDA) or CPU
- **11 output visualizations**: loss curves, reconstructions, latent space maps, generated samples, and a full 2D manifold traversal
- **Saved model checkpoints** (`.pth`) you can reload and experiment with
- Complete control over hyperparameters (epochs, latent dimensions, learning rate, etc.)

### 🌐 Online Demo (GitHub Pages)

https://sonesh-2202.github.io/AE-VAE_demo/demo/index.html 

The `demo/` folder contains a **zero-cost, client-side web demo** that runs entirely in the browser using ONNX Runtime Web. It's a lightweight preview — **download and train locally for the full experience.**

---

## 🚀 Quick Start

### Prerequisites

| Requirement | Version | Why |
|------------|---------|-----|
| **Python** | 3.8+ | Runtime |
| **PyTorch** | 2.0+ | Neural network framework |
| **torchvision** | (comes with PyTorch) | MNIST dataset + transforms |
| **matplotlib** | any | Plotting visualizations |
| **numpy** | any | Array operations |
| **scipy** | any | Latent space manifold generation |
| **NVIDIA GPU** | Optional but recommended | 10× faster training with CUDA |

### Installation

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/AE-VAE.git
cd AE-VAE

# Install all dependencies
pip install -r requirements.txt
```

### Train Both Models

```bash
# Train the Standard Autoencoder (50 epochs, ~7 min on GPU)
python train_ae.py

# Train the Variational Autoencoder (50 epochs, ~7 min on GPU)
python train_vae.py
```

Both scripts will:
1. Auto-download the MNIST dataset (first run only, ~12 MB)
2. Detect and use your GPU if available (falls back to CPU)
3. Train for 50 epochs with a learning rate scheduler
4. Save the best model checkpoint to `outputs/`
5. Generate all visualization plots to `outputs/`

---

## 📁 Project Structure

```
AE&VAE/
│
├── models/                          # Neural network architectures
│   ├── __init__.py
│   ├── autoencoder.py               # Standard Convolutional AE (141K params)
│   └── vae.py                       # Variational AE with reparameterization (200K params)
│
├── utils/                           # Helper modules
│   ├── __init__.py
│   ├── data_loader.py               # MNIST data loading & preprocessing
│   └── visualize.py                 # All plotting & visualization functions
│
├── demo/                            # Interactive browser demo (GitHub Pages)
│   ├── index.html                   # Main page — hero, demo, explorer, architecture
│   ├── style.css                    # Premium dark theme with glassmorphism
│   ├── app.js                       # ONNX model loading, canvas drawing, inference
│   └── models/                      # Pre-trained ONNX models for browser
│       ├── autoencoder.onnx         # Full AE model (~591 KB)
│       ├── ae_encoder.onnx          # AE encoder only (~289 KB)
│       ├── vae.onnx                 # Full VAE model (~815 KB)
│       └── vae_decoder.onnx         # VAE decoder only (~405 KB)
│
├── outputs/                         # Generated after training (not committed)
│   ├── ae_training_loss.png         # AE loss curve over epochs
│   ├── ae_reconstructions.png       # Original vs. reconstructed digits (AE)
│   ├── ae_latent_space.png          # 2D scatter of AE latent space
│   ├── ae_generated_samples.png     # Samples from random latent vectors
│   ├── autoencoder.pth              # Saved AE model weights (best checkpoint)
│   ├── vae_training_loss.png        # VAE loss breakdown (recon + KL)
│   ├── vae_reconstructions.png      # Original vs. reconstructed digits (VAE)
│   ├── vae_latent_space.png         # 2D VAE latent space colored by digit class
│   ├── vae_generated_samples.png    # New digits generated from N(0,I)
│   ├── vae_manifold.png             # 20×20 grid of decoded latent space
│   └── vae.pth                      # Saved VAE model weights (best checkpoint)
│
├── train_ae.py                      # Train the Standard Autoencoder
├── train_vae.py                     # Train the Variational Autoencoder
├── export_onnx.py                   # Convert trained models → ONNX for browser
├── requirements.txt                 # Python dependencies
│
├── README.md                        # ← You are here
├── AUTOENCODER_GUIDE.md             # Deep-dive theory & math guide
├── CODE_WALKTHROUGH.md              # Line-by-line code explanations
└── EXECUTION_GUIDE.md               # Step-by-step "how to run" guide
```

---

## 🤖 The Two Models

### 🔵 Standard Autoencoder (AE)

| Property | Value |
|----------|-------|
| **Parameters** | 141,329 trainable |
| **Latent Dim** | 16 |
| **Compression** | 784 → 16 (49:1 ratio) |
| **Loss** | MSE (Mean Squared Error) |
| **Encoder** | Conv2d(1→32) → Conv2d(32→64) → Linear(16) |
| **Decoder** | Linear(16) → ConvTranspose2d(64→32) → ConvTranspose2d(32→1) |

The AE creates a **deterministic** bottleneck — each input maps to exactly one point in latent space. Great for reconstruction, but random sampling produces noise.

### 🟣 Variational Autoencoder (VAE)

| Property | Value |
|----------|-------|
| **Parameters** | 200,197 trainable |
| **Latent Dim** | 2 |
| **Compression** | 784 → 2 (392:1 ratio) |
| **Loss** | BCE Reconstruction + β·KL Divergence |
| **KL Weight (β)** | 0.5 (prioritizes reconstruction quality) |
| **Encoder** | 3× Conv2d → fc_μ / fc_log_σ² |
| **Decoder** | Linear → 3× ConvTranspose2d |
| **Special** | Reparameterization Trick: z = μ + σ·ε |

The VAE learns a **probabilistic** latent space — it outputs a distribution (mean + variance), then samples from it. This makes the latent space smooth and continuous, allowing **generation of new digits** by sampling random points.

### AE vs VAE — Key Differences

| Aspect | Standard AE | VAE |
|--------|:-----------:|:---:|
| Latent space | Deterministic (one point) | Probabilistic (distribution) |
| Can generate new data? | ❌ Poorly | ✅ Yes — smooth latent space |
| Reconstruction quality | ★★★★★ Sharper | ★★★☆☆ Softer (2D tradeoff) |
| Loss function | MSE only | BCE + KL Divergence |
| Latent dim in this project | 16 | 2 |

---

## ⚙️ Training Configuration

Both scripts use optimized settings for quality results on your GPU:

| Parameter | `train_ae.py` | `train_vae.py` |
|-----------|:---:|:---:|
| **Epochs** | 50 | 50 |
| **Latent Dim** | 16 | 2 |
| **Learning Rate** | 1e-3 | 1e-3 |
| **LR Scheduler** | ReduceLROnPlateau (patience=5) | ReduceLROnPlateau (patience=7) |
| **Batch Size** | 128 | 128 |
| **KL Weight (β)** | — | 0.5 |
| **Best Model Saving** | ✅ Saves lowest test loss | ✅ Saves lowest test loss |
| **GPU Optimized** | ✅ cudnn.benchmark, non_blocking | ✅ cudnn.benchmark, non_blocking |

All hyperparameters are defined at the top of each script — edit them to experiment:

```python
# In train_ae.py or train_vae.py:
LATENT_DIM = 16          # Try 2, 8, 32, 64...
LEARNING_RATE = 1e-3     # Try 5e-4, 1e-4
BATCH_SIZE = 128         # Try 64, 256
EPOCHS = 50              # More = better, diminishing returns after ~80
KL_WEIGHT = 0.5          # VAE only: lower = sharper but less structured latent space
```

---

## 📊 Output Visualizations

After training, the `outputs/` folder will contain **11 files**:

### Standard Autoencoder Outputs

| File | What It Shows |
|------|--------------|
| `ae_training_loss.png` | MSE loss curve decreasing over 50 epochs |
| `ae_reconstructions.png` | Top row: real MNIST digits → Bottom row: AE's reconstruction |
| `ae_latent_space.png` | First 2 of 16 latent dimensions plotted as scatter, colored by digit class |
| `ae_generated_samples.png` | 8×8 grid of images decoded from random latent vectors |
| `autoencoder.pth` | Best model checkpoint (lowest test loss) — reload with `model.load_state_dict()` |

### Variational Autoencoder Outputs

| File | What It Shows |
|------|--------------|
| `vae_training_loss.png` | Three-line plot: total loss, reconstruction (BCE), and KL divergence |
| `vae_reconstructions.png` | Top row: real digits → Bottom row: VAE reconstruction (softer due to 2D compression) |
| `vae_latent_space.png` | Beautiful 2D scatter where each digit class naturally clusters |
| `vae_generated_samples.png` | New digits generated by sampling z ~ N(0, I) and decoding |
| `vae_manifold.png` | 20×20 grid traversing latent space — digits morph continuously between classes |
| `vae.pth` | Best model checkpoint (lowest test loss) |

---

## 🌐 Interactive Web Demo

The `demo/` folder is a **standalone static website** that runs trained models directly in your browser using [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/). Zero server needed.

### Features
- ✏️ **Draw a digit** on the canvas and see AE vs VAE reconstruct it
- 📁 **Upload an image** to pass through both models
- 🔮 **Explore the VAE latent space** — click/drag on a 2D grid to generate digits
- 📐 **Architecture diagrams** showing both model structures
- 📱 **Fully responsive** — works on desktop and mobile

### Run Locally

```bash
python -m http.server 8080 --directory demo
# Open http://localhost:8080 in your browser
```

### Deploy to GitHub Pages

1. Push your repo to GitHub
2. Go to **Settings → Pages → Source → Deploy from a branch**
3. Set the branch and folder to `/demo`
4. Your live URL: `https://yourusername.github.io/repo-name/`

### Re-export Models After Retraining

If you retrain with different settings, update the browser models:

```bash
python export_onnx.py
```

This converts your `.pth` checkpoints to `.onnx` files in `demo/models/`.

> ⚠️ **Note**: The web demo is a lightweight preview. Models are small (~2 MB total) and limited to MNIST digits. **Download and run locally** for the full training experience with GPU acceleration and all 11 visualizations.

---

## 📚 Additional Documentation

| Document | What's Inside |
|----------|--------------|
| **[AUTOENCODER_GUIDE.md](AUTOENCODER_GUIDE.md)** | Complete theory & math: encoder/decoder architecture, MSE loss, BCE loss, KL Divergence derivation, reparameterization trick, β-VAE, latent space properties, real-world applications, hyperparameter tuning, common pitfalls |
| **[CODE_WALKTHROUGH.md](CODE_WALKTHROUGH.md)** | Line-by-line explanation of every Python file: every import, class, function, and significant line annotated with comments on what it does and why |
| **[EXECUTION_GUIDE.md](EXECUTION_GUIDE.md)** | Step-by-step guide: prerequisites, installation, running training, understanding outputs, customizing hyperparameters, loading saved models, troubleshooting common issues |

---

## 🛠️ Common Issues & Fixes

| Issue | Fix |
|-------|-----|
| **CUDA out of memory** | Reduce `BATCH_SIZE` to 64 or 32 in the training script |
| **No GPU detected** | Install CUDA-enabled PyTorch: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121` |
| **Training is slow on CPU** | Expected — GPU gives ~10× speedup. Reduce `EPOCHS` to 10 for a quick test |
| **ModuleNotFoundError** | Run `pip install -r requirements.txt` — you're missing a dependency |
| **Plots don't appear** | They're saved to `outputs/`, not displayed. Open the PNG files there |
| **ONNX export fails** | Make sure you've trained both models first (`outputs/autoencoder.pth` and `outputs/vae.pth` must exist) |

---

## 📖 How to Learn From This Project

**If you're new to autoencoders**, here's the recommended reading order:

1. **[AUTOENCODER_GUIDE.md](AUTOENCODER_GUIDE.md)** — Understand the theory first
2. **[CODE_WALKTHROUGH.md](CODE_WALKTHROUGH.md)** — See how theory maps to code
3. **[EXECUTION_GUIDE.md](EXECUTION_GUIDE.md)** — Run it yourself and see the results
4. **Experiment!** — Change hyperparameters, try different latent dims, modify the architecture

---

## 🙏 Credits & References

- **Dataset**: [MNIST](http://yann.lecun.com/exdb/mnist/) — 70,000 handwritten digit images by Yann LeCun
- **Framework**: [PyTorch](https://pytorch.org/) — Open source ML framework by Meta AI
- **Web Runtime**: [ONNX Runtime Web](https://onnxruntime.ai/) — Neural network inference in the browser by Microsoft
- **Original VAE Paper**: [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) — Kingma & Welling, 2013

---

<p align="center">
  Built with ❤️ and PyTorch · <a href="AUTOENCODER_GUIDE.md">Read the Guide</a> · <a href="demo/">Try the Demo</a>
</p>
