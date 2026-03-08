# The Complete Guide to Autoencoders & Variational Autoencoders

> A deep-dive from first principles to working code. By the end of this guide you will understand **what** autoencoders are, **why** they work, the **math** behind them, and **how** to build them in PyTorch.

---

## Table of Contents

1. [What Is an Autoencoder?](#1-what-is-an-autoencoder)
2. [Why Do We Need Autoencoders?](#2-why-do-we-need-autoencoders)
3. [Architecture Deep-Dive](#3-architecture-deep-dive)
4. [The Mathematics](#4-the-mathematics)
5. [Standard Autoencoder — Code Walkthrough](#5-standard-autoencoder--code-walkthrough)
6. [From AE to VAE — The Key Insight](#6-from-ae-to-vae--the-key-insight)
7. [Variational Autoencoder — Theory](#7-variational-autoencoder--theory)
8. [The Reparameterization Trick](#8-the-reparameterization-trick)
9. [KL Divergence — Derivation & Intuition](#9-kl-divergence--derivation--intuition)
10. [VAE — Code Walkthrough](#10-vae--code-walkthrough)
11. [Training Pipeline Explained](#11-training-pipeline-explained)
12. [Hyperparameter Tuning Guide](#12-hyperparameter-tuning-guide)
13. [AE vs. VAE — Comparison](#13-ae-vs-vae--comparison)
14. [Real-World Applications](#14-real-world-applications)
15. [Common Pitfalls & Debugging](#15-common-pitfalls--debugging)
16. [Further Reading & References](#16-further-reading--references)
17. [Verified Training Results](#17-verified-training-results)

---

## 1. What Is an Autoencoder?

An **autoencoder** is a neural network that learns to **copy its input to its output** — but with a twist. It is forced to learn a **compressed representation** (called the *latent space* or *bottleneck*) along the way.

```
┌──────────┐      ┌──────────┐      ┌──────────┐
│          │      │          │      │          │
│  INPUT   │─────▶│  LATENT  │─────▶│  OUTPUT  │
│  (x)     │      │  (z)     │      │  (x̂)    │
│  784 dim │      │  16 dim  │      │  784 dim │
│          │      │          │      │          │
└──────────┘      └──────────┘      └──────────┘
    Encoder          Bottleneck         Decoder
   (compress)      (representation)    (reconstruct)
```

### The Core Idea

The network has two halves:

| Component | Job | Analogy |
|-----------|-----|---------|
| **Encoder** | Compress the input into a small latent vector `z` | A summary or zip file |
| **Decoder** | Reconstruct the original input from `z` | Unzipping / expanding |

Because the latent vector `z` is much **smaller** than the input, the network is forced to learn only the **most important features** needed to reconstruct the data. Noise and irrelevant details are discarded.

### Key Terminology

| Term | Definition |
|------|-----------|
| **Latent Space** | The compressed, low-dimensional representation learned by the encoder |
| **Latent Vector (z)** | A single point in the latent space representing one input |
| **Bottleneck** | The narrowest layer — forces compression |
| **Reconstruction** | The decoder's output — ideally identical to the input |
| **Reconstruction Error** | The difference between input and output (what we minimize) |

---

## 2. Why Do We Need Autoencoders?

Autoencoders are versatile tools in the ML toolkit. Here's why they matter:

### Dimensionality Reduction
Like PCA but **nonlinear**. A 784-pixel MNIST image can be represented by just 2–16 numbers while retaining enough information to reconstruct it.

### Feature Learning
The latent space captures **meaningful features** of the data. For digits, it might learn things like stroke thickness, tilt angle, and overall shape — without being told to.

### Data Denoising
Train on noisy inputs, reconstruct clean outputs. The bottleneck forces the model to learn the true signal and ignore the noise.

### Anomaly Detection
If you train on "normal" data, anomalies will have **high reconstruction error** because the model has never learned to represent them.

### Generative Modeling (VAE)
Variational autoencoders can **generate entirely new data** by sampling from the learned latent distribution.

---

## 3. Architecture Deep-Dive

Our implementation uses **convolutional** layers, which are far more effective for image data than simple fully-connected layers.

### Encoder Architecture

```
Input Image: (1, 28, 28)
         │
         ▼
┌─────────────────────────────────┐
│  Conv2d(1→32, 3×3, stride=2)   │  Reduces spatial size by half
│  BatchNorm2d(32)                │  Stabilizes training
│  ReLU                           │  Non-linear activation
├─────────────────────────────────┤
│  Output: (32, 14, 14)          │
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  Conv2d(32→64, 3×3, stride=2)  │  Reduces spatial size again
│  BatchNorm2d(64)                │
│  ReLU                           │
├─────────────────────────────────┤
│  Output: (64, 7, 7)            │
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  Flatten → (3136,)             │
│  Linear(3136 → latent_dim)     │  Final compression
├─────────────────────────────────┤
│  Output: (latent_dim,)         │  e.g., (16,)
└─────────────────────────────────┘
```

**Why stride=2 instead of max-pooling?** Strided convolutions let the network *learn* how to downsample, rather than using a fixed rule (taking the max). This generally produces better results.

**Why BatchNorm?** It normalizes activations between layers, which:
- Speeds up training (allows higher learning rates)
- Reduces sensitivity to weight initialization
- Acts as a mild regularizer

### Decoder Architecture (Mirror of Encoder)

```
Latent Vector: (latent_dim,)
         │
         ▼
┌──────────────────────────────────────────────┐
│  Linear(latent_dim → 3136)                   │
│  ReLU                                        │
│  Unflatten → (64, 7, 7)                     │
└──────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────┐
│  ConvTranspose2d(64→32, 3×3, stride=2)       │  Upsamples
│  BatchNorm2d(32)                             │
│  ReLU                                        │
├──────────────────────────────────────────────┤
│  Output: (32, 14, 14)                        │
└──────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────┐
│  ConvTranspose2d(32→1, 3×3, stride=2)        │  Final upsample
│  Sigmoid                                     │  Pixel values → [0, 1]
├──────────────────────────────────────────────┤
│  Output: (1, 28, 28)                         │
└──────────────────────────────────────────────┘
```

**Why Sigmoid at the end?** Our input pixels are in [0, 1], so the decoder output must also be in that range. Sigmoid squashes any value to (0, 1).

**What is ConvTranspose2d?** It's the "reverse" of Conv2d. While Conv2d reduces spatial dimensions, ConvTranspose2d *increases* them — essentially "un-convolving" the representation back to image size.

---

## 4. The Mathematics

### Objective Function

The autoencoder minimizes the **reconstruction error** between input `x` and output `x̂`:

```
L(θ, φ) = ||x - x̂||²
```

Where:
- `θ` = encoder parameters
- `φ` = decoder parameters
- `x̂ = Decoder(Encoder(x))`

### Mean Squared Error (MSE)

For our standard autoencoder, we use MSE loss:

```
MSE = (1/n) · Σᵢ (xᵢ - x̂ᵢ)²
```

This treats each pixel independently and penalizes large errors quadratically.

### Binary Cross-Entropy (BCE)

For the VAE (with Sigmoid output), BCE is more appropriate:

```
BCE = -(1/n) · Σᵢ [xᵢ · log(x̂ᵢ) + (1 - xᵢ) · log(1 - x̂ᵢ)]
```

**Why BCE for VAE?** When the output is squashed by Sigmoid, BCE provides better gradients than MSE. It interprets each pixel as a Bernoulli probability — "what's the probability this pixel is white?"

---

## 5. Standard Autoencoder — Code Walkthrough

### The Encoder Class

```python
class Encoder(nn.Module):
    def __init__(self, latent_dim: int = 16):
        super().__init__()
        self.conv_layers = nn.Sequential(
            # (1, 28, 28) → (32, 14, 14): Learn 32 spatial features
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # (32, 14, 14) → (64, 7, 7): Learn higher-level features
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.flatten = nn.Flatten()               # (64, 7, 7) → (3136,)
        self.fc = nn.Linear(64 * 7 * 7, latent_dim)  # (3136,) → (latent_dim,)
```

**What's happening at each layer:**
1. **First Conv layer**: Detects low-level features (edges, corners). Stride=2 halves resolution.
2. **Second Conv layer**: Combines low-level features into higher-level patterns (loops, strokes).
3. **Flatten + Linear**: Compresses the entire feature map into the final latent vector.

### The Decoder Class

```python
class Decoder(nn.Module):
    def __init__(self, latent_dim: int = 16):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 64 * 7 * 7)  # Expand
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(64, 7, 7))
        self.conv_layers = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),  # Output in [0, 1]
        )
```

The decoder is a **mirror image** of the encoder — it reverses every compression step.

### Forward Pass

```python
class Autoencoder(nn.Module):
    def forward(self, x):
        z = self.encode(x)   # x → z  (compress)
        x_hat = self.decode(z)  # z → x̂ (reconstruct)
        return x_hat
```

That's really all there is to it. The magic happens through training.

---

## 6. From AE to VAE — The Key Insight

### The Problem with Standard AE

A standard autoencoder maps each input to a **specific point** in latent space. This creates two problems:

1. **Gaps in the latent space**: If you sample a random point, it probably falls in a "dead zone" and decodes to garbage.
2. **No structure**: Similar inputs might not map to nearby points.

```
Standard AE Latent Space:           VAE Latent Space:
    ·                                  ☁️☁️☁️
  ·    ·                             ☁️☁️☁️☁️☁️
    ·      ·                        ☁️☁️☁️☁️☁️☁️
  ·                                  ☁️☁️☁️☁️☁️
        ·                              ☁️☁️☁️
  (scattered points,                (smooth, continuous
   no structure)                     distribution)
```

### The VAE Solution

Instead of mapping each input to a **point**, the VAE maps it to a **probability distribution** (specifically, a Gaussian). This means:

- Each input produces a **mean (μ)** and **variance (σ²)** in latent space
- We **sample** `z` from this distribution: `z ~ N(μ, σ²)`
- The distribution is regularized to be close to `N(0, I)`

This creates a smooth, continuous latent space where:
- Similar inputs cluster together
- Random samples decode to meaningful outputs
- You can interpolate between any two points and get valid outputs

---

## 7. Variational Autoencoder — Theory

### The Probabilistic Framework

The VAE is grounded in Bayesian inference. We want to model:

```
p(x) = ∫ p(x|z) · p(z) dz
```

Where:
- `p(z)` is the **prior** — we choose `N(0, I)` (standard normal)
- `p(x|z)` is the **likelihood** — the decoder, which generates data given a latent code
- `p(z|x)` is the **posterior** — the "true" encoder, which is intractable

Since we can't compute `p(z|x)` directly, we approximate it with a learned distribution `q(z|x)` — that's our encoder network.

### The ELBO (Evidence Lower Bound)

The VAE maximizes the ELBO:

```
ELBO = E_q[log p(x|z)] - KL(q(z|x) || p(z))
       ├─────────────┘   ├──────────────────┘
       Reconstruction     KL Divergence
       (how good is       (how close is q
        the decoder?)      to the prior?)
```

Equivalently, the **loss** we minimize is:

```
Loss = -ELBO = Reconstruction Loss + KL Divergence
```

This is an elegant balance:
- **Reconstruction Loss** pushes the model to encode useful information
- **KL Divergence** pushes the model toward a regular, well-structured latent space

---

## 8. The Reparameterization Trick

### The Problem

We need to sample `z ~ N(μ, σ²)` during the forward pass. But sampling is a **random, non-differentiable** operation — gradients can't flow through it!

### The Solution

Instead of sampling `z` directly, we rewrite it:

```
z = μ + σ · ε,    where ε ~ N(0, I)
```

Now:
- `ε` is external random noise (no gradients needed)
- `μ` and `σ` are deterministic outputs of the encoder (gradients flow through them!)

```
                 BEFORE (can't backprop):
                 z ~ N(μ, σ²)     ← sampling blocks gradients

                 AFTER (reparameterization):
                 ε ~ N(0, 1)      ← fixed random noise
                 z = μ + σ · ε    ← deterministic function of μ, σ
                                     ✓ gradients flow through μ and σ!
```

### In Code

```python
def reparameterize(self, mu, log_var):
    std = torch.exp(0.5 * log_var)   # σ = exp(0.5 · log σ²)
    eps = torch.randn_like(std)       # ε ~ N(0, I)
    z = mu + std * eps                # z = μ + σ · ε
    return z
```

**Why `log_var` instead of `var`?** Because variance σ² must be positive, but neural network outputs can be anything. By predicting `log σ²` instead, we avoid the need for a positivity constraint — `exp(log σ²)` is always positive.

---

## 9. KL Divergence — Derivation & Intuition

### What Is KL Divergence?

KL Divergence measures how "different" one probability distribution is from another:

```
KL(q || p) = ∫ q(z) · log(q(z)/p(z)) dz
```

- KL = 0 means q and p are identical
- KL > 0 means q is different from p (always non-negative)

### Intuition

Think of it as a **penalty** for the encoder's distribution `q(z|x)` being different from the prior `p(z) = N(0, I)`. This penalty:

- **Prevents the encoder from cheating** by mapping each input to a tiny, distinct point
- **Ensures the latent space is smooth** — nearby points decode to similar outputs
- **Enables generation** — we can sample from `N(0, I)` and get meaningful outputs

### Closed-Form for Gaussians

When both `q` and `p` are Gaussians, KL has a beautiful closed-form:

```
q(z|x) = N(μ, σ²)
p(z)   = N(0, 1)

KL(q || p) = -½ · Σⱼ (1 + log σⱼ² - μⱼ² - σⱼ²)
```

**Step-by-step derivation:**

```
KL = ∫ q(z) log(q(z)/p(z)) dz

   = ∫ q(z) [log q(z) - log p(z)] dz

   = E_q[log q(z)] - E_q[log p(z)]

For a single dimension:
   E_q[log q(z)] = -½ log(2π) - ½ log σ² - ½
   E_q[log p(z)] = -½ log(2π) - ½(μ² + σ²)

   KL = [-½ log σ² - ½] - [-½(μ² + σ²)]
      = -½ log σ² - ½ + ½μ² + ½σ²
      = -½ (1 + log σ² - μ² - σ²)

Summing over all latent dimensions j:
   KL = -½ · Σⱼ (1 + log σⱼ² - μⱼ² - σⱼ²)
```

### In Code

```python
kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / batch_size
```

This single line encodes the entire derivation above!

---

## 10. VAE — Code Walkthrough

### Encoder — Two Heads

Unlike the standard AE encoder (one output), the VAE encoder has **two heads**:

```python
self.fc_mu = nn.Linear(128 * 4 * 4, latent_dim)       # → μ
self.fc_log_var = nn.Linear(128 * 4 * 4, latent_dim)  # → log σ²
```

Both share the same convolutional backbone but output different things:
- `fc_mu` predicts where the latent code **should be** (center of the distribution)
- `fc_log_var` predicts how **spread out** the distribution is

### The Full Forward Pass

```python
def forward(self, x):
    # 1. Encode → get distribution parameters
    mu, log_var = self.encode(x)

    # 2. Sample z using the reparameterization trick
    z = self.reparameterize(mu, log_var)

    # 3. Decode → reconstruct the input
    x_hat = self.decode(z)

    return x_hat, mu, log_var  # Return all three for the loss
```

### The Loss Function

```python
@staticmethod
def loss_function(x_hat, x, mu, log_var, kl_weight=1.0):
    # Reconstruction: "How well did we rebuild the input?"
    recon_loss = F.binary_cross_entropy(x_hat, x, reduction="sum") / x.size(0)

    # KL Divergence: "How close is the learned distribution to N(0,I)?"
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / x.size(0)

    # Combined loss
    total_loss = recon_loss + kl_weight * kl_loss
    return total_loss, recon_loss, kl_loss
```

The `kl_weight` parameter (β) lets you control the trade-off:
- **β = 1**: Standard VAE — good reconstructions, structured latent space
- **β > 1**: β-VAE — more disentangled latent space, slightly blurrier reconstructions
- **β < 1**: Prioritize reconstruction quality over latent space regularity

> **In this project, we use β = 0.5** with `latent_dim = 2`. Since we're compressing 784 pixels into just 2 numbers (392:1 ratio), lowering β lets the model focus on reconstruction quality rather than forcing perfect Gaussian structure. This produces sharper, more recognizable digit outputs.

---

## 11. Training Pipeline Explained

### Data Flow

```
MNIST Image → [0,1] normalization → DataLoader → Model → Loss → Backprop
```

### Training Loop (Pseudocode)

```python
for epoch in range(EPOCHS):
    for images in train_loader:

        # Forward pass
        if is_vae:
            x_hat, mu, log_var = model(images)
            loss = vae_loss(x_hat, images, mu, log_var)
        else:
            x_hat = model(images)
            loss = mse_loss(x_hat, images)

        # Backward pass
        optimizer.zero_grad()   # Clear old gradients
        loss.backward()         # Compute new gradients
        optimizer.step()        # Update weights
```

### What Happens During Training?

| Epoch Range | What the Model Learns |
|-------------|----------------------|
| 1–3 | Basic structure — outputs look like blurry blobs |
| 4–8 | Digit shapes start emerging — the model learns stroke patterns |
| 9–15 | Fine details — digits become recognizable and sharp |
| 15–20 | Refinement — minor improvements, details get crisper |

### Optimizer: Adam

We use the **Adam** optimizer because:
- It adapts the learning rate per-parameter
- It uses momentum (exponential moving averages of gradients)
- It converges faster than basic SGD for most deep learning tasks

---

## 12. Hyperparameter Tuning Guide

### Key Hyperparameters

| Parameter | Our Value | Range to Try | Effect |
|-----------|-----------|-------------|--------|
| **Latent Dim** | AE: 16, VAE: 2 | 2–128 | Lower = more compression, higher = more capacity |
| **Learning Rate** | 1e-3 | 1e-4 to 1e-2 | Too high = unstable, too low = slow |
| **Batch Size** | 128 | 32–512 | Larger = more stable gradients, needs more RAM |
| **Epochs** | 20 | 10–100 | More = better, but diminishing returns |
| **KL Weight (β)** | 1.0 | 0.1–10 | Higher = more structured latent space (VAE) |

### Latent Dimension Trade-Off

```
Small (2–4):
  ✓ Easy to visualize in 2D
  ✓ Forces maximum compression
  ✗ May lose detail (blurry outputs)

Medium (8–32):
  ✓ Good balance of compression and quality
  ✓ Sweet spot for most tasks

Large (64–256):
  ✓ Minimal information loss
  ✗ Less useful as a compressed representation
  ✗ May not learn meaningful features
```

### Tips for Better Results

1. **Learning Rate Scheduling**: Reduce LR when loss plateaus
   ```python
   scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
   ```

2. **KL Annealing (VAE)**: Start with β=0 and gradually increase to 1 over training. This prevents KL collapse (where the model ignores the latent space).

3. **Gradient Clipping**: Prevents exploding gradients
   ```python
   torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
   ```

---

## 13. AE vs. VAE — Comparison

| Feature | Standard AE | VAE |
|---------|-------------|-----|
| **Latent Space** | Deterministic points | Probability distributions |
| **Loss Function** | MSE only | Reconstruction + KL Divergence |
| **Generation** | Poor (gaps in latent space) | Excellent (smooth, continuous) |
| **Interpolation** | May produce artifacts | Smooth transitions |
| **Training** | Simpler, faster | More complex, needs balancing |
| **Reconstruction Quality** | Generally sharper | Slightly blurrier |
| **Latent Space Structure** | Unordered | Organized, meaningful |

### When to Use Which?

- **Use Standard AE** for: dimensionality reduction, feature extraction, denoising, anomaly detection
- **Use VAE** for: data generation, latent space exploration, creative applications, learning disentangled representations

---

## 14. Real-World Applications

### 1. Image Denoising
```
Noisy Image → Encoder → Latent (clean features) → Decoder → Clean Image
```
Train with clean targets and noisy inputs. The bottleneck filters out noise.

### 2. Anomaly Detection
```
Normal data → Train AE → Low reconstruction error
Anomalous data → Trained AE → HIGH reconstruction error ← Flag!
```
Used in manufacturing (defect detection), cybersecurity (intrusion detection), and medical imaging.

### 3. Data Compression
Reduce high-dimensional data to a compact latent representation. Used in recommendation systems and search engines.

### 4. Drug Discovery (VAE)
Encode molecular structures into latent space. Navigate the latent space to find new molecules with desired properties.

### 5. Face Generation & Editing (VAE)
Learn a disentangled latent space where each dimension controls a facial feature (age, smile, glasses). Change one dimension to edit the face.

### 6. Music & Audio Generation
Autoencoders like Google's MusicVAE learn latent representations of musical sequences and can interpolate between melodies.

---

## 15. Common Pitfalls & Debugging

### Pitfall 1: Blurry VAE Outputs
**Symptom**: Generated images are blurry.
**Cause**: KL divergence is too strong, forcing the model to ignore latent information.
**Fix**: Reduce `kl_weight` (β) or use KL annealing.

### Pitfall 2: KL Collapse / Posterior Collapse
**Symptom**: KL loss drops to ~0, model ignores latent space.
**Cause**: The decoder is so powerful it doesn't need the latent code.
**Fix**: Use KL annealing, reduce decoder capacity, or increase latent dim.

### Pitfall 3: Mode Collapse
**Symptom**: All generated samples look the same.
**Cause**: The model maps everything to the same region of latent space.
**Fix**: Increase latent dim, use a more expressive encoder, or check for bugs.

### Pitfall 4: Checkerboard Artifacts
**Symptom**: Generated images have grid-like patterns.
**Cause**: ConvTranspose2d with certain stride/kernel combinations.
**Fix**: Use `kernel_size` divisible by `stride`, or upsample + Conv2d instead.

### Pitfall 5: Training Instability
**Symptom**: Loss spikes or oscillates wildly.
**Fix**: Reduce learning rate, add gradient clipping, or increase batch size.

---

## 16. Further Reading & References

### Foundational Papers

| Paper | Year | Key Contribution |
|-------|------|-----------------|
| **Auto-Encoding Variational Bayes** (Kingma & Welling) | 2013 | Introduced the VAE framework |
| **Stochastic Backpropagation** (Rezende et al.) | 2014 | Independent derivation of VAE |
| **β-VAE** (Higgins et al.) | 2017 | Disentangled representations via β weighting |
| **VQ-VAE** (van den Oord et al.) | 2017 | Discrete latent spaces for high-fidelity generation |
| **NVAE** (Vahdat & Kautz) | 2020 | Hierarchical VAE achieving state-of-the-art image quality |

### Tutorial Resources

- [Stanford CS231n: Generative Models](http://cs231n.stanford.edu/)
- [Lil'Log: From Autoencoder to Beta-VAE](https://lilianweng.github.io/posts/2018-08-12-vae/)
- [PyTorch Official VAE Example](https://github.com/pytorch/examples/tree/main/vae)
- [Understanding VAEs (Doersch, 2016)](https://arxiv.org/abs/1606.05908)

### Advanced Variants

| Variant | Key Idea |
|---------|----------|
| **Denoising AE** | Train with corrupted inputs → learn robust features |
| **Sparse AE** | Add sparsity penalty → learn selective features |
| **Contractive AE** | Penalize sensitivity to input changes → learn invariances |
| **β-VAE** | Tunable β on KL → disentangled latent factors |
| **VQ-VAE** | Discrete latent codes → used in DALL-E |
| **CVAE** | Conditional generation — "generate a 7" |
| **WAE** | Wasserstein distance instead of KL divergence |

---

## 17. Verified Training Results

Both models have been fully trained and tested on MNIST. Here are the verified results:

### Standard Autoencoder

| Metric | Value |
|--------|-------|
| Parameters | 141,329 |
| Loss Function | MSE |
| Latent Dimension | 16 |
| Device | CUDA (GPU) |
| Dataset | MNIST (60K train / 10K test) |

**Generated Outputs:**
- `ae_training_loss.png` — Loss converges smoothly over epochs
- `ae_reconstructions.png` — Digits are clearly recognizable in reconstructions
- `ae_latent_space.png` — First 2 latent dims show emerging class clusters
- `ae_generated_samples.png` — Random samples from latent space

### Variational Autoencoder

| Metric | Value |
|--------|-------|
| Parameters | 200,197 |
| Loss Function | BCE + KL Divergence |
| Latent Dimension | 2 |
| KL Weight (β) | 1.0 |
| Device | CUDA (GPU) |
| Dataset | MNIST (60K train / 10K test) |

**Generated Outputs:**
- `vae_training_loss.png` — Shows total, reconstruction, and KL losses separately
- `vae_reconstructions.png` — Good reconstructions even with only 2 latent dims
- `vae_latent_space.png` — Beautiful 2D scatter with clear digit clusters
- `vae_generated_samples.png` — New digits generated from N(0,I)
- `vae_manifold.png` — 20×20 grid of the 2D latent manifold showing smooth digit transitions

### Key Observations

1. **VAE latent space is structured**: The 2D scatter plot shows clear digit clusters, confirming KL regularization works.
2. **Smooth manifold**: The 2D manifold grid shows continuous transitions between digit classes — a hallmark of a well-trained VAE.
3. **AE latent space is less organized**: Without KL regularization, the AE latent space has more scattered, irregular clusters.
4. **Both models reconstruct well**: Reconstructed digits are clearly readable for both architectures.

---

## Summary

```
Standard                              Variational
Autoencoder                           Autoencoder
─────────────                         ─────────────────
x → Encoder → z → Decoder → x̂        x → Encoder → (μ, log σ²)
                                                    ↓
Loss = MSE(x, x̂)                     z = μ + σ · ε  (reparameterize)
                                                    ↓
                                      z → Decoder → x̂

                                      Loss = BCE(x, x̂) + KL(q(z|x) || N(0,I))
```

Autoencoders are among the most foundational architectures in deep learning. Understanding them gives you the building blocks for:
- Representation learning
- Generative modeling
- Unsupervised feature discovery
- And much more

**Run the code, experiment with hyperparameters, and watch the latent space come to life!**
