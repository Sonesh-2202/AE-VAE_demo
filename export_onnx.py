"""
Export trained AE and VAE models to ONNX format for browser inference.
Produces single self-contained .onnx files (no external .data files).
"""

import os
import sys
import struct
import torch

sys.path.insert(0, os.path.dirname(__file__))

from models.autoencoder import Autoencoder
from models.vae import VAE

DEVICE = torch.device("cpu")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "demo", "models")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def export_model(model, dummy_input, filepath, input_names, output_names, dynamic_axes):
    """Export a PyTorch model to a self-contained ONNX file."""
    # Export with opset 13 (widely supported)
    torch.onnx.export(
        model, dummy_input, filepath,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=13,
    )

    # If torch created an external .data file, re-pack everything into one file
    data_file = filepath + ".data"
    if os.path.exists(data_file):
        import onnx
        onnx_model = onnx.load(filepath, load_external_data=True)
        # Save with everything embedded
        onnx.save_model(onnx_model, filepath,
                        save_as_external_data=False,
                        all_tensors_to_one_file=False)
        if os.path.exists(data_file):
            os.remove(data_file)

    size_kb = os.path.getsize(filepath) / 1024
    print(f"  ✓ {os.path.basename(filepath)} ({size_kb:.0f} KB)")


def export_autoencoder():
    print("\n🔵 Exporting Standard Autoencoder...")
    model = Autoencoder(latent_dim=16)
    model.load_state_dict(torch.load("outputs/autoencoder.pth", map_location=DEVICE, weights_only=True))
    model.eval()

    dummy = torch.randn(1, 1, 28, 28)

    # Full AE (encode + decode)
    export_model(
        model, dummy,
        os.path.join(OUTPUT_DIR, "autoencoder.onnx"),
        ["input"], ["output"],
        {"input": {0: "batch"}, "output": {0: "batch"}},
    )

    # Encoder only (for latent visualization)
    class EncWrap(torch.nn.Module):
        def __init__(self, enc):
            super().__init__()
            self.enc = enc
        def forward(self, x):
            return self.enc(x)

    enc = EncWrap(model.encoder)
    enc.eval()
    export_model(
        enc, dummy,
        os.path.join(OUTPUT_DIR, "ae_encoder.onnx"),
        ["input"], ["latent"],
        {"input": {0: "batch"}, "latent": {0: "batch"}},
    )


def export_vae():
    print("\n🟣 Exporting Variational Autoencoder...")
    model = VAE(latent_dim=2)
    model.load_state_dict(torch.load("outputs/vae.pth", map_location=DEVICE, weights_only=True))
    model.eval()

    dummy = torch.randn(1, 1, 28, 28)

    # Full VAE (encode + reparameterize + decode)
    class VAEFwd(torch.nn.Module):
        def __init__(self, vae):
            super().__init__()
            self.vae = vae
        def forward(self, x):
            mu, log_var = self.vae.encode(x)
            z = self.vae.reparameterize(mu, log_var)
            return self.vae.decode(z)

    vae_fwd = VAEFwd(model)
    vae_fwd.eval()
    export_model(
        vae_fwd, dummy,
        os.path.join(OUTPUT_DIR, "vae.onnx"),
        ["input"], ["output"],
        {"input": {0: "batch"}, "output": {0: "batch"}},
    )

    # Decoder only (for latent space exploration)
    class DecWrap(torch.nn.Module):
        def __init__(self, vae):
            super().__init__()
            self.vae = vae
        def forward(self, z):
            return self.vae.decode(z)

    dec = DecWrap(model)
    dec.eval()
    export_model(
        dec, torch.randn(1, 2),
        os.path.join(OUTPUT_DIR, "vae_decoder.onnx"),
        ["latent"], ["output"],
        {"latent": {0: "batch"}, "output": {0: "batch"}},
    )


if __name__ == "__main__":
    export_autoencoder()
    export_vae()

    # Verify
    print("\n📋 Verification:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        fp = os.path.join(OUTPUT_DIR, f)
        size = os.path.getsize(fp) / 1024
        ext = os.path.splitext(f)[1]
        if ext == ".data":
            print(f"  ⚠ UNWANTED: {f} ({size:.0f} KB)")
        else:
            print(f"  ✓ {f} ({size:.0f} KB)")

    data_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith(".data")]
    if data_files:
        print(f"\n❌ External data files still exist: {data_files}")
    else:
        print(f"\n✅ All models are self-contained — ready for browser!")
