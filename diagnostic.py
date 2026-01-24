"""
Simple diagnostic: Check encoder output range and prediction bounds
"""
import os
import torch
import numpy as np
from config import Config
from utils import set_seed, get_device
from data import EnhancedPM25Dataset
from models import ControlledKoopmanModel

config = Config()
set_seed(42)
device = get_device()

dataset = EnhancedPM25Dataset(config.DATA_DIR, config, augment=False)
model = ControlledKoopmanModel(config).to(device)
model_path = os.path.join(config.OUTPUT_DIR, 'trained_model.pth')
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

print("\n=== DIAGNOSTIC RESULTS ===\n")

# Test 1: Encoder output range
print("[1] Encoder Output Range Analysis")
print("-" * 50)
with torch.no_grad():
    x_seq, meteo_seq, mask = dataset[15]  # (T, 1, H, W), (T, 2), ...
    x_last = x_seq[-1].to(device)  # (1, H, W) -> need [B, C, H, W]
    x_last = x_last.unsqueeze(0)  # [1, 1, H, W]
    z = model.encoder(x_last)
    print(f"Input shape: {x_last.shape}, range: [{x_last.min():.4f}, {x_last.max():.4f}]")
    print(f"Encoder output norm: {z.norm():.4f}")
    print(f"Encoder output range: [{z.min():.4f}, {z.max():.4f}]")

# Test 2: Decoder output range
print("\n[2] Decoder Output Range Analysis")
print("-" * 50)
with torch.no_grad():
    x_recon = model.decoder(z)
    print(f"Decoder output shape: {x_recon.shape}, range: [{x_recon.min():.4f}, {x_recon.max():.4f}]")
    recon_mse = torch.mean((x_last - x_recon) ** 2).item()
    print(f"Reconstruction MSE: {recon_mse:.6f}")
    if x_recon.min() < -0.1 or x_recon.max() > 1.1:
        print("WARNING: Decoder outputs out of [0,1]!")
        print("  => Issue: Decoder needs output clamping")

# Test 3: K matrix eigenvalues
print("\n[3] Koopman Matrix Eigenvalues")
print("-" * 50)
with torch.no_grad():
    K = model.K.weight.data  # Linear layer weights
    K_np = K.cpu().numpy()
    eigvals = np.linalg.eigvals(K_np)
    eigvals_mag = np.abs(eigvals)
    print(f"Max eig magnitude: {eigvals_mag.max():.4f}")
    print(f"Mean eig magnitude: {eigvals_mag.mean():.4f}")
    eig_count = (eigvals_mag > 0.9).sum()
    print(f"Eigs > 0.9: {eig_count}/{len(eigvals_mag)}")
    print(f"K matrix norm: {np.linalg.norm(K_np):.4f}")

# Test 4: Single step prediction
print("\n[4] Single-Step Prediction")
print("-" * 50)
with torch.no_grad():
    pred = model.decoder(z + 0.01 * torch.randn_like(z))
    out_of_range = (((pred < 0) | (pred > 1)).sum() / pred.numel()).item() * 100
    print(f"Pred range: [{pred.min():.4f}, {pred.max():.4f}]")
    print(f"Out-of-range pixels: {out_of_range:.1f}%")

# Test 5: Compare with baseline
print("\n[5] Baseline Comparison")
print("-" * 50)
with torch.no_grad():
    x_seq1, _, _ = dataset[20]  # (T, 1, H, W)
    x_seq2, _, _ = dataset[21]
    x_last1 = x_seq1[-1].to(device)  # (1, H, W)
    x_last2 = x_seq2[-1].to(device)
    x_mean = torch.mean(x_seq1.to(device), dim=0)  # (1, H, W)
    
    persist_error = torch.mean((x_last1 - x_last2) ** 2).item()
    mean_error = torch.mean((x_mean - x_last2) ** 2).item()
    
    print(f"Persistence MSE: {persist_error:.6f}")
    print(f"Mean MSE: {mean_error:.6f}")

print("\n" + "="*50)
print("KEY FINDINGS:")
print("- Decoder output range issues would explain negative R2")
print("- Eigenvalues > 0.9 mean weak learning of dynamics")
print("- Out-of-range predictions impossible to get good R2")
print("="*50 + "\n")
