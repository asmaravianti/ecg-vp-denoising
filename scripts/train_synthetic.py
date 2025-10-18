import math
from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from ecgdae.data import ArrayECGDataset, WindowingConfig, gaussian_snr_mixer
from ecgdae.losses import STFTWeightedWWPRDLoss


class TinyCAE(nn.Module):
    def __init__(self, channels: int = 1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(channels, 16, kernel_size=9, stride=2, padding=4),
            nn.GELU(),
            nn.Conv1d(16, 32, kernel_size=9, stride=2, padding=4),
            nn.GELU(),
            nn.Conv1d(32, 16, kernel_size=9, stride=2, padding=4),
            nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose1d(16, channels, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)


def synthetic_ecg(length: int, sr: int) -> np.ndarray:
    t = np.arange(length) / sr
    # Sum of sinusoids with drifting baseline to mimic ECG-like morphology
    signal = (
        0.6 * np.sin(2 * math.pi * 1.3 * t)
        + 0.3 * np.sin(2 * math.pi * 3.0 * t + 0.4)
        + 0.1 * np.sin(2 * math.pi * 50.0 * t)
        + 0.05 * np.sin(2 * math.pi * 0.2 * t)
    ).astype(np.float32)
    return signal


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sr = 360
    cfg = WindowingConfig(sample_rate=sr, window_seconds=2.0, step_seconds=2.0)

    # Build tiny synthetic dataset
    signals = [synthetic_ecg(length=sr * 120, sr=sr) for _ in range(4)]
    ds = ArrayECGDataset(signals, cfg, noise_mixer=gaussian_snr_mixer(10.0))
    dl = DataLoader(ds, batch_size=32, shuffle=True, drop_last=True)

    model = TinyCAE().to(device)
    loss_fn = STFTWeightedWWPRDLoss().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)

    for step, (x_noisy, x_clean) in enumerate(dl):
        x_noisy = x_noisy.to(device)
        x_clean = x_clean.to(device)
        y = model(x_noisy)
        loss = loss_fn(x_clean, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 10 == 0:
            print(f"step {step}: WWPRD={loss.item():.2f}")
        if step >= 60:
            break

    # Show final improvement on a held-out batch
    x_noisy, x_clean = next(iter(dl))
    x_noisy = x_noisy.to(device)
    x_clean = x_clean.to(device)
    with torch.no_grad():
        y0 = x_noisy
        y1 = model(x_noisy)
        l0 = loss_fn(x_clean, y0).item()
        l1 = loss_fn(x_clean, y1).item()
    print(f"WWPRD before model: {l0:.2f} | after model: {l1:.2f}")


if __name__ == "__main__":
    main()



