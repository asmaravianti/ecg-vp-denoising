from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class WindowingConfig:
    sample_rate: int = 360
    window_seconds: float = 2.0
    step_seconds: float = 2.0

    @property
    def window_size(self) -> int:
        return int(self.sample_rate * self.window_seconds)

    @property
    def step_size(self) -> int:
        return int(self.sample_rate * self.step_seconds)


class ArrayECGDataset(Dataset):
    """Simple dataset from in-memory arrays.

    Use this for smoke tests and early experiments; replace with WFDB loader later.
    """

    def __init__(
        self,
        signals: List[np.ndarray],  # list of 1D arrays (clean ECG)
        config: WindowingConfig,
        noise_mixer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> None:
        self.config = config
        self.clean_windows: List[np.ndarray] = []
        for s in signals:
            self.clean_windows.extend(self._segment(s))
        self.noise_mixer = noise_mixer

    def _segment(self, x: np.ndarray) -> List[np.ndarray]:
        ws, ss = self.config.window_size, self.config.step_size
        out = []
        for start in range(0, max(1, len(x) - ws + 1), ss):
            out.append(x[start : start + ws].astype(np.float32))
        return out

    def __len__(self) -> int:
        return len(self.clean_windows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        clean = self.clean_windows[idx]
        noisy = clean.copy()
        if self.noise_mixer is not None:
            noisy = self.noise_mixer(noisy)
        x = torch.from_numpy(noisy).unsqueeze(0)  # (1, T)
        y = torch.from_numpy(clean).unsqueeze(0)
        return x, y


def gaussian_snr_mixer(target_snr_db: float) -> Callable[[np.ndarray], np.ndarray]:
    def _mix(x: np.ndarray) -> np.ndarray:
        p_signal = np.mean(x**2) + 1e-12
        snr_linear = 10 ** (target_snr_db / 10.0)
        p_noise = p_signal / snr_linear
        noise = np.random.randn(*x.shape).astype(np.float32)
        noise = noise / (np.var(noise) + 1e-12) ** 0.5
        noise = noise * (p_noise ** 0.5)
        return x + noise
    return _mix



