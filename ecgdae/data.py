from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Dict

import numpy as np
import torch
import wfdb
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


class MITBIHLoader:
    """Loader for MIT-BIH Arrhythmia Database.
    
    Downloads and caches MIT-BIH records from PhysioNet.
    Returns clean ECG signals for training.
    """
    
    # Standard MIT-BIH Arrhythmia Database records
    MITBIH_RECORDS = [
        '100', '101', '102', '103', '104', '105', '106', '107', '108', '109',
        '111', '112', '113', '114', '115', '116', '117', '118', '119', '121',
        '122', '123', '124', '200', '201', '202', '203', '205', '207', '208',
        '209', '210', '212', '213', '214', '215', '217', '219', '220', '221',
        '222', '223', '228', '230', '231', '232', '233', '234'
    ]
    
    def __init__(self, data_dir: str = "./data/mitbih", channel: int = 0):
        """Initialize MIT-BIH loader.
        
        Args:
            data_dir: Directory to cache downloaded data
            channel: Which ECG channel to use (0 or 1, typically MLII or V5)
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.channel = channel
    
    def load_record(self, record_name: str) -> Tuple[np.ndarray, int]:
        """Load a single MIT-BIH record.
        
        Args:
            record_name: Record identifier (e.g., '100')
            
        Returns:
            signal: ECG signal array (1D)
            fs: Sampling frequency
        """
        try:
            record = wfdb.rdrecord(
                record_name,
                pn_dir='mitdb',
                channels=[self.channel]
            )
            signal = record.p_signal[:, 0].astype(np.float32)
            fs = record.fs
            return signal, fs
        except Exception as e:
            print(f"Error loading record {record_name}: {e}")
            return None, None
    
    def load_all_records(
        self, 
        record_list: Optional[List[str]] = None,
        normalize: bool = True
    ) -> List[np.ndarray]:
        """Load multiple MIT-BIH records.
        
        Args:
            record_list: List of record names to load (default: all)
            normalize: Whether to normalize each signal to zero mean, unit std
            
        Returns:
            List of ECG signal arrays
        """
        if record_list is None:
            record_list = self.MITBIH_RECORDS
        
        signals = []
        for rec in record_list:
            signal, fs = self.load_record(rec)
            if signal is not None:
                if normalize:
                    signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
                signals.append(signal)
                print(f"Loaded record {rec}: length={len(signal)}, fs={fs} Hz")
        
        return signals


class NSTDBNoiseMixer:
    """Noise mixer using MIT-BIH Noise Stress Test Database (NSTDB).
    
    Provides realistic ECG noise from baseline wander, muscle artifacts, 
    and electrode motion artifacts.
    """
    
    # NSTDB noise records
    NOISE_RECORDS = {
        'baseline_wander': 'bw',  # baseline wander
        'muscle_artifact': 'ma',  # muscle artifact
        'electrode_motion': 'em'  # electrode motion artifact
    }
    
    def __init__(self, data_dir: str = "./data/nstdb", channel: int = 0):
        """Initialize NSTDB noise mixer.
        
        Args:
            data_dir: Directory to cache downloaded data
            channel: Which noise channel to use
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.channel = channel
        self.noise_cache: Dict[str, np.ndarray] = {}
    
    def load_noise(self, noise_type: str = 'muscle_artifact') -> np.ndarray:
        """Load noise signal from NSTDB.
        
        Args:
            noise_type: Type of noise ('baseline_wander', 'muscle_artifact', 'electrode_motion')
            
        Returns:
            Noise signal array
        """
        if noise_type in self.noise_cache:
            return self.noise_cache[noise_type]
        
        record_name = self.NOISE_RECORDS.get(noise_type, 'ma')
        
        try:
            record = wfdb.rdrecord(
                record_name,
                pn_dir='nstdb',
                channels=[self.channel]
            )
            noise = record.p_signal[:, 0].astype(np.float32)
            # Normalize noise
            noise = (noise - np.mean(noise)) / (np.std(noise) + 1e-8)
            self.noise_cache[noise_type] = noise
            print(f"Loaded NSTDB noise '{noise_type}': length={len(noise)}")
            return noise
        except Exception as e:
            print(f"Error loading NSTDB noise '{noise_type}': {e}")
            print("Falling back to Gaussian noise")
            return None
    
    def create_mixer(
        self, 
        target_snr_db: float,
        noise_type: str = 'muscle_artifact',
        mix_gaussian: bool = False,
        gaussian_ratio: float = 0.3
    ) -> Callable[[np.ndarray], np.ndarray]:
        """Create a noise mixing function.
        
        Args:
            target_snr_db: Target signal-to-noise ratio in dB
            noise_type: Type of NSTDB noise to use
            mix_gaussian: Whether to mix with Gaussian noise
            gaussian_ratio: Ratio of Gaussian noise if mixed (0-1)
            
        Returns:
            Mixing function that adds noise to clean signal
        """
        noise_signal = self.load_noise(noise_type)
        
        def _mix(x: np.ndarray) -> np.ndarray:
            # Calculate signal power
            p_signal = np.mean(x**2) + 1e-12
            snr_linear = 10 ** (target_snr_db / 10.0)
            p_noise = p_signal / snr_linear
            
            if noise_signal is not None:
                # Sample random segment from NSTDB noise
                if len(noise_signal) >= len(x):
                    start_idx = np.random.randint(0, len(noise_signal) - len(x) + 1)
                    noise_segment = noise_signal[start_idx:start_idx + len(x)]
                else:
                    # Repeat noise if signal is longer
                    repeats = (len(x) // len(noise_signal)) + 1
                    noise_segment = np.tile(noise_signal, repeats)[:len(x)]
                
                # Scale noise to target power
                noise_segment = noise_segment / (np.std(noise_segment) + 1e-12)
                noise_segment = noise_segment * (p_noise ** 0.5)
                
                # Optionally mix with Gaussian noise
                if mix_gaussian and gaussian_ratio > 0:
                    gaussian_noise = np.random.randn(*x.shape).astype(np.float32)
                    gaussian_noise = gaussian_noise / (np.std(gaussian_noise) + 1e-12)
                    gaussian_noise = gaussian_noise * (p_noise ** 0.5)
                    
                    noise_segment = (1 - gaussian_ratio) * noise_segment + gaussian_ratio * gaussian_noise
                
                return x + noise_segment
            else:
                # Fallback to Gaussian noise
                noise = np.random.randn(*x.shape).astype(np.float32)
                noise = noise / (np.var(noise) + 1e-12) ** 0.5
                noise = noise * (p_noise ** 0.5)
                return x + noise
        
        return _mix


class MITBIHDataset(Dataset):
    """MIT-BIH dataset with windowing and noise mixing.
    
    Loads MIT-BIH records, applies windowing, and optionally adds noise.
    """
    
    def __init__(
        self,
        records: Optional[List[str]] = None,
        config: Optional[WindowingConfig] = None,
        noise_mixer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        data_dir: str = "./data/mitbih",
        channel: int = 0,
        normalize: bool = True,
    ):
        """Initialize MIT-BIH dataset.
        
        Args:
            records: List of record names to load (default: all)
            config: Windowing configuration
            noise_mixer: Function to add noise to clean signals
            data_dir: Directory to cache data
            channel: ECG channel to use
            normalize: Whether to normalize signals
        """
        self.config = config or WindowingConfig()
        self.noise_mixer = noise_mixer
        
        # Load MIT-BIH data
        loader = MITBIHLoader(data_dir=data_dir, channel=channel)
        signals = loader.load_all_records(record_list=records, normalize=normalize)
        
        # Window signals
        self.clean_windows: List[np.ndarray] = []
        for signal in signals:
            self.clean_windows.extend(self._segment(signal))
        
        print(f"Created dataset with {len(self.clean_windows)} windows")
    
    def _segment(self, x: np.ndarray) -> List[np.ndarray]:
        """Segment signal into windows."""
        ws, ss = self.config.window_size, self.config.step_size
        out = []
        for start in range(0, max(1, len(x) - ws + 1), ss):
            out.append(x[start : start + ws].astype(np.float32))
        return out
    
    def __len__(self) -> int:
        return len(self.clean_windows)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a data sample.
        
        Returns:
            noisy: Noisy ECG signal (1, T)
            clean: Clean ECG signal (1, T)
        """
        clean = self.clean_windows[idx]
        noisy = clean.copy()
        if self.noise_mixer is not None:
            noisy = self.noise_mixer(noisy)
        x = torch.from_numpy(noisy).unsqueeze(0)  # (1, T)
        y = torch.from_numpy(clean).unsqueeze(0)
        return x, y



