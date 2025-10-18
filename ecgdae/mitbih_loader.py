"""
MIT-BIH Arrhythmia Database loader with windowing and noise augmentation.

This module handles loading ECG signals from MIT-BIH database, segmenting them
into windows, and adding realistic noise for training denoising models.

Author: [Your Name]
Date: [Current Date]
Course: TDK Project - ECG Denoising and Compression
"""

import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

# Suppress WFDB warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="wfdb")


@dataclass
class MITBIHConfig:
    """Configuration for MIT-BIH dataset loading."""
    
    # Dataset paths
    data_dir: str = "data/mitbih"
    noise_dir: str = "data/nstdb"
    
    # Signal processing
    sample_rate: int = 360  # Hz
    window_seconds: float = 2.0  # seconds
    step_seconds: float = 2.0  # seconds (no overlap)
    
    # Record selection
    train_records: List[str] = None  # Will be set to default if None
    test_records: List[str] = None    # Will be set to default if None
    
    # Noise augmentation
    noise_types: List[str] = None  # Will be set to default if None
    snr_levels: List[float] = None  # Will be set to default if None
    
    def __post_init__(self):
        """Set default values after initialization."""
        if self.train_records is None:
            # Use first 20 records for training (balanced mix of normal/abnormal)
            self.train_records = [
                "100", "101", "102", "103", "104", "105", "106", "107", "108", "109",
                "111", "112", "113", "114", "115", "116", "117", "118", "119", "121"
            ]
        
        if self.test_records is None:
            # Use next 10 records for testing
            self.test_records = [
                "122", "123", "124", "200", "201", "202", "203", "205", "207", "208"
            ]
        
        if self.noise_types is None:
            # Types of realistic noise from NSTDB
            self.noise_types = ["bw", "em", "ma"]  # baseline wander, electrode motion, muscle artifact
        
        if self.snr_levels is None:
            # Signal-to-noise ratios in dB (lower = more noise)
            self.snr_levels = [5.0, 10.0, 20.0]


class MITBIHDataset(Dataset):
    """
    MIT-BIH Arrhythmia Database dataset with windowing and noise augmentation.
    
    This class loads ECG signals from MIT-BIH database, segments them into
    fixed-length windows, and optionally adds realistic noise for training
    denoising models.
    
    Attributes:
        config: MITBIHConfig object with dataset parameters
        windows: List of ECG windows (clean signals)
        noise_windows: List of corresponding noisy windows
        record_ids: List of record IDs for each window
    """
    
    def __init__(
        self,
        config: MITBIHConfig,
        split: str = "train",
        add_noise: bool = True,
        noise_snr: Optional[float] = None,
        noise_type: Optional[str] = None,
    ):
        """
        Initialize MIT-BIH dataset.
        
        Args:
            config: Dataset configuration
            split: "train" or "test"
            add_noise: Whether to add noise to create noisy inputs
            noise_snr: Specific SNR level (if None, uses random from config)
            noise_type: Specific noise type (if None, uses random from config)
        """
        self.config = config
        self.split = split
        self.add_noise = add_noise
        self.noise_snr = noise_snr
        self.noise_type = noise_type
        
        # Select records based on split
        if split == "train":
            self.record_ids = config.train_records
        elif split == "test":
            self.record_ids = config.test_records
        else:
            raise ValueError(f"Unknown split: {split}")
        
        # Load and segment ECG signals
        self.windows = []
        self.noise_windows = []
        self.window_record_ids = []
        
        self._load_and_segment()
        
        print(f"Loaded {len(self.windows)} windows from {len(self.record_ids)} records")
        print(f"Window size: {self.config.window_seconds}s ({self.window_size} samples)")
    
    @property
    def window_size(self) -> int:
        """Number of samples per window."""
        return int(self.config.sample_rate * self.config.window_seconds)
    
    @property
    def step_size(self) -> int:
        """Number of samples between window starts."""
        return int(self.config.sample_rate * self.config.step_seconds)
    
    def _load_and_segment(self):
        """Load ECG records and segment into windows."""
        for record_id in self.record_ids:
            try:
                # Load ECG signal
                signal = self._load_record(record_id)
                
                # Segment into windows
                windows = self._segment_signal(signal)
                
                # Add windows to dataset
                for window in windows:
                    self.windows.append(window)
                    self.window_record_ids.append(record_id)
                    
                    # Create noisy version if requested
                    if self.add_noise:
                        noisy_window = self._add_noise(window)
                        self.noise_windows.append(noisy_window)
                    else:
                        self.noise_windows.append(window.copy())
                        
            except Exception as e:
                print(f"Warning: Could not load record {record_id}: {e}")
                continue
    
    def _load_record(self, record_id: str) -> np.ndarray:
        """
        Load a single MIT-BIH record.
        
        For now, we'll create synthetic ECG-like signals since we don't have
        the actual MIT-BIH files. In a real implementation, you would use
        the WFDB library to load .dat files.
        
        Args:
            record_id: Record identifier (e.g., "100")
            
        Returns:
            ECG signal as numpy array
        """
        # Create synthetic ECG-like signal for demonstration
        # In practice, you would load the actual .dat file here
        duration = 30 * 60  # 30 minutes
        t = np.arange(duration * self.config.sample_rate) / self.config.sample_rate
        
        # Create ECG-like morphology with heart rate variability
        hr = 70 + 10 * np.sin(2 * np.pi * 0.1 * t)  # Varying heart rate
        rr_intervals = 60.0 / hr  # R-R intervals in seconds
        
        # Generate R-peaks
        r_peaks = []
        current_time = 0
        while current_time < duration:
            r_peaks.append(current_time)
            current_time += rr_intervals[int(current_time * self.config.sample_rate)]
        
        # Create ECG waveform
        signal = np.zeros_like(t)
        for r_time in r_peaks:
            r_idx = int(r_time * self.config.sample_rate)
            if r_idx < len(signal):
                # QRS complex (simplified)
                qrs_start = max(0, r_idx - 20)
                qrs_end = min(len(signal), r_idx + 20)
                signal[qrs_start:qrs_end] += np.exp(-0.5 * ((np.arange(qrs_start, qrs_end) - r_idx) / 5) ** 2)
        
        # Add baseline wander and high-frequency noise
        signal += 0.1 * np.sin(2 * np.pi * 0.5 * t)  # Baseline wander
        signal += 0.05 * np.random.randn(len(signal))  # High-frequency noise
        
        return signal.astype(np.float32)
    
    def _segment_signal(self, signal: np.ndarray) -> List[np.ndarray]:
        """
        Segment a long ECG signal into windows.
        
        Args:
            signal: Long ECG signal
            
        Returns:
            List of ECG windows
        """
        windows = []
        ws = self.window_size
        ss = self.step_size
        
        for start in range(0, len(signal) - ws + 1, ss):
            window = signal[start:start + ws]
            windows.append(window)
        
        return windows
    
    def _add_noise(self, clean_signal: np.ndarray) -> np.ndarray:
        """
        Add realistic noise to clean ECG signal.
        
        Args:
            clean_signal: Clean ECG window
            
        Returns:
            Noisy ECG window
        """
        # Select noise parameters
        snr_db = self.noise_snr
        if snr_db is None:
            snr_db = np.random.choice(self.config.snr_levels)
        
        noise_type = self.noise_type
        if noise_type is None:
            noise_type = np.random.choice(self.config.noise_types)
        
        # Generate noise based on type
        if noise_type == "bw":  # Baseline wander
            noise = self._generate_baseline_wander(len(clean_signal))
        elif noise_type == "em":  # Electrode motion
            noise = self._generate_electrode_motion(len(clean_signal))
        elif noise_type == "ma":  # Muscle artifact
            noise = self._generate_muscle_artifact(len(clean_signal))
        else:
            # Fallback to Gaussian noise
            noise = np.random.randn(len(clean_signal))
        
        # Scale noise to achieve target SNR
        signal_power = np.mean(clean_signal ** 2)
        snr_linear = 10 ** (snr_db / 10.0)
        noise_power = signal_power / snr_linear
        noise = noise * np.sqrt(noise_power / (np.var(noise) + 1e-12))
        
        return (clean_signal + noise).astype(np.float32)
    
    def _generate_baseline_wander(self, length: int) -> np.ndarray:
        """Generate baseline wander noise (slow drift)."""
        t = np.arange(length) / self.config.sample_rate
        # Low-frequency components
        noise = 0.5 * np.sin(2 * np.pi * 0.1 * t) + 0.3 * np.sin(2 * np.pi * 0.05 * t)
        return noise.astype(np.float32)
    
    def _generate_electrode_motion(self, length: int) -> np.ndarray:
        """Generate electrode motion artifacts (sudden jumps)."""
        noise = np.zeros(length)
        # Add occasional sudden changes
        jump_positions = np.random.choice(length, size=length // 1000, replace=False)
        for pos in jump_positions:
            jump_size = np.random.uniform(-0.5, 0.5)
            noise[pos:] += jump_size
        return noise.astype(np.float32)
    
    def _generate_muscle_artifact(self, length: int) -> np.ndarray:
        """Generate muscle artifact noise (high-frequency bursts)."""
        noise = np.zeros(length)
        # Add high-frequency bursts
        burst_positions = np.random.choice(length, size=length // 500, replace=False)
        for pos in burst_positions:
            burst_length = np.random.randint(10, 50)
            burst_end = min(pos + burst_length, length)
            burst = np.random.randn(burst_end - pos) * 0.3
            noise[pos:burst_end] += burst
        return noise.astype(np.float32)
    
    def __len__(self) -> int:
        """Return number of windows in dataset."""
        return len(self.windows)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single window pair (noisy input, clean target).
        
        Args:
            idx: Window index
            
        Returns:
            Tuple of (noisy_input, clean_target) as tensors
        """
        noisy = torch.from_numpy(self.noise_windows[idx]).unsqueeze(0)  # (1, T)
        clean = torch.from_numpy(self.windows[idx]).unsqueeze(0)        # (1, T)
        
        return noisy, clean


def create_mitbih_datasets(
    config: MITBIHConfig,
    add_noise: bool = True,
) -> Tuple[MITBIHDataset, MITBIHDataset]:
    """
    Create training and testing MIT-BIH datasets.
    
    Args:
        config: Dataset configuration
        add_noise: Whether to add noise to training data
        
    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    train_dataset = MITBIHDataset(
        config=config,
        split="train",
        add_noise=add_noise,
    )
    
    test_dataset = MITBIHDataset(
        config=config,
        split="test",
        add_noise=False,  # Test data should be clean
    )
    
    return train_dataset, test_dataset


if __name__ == "__main__":
    # Test the dataset
    config = MITBIHConfig()
    train_dataset, test_dataset = create_mitbih_datasets(config)
    
    print(f"Train dataset: {len(train_dataset)} windows")
    print(f"Test dataset: {len(test_dataset)} windows")
    
    # Test a single sample
    noisy, clean = train_dataset[0]
    print(f"Sample shape: noisy {noisy.shape}, clean {clean.shape}")
    print(f"Noisy range: [{noisy.min():.3f}, {noisy.max():.3f}]")
    print(f"Clean range: [{clean.min():.3f}, {clean.max():.3f}]")
