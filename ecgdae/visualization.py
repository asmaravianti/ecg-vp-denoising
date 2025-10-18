"""
Visualization utilities for ECG signals and model results.

This module provides functions to plot ECG waveforms, compare clean vs noisy
vs reconstructed signals, and visualize training progress.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import List, Optional, Tuple, Union


def plot_ecg_window(
    clean: Union[np.ndarray, torch.Tensor],
    noisy: Optional[Union[np.ndarray, torch.Tensor]] = None,
    reconstructed: Optional[Union[np.ndarray, torch.Tensor]] = None,
    sample_rate: int = 360,
    title: str = "ECG Signal",
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot ECG signal window with optional noisy and reconstructed versions.
    
    Args:
        clean: Clean ECG signal (target)
        noisy: Noisy ECG signal (input)
        reconstructed: Reconstructed ECG signal (model output)
        sample_rate: Sampling rate in Hz
        title: Plot title
        save_path: Path to save figure (optional)
        show: Whether to display the plot
        
    Returns:
        matplotlib Figure object
    """
    # Convert tensors to numpy if needed
    if isinstance(clean, torch.Tensor):
        clean = clean.squeeze().cpu().numpy()
    if isinstance(noisy, torch.Tensor):
        noisy = noisy.squeeze().cpu().numpy()
    if isinstance(reconstructed, torch.Tensor):
        reconstructed = reconstructed.squeeze().cpu().numpy()
    
    # Create time axis
    duration = len(clean) / sample_rate
    time = np.linspace(0, duration, len(clean))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot signals
    if noisy is not None:
        ax.plot(time, noisy, 'b-', alpha=0.7, label='Noisy Input', linewidth=1)
    
    ax.plot(time, clean, 'g-', label='Clean Target', linewidth=2)
    
    if reconstructed is not None:
        ax.plot(time, reconstructed, 'r--', label='Reconstructed', linewidth=2)
    
    # Formatting
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Amplitude')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Set reasonable y-axis limits
    all_signals = [clean]
    if noisy is not None:
        all_signals.append(noisy)
    if reconstructed is not None:
        all_signals.append(reconstructed)
    
    y_min = min(s.min() for s in all_signals)
    y_max = max(s.max() for s in all_signals)
    y_margin = (y_max - y_min) * 0.1
    ax.set_ylim(y_min - y_margin, y_max + y_margin)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig


def plot_training_progress(
    train_losses: List[float],
    val_losses: Optional[List[float]] = None,
    title: str = "Training Progress",
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot training and validation loss curves.
    
    Args:
        train_losses: List of training loss values
        val_losses: List of validation loss values (optional)
        title: Plot title
        save_path: Path to save figure (optional)
        show: Whether to display the plot
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    
    if val_losses is not None:
        ax.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (WWPRD)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig


def plot_qrs_comparison(
    clean: Union[np.ndarray, torch.Tensor],
    noisy: Union[np.ndarray, torch.Tensor],
    reconstructed: Union[np.ndarray, torch.Tensor],
    sample_rate: int = 360,
    qrs_center: int = 360,  # Sample index of QRS center
    window_samples: int = 200,  # Samples around QRS
    title: str = "QRS Complex Comparison",
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot detailed comparison of QRS complexes (the main heartbeat spikes).
    
    Args:
        clean: Clean ECG signal
        noisy: Noisy ECG signal
        reconstructed: Reconstructed ECG signal
        sample_rate: Sampling rate in Hz
        qrs_center: Sample index where QRS complex is centered
        window_samples: Number of samples to show around QRS
        title: Plot title
        save_path: Path to save figure (optional)
        show: Whether to display the plot
        
    Returns:
        matplotlib Figure object
    """
    # Convert tensors to numpy if needed
    if isinstance(clean, torch.Tensor):
        clean = clean.squeeze().cpu().numpy()
    if isinstance(noisy, torch.Tensor):
        noisy = noisy.squeeze().cpu().numpy()
    if isinstance(reconstructed, torch.Tensor):
        reconstructed = reconstructed.squeeze().cpu().numpy()
    
    # Extract QRS window
    start_idx = max(0, qrs_center - window_samples // 2)
    end_idx = min(len(clean), qrs_center + window_samples // 2)
    
    clean_qrs = clean[start_idx:end_idx]
    noisy_qrs = noisy[start_idx:end_idx]
    recon_qrs = reconstructed[start_idx:end_idx]
    
    # Create time axis
    time = np.linspace(0, len(clean_qrs) / sample_rate, len(clean_qrs))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(time, noisy_qrs, 'b-', alpha=0.7, label='Noisy Input', linewidth=1)
    ax.plot(time, clean_qrs, 'g-', label='Clean Target', linewidth=2)
    ax.plot(time, recon_qrs, 'r--', label='Reconstructed', linewidth=2)
    
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Amplitude')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig


def create_overview_plot(
    clean: Union[np.ndarray, torch.Tensor],
    noisy: Union[np.ndarray, torch.Tensor],
    reconstructed: Union[np.ndarray, torch.Tensor],
    sample_rate: int = 360,
    prd: Optional[float] = None,
    wwprd: Optional[float] = None,
    snr: Optional[float] = None,
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Create a comprehensive overview plot with full window and QRS zoom.
    
    Args:
        clean: Clean ECG signal
        noisy: Noisy ECG signal
        reconstructed: Reconstructed ECG signal
        sample_rate: Sampling rate in Hz
        prd: PRD value to display
        wwprd: WWPRD value to display
        snr: SNR value to display
        save_path: Path to save figure (optional)
        show: Whether to display the plot
        
    Returns:
        matplotlib Figure object
    """
    # Convert tensors to numpy if needed
    if isinstance(clean, torch.Tensor):
        clean = clean.squeeze().cpu().numpy()
    if isinstance(noisy, torch.Tensor):
        noisy = noisy.squeeze().cpu().numpy()
    if isinstance(reconstructed, torch.Tensor):
        reconstructed = reconstructed.squeeze().cpu().numpy()
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Full window plot
    duration = len(clean) / sample_rate
    time = np.linspace(0, duration, len(clean))
    
    ax1.plot(time, noisy, 'b-', alpha=0.7, label='Noisy Input', linewidth=1)
    ax1.plot(time, clean, 'g-', label='Clean Target', linewidth=2)
    ax1.plot(time, reconstructed, 'r--', label='Reconstructed', linewidth=2)
    
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Amplitude')
    ax1.set_title('Full ECG Window')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # QRS zoom plot (find the largest peak)
    qrs_center = np.argmax(np.abs(clean))
    window_samples = 200
    start_idx = max(0, qrs_center - window_samples // 2)
    end_idx = min(len(clean), qrs_center + window_samples // 2)
    
    clean_qrs = clean[start_idx:end_idx]
    noisy_qrs = noisy[start_idx:end_idx]
    recon_qrs = reconstructed[start_idx:end_idx]
    time_qrs = np.linspace(0, len(clean_qrs) / sample_rate, len(clean_qrs))
    
    ax2.plot(time_qrs, noisy_qrs, 'b-', alpha=0.7, label='Noisy Input', linewidth=1)
    ax2.plot(time_qrs, clean_qrs, 'g-', label='Clean Target', linewidth=2)
    ax2.plot(time_qrs, recon_qrs, 'r--', label='Reconstructed', linewidth=2)
    
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Amplitude')
    ax2.set_title('QRS Complex Detail')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add metrics to the plot
    metrics_text = []
    if prd is not None:
        metrics_text.append(f'PRD: {prd:.2f}%')
    if wwprd is not None:
        metrics_text.append(f'WWPRD: {wwprd:.2f}%')
    if snr is not None:
        metrics_text.append(f'SNR: {snr:.1f} dB')
    
    if metrics_text:
        fig.text(0.02, 0.02, ' | '.join(metrics_text), fontsize=12, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig


if __name__ == "__main__":
    # Test the plotting functions
    print("Testing ECG visualization functions...")
    
    # Create synthetic ECG data
    sample_rate = 360
    duration = 2.0  # seconds
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Create ECG-like signal
    clean = np.sin(2 * np.pi * 1.2 * t) + 0.5 * np.sin(2 * np.pi * 3.0 * t)
    noisy = clean + 0.3 * np.random.randn(len(clean))
    reconstructed = clean + 0.1 * np.random.randn(len(clean))
    
    # Test plotting functions
    plot_ecg_window(clean, noisy, reconstructed, sample_rate, "Test ECG Plot")
    create_overview_plot(clean, noisy, reconstructed, sample_rate, 
                        prd=15.2, wwprd=12.8, snr=18.5)
    
    print("Plotting functions work correctly!")
