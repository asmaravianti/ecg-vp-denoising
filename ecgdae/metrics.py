"""Evaluation metrics for ECG compression and denoising."""

import numpy as np
import torch
from typing import Tuple, Dict, Optional


def compute_prd(clean: np.ndarray, reconstructed: np.ndarray, eps: float = 1e-12) -> float:
    """Compute Percent Root-mean-square Difference (PRD).

    PRD = 100 * sqrt(sum((clean - recon)^2) / sum(clean^2))

    Lower is better. Clinical quality ranges:
    - Excellent: PRD < 4.33%
    - Very Good: 4.33% ≤ PRD < 9.00%
    - Good: 9.00% ≤ PRD < 15.00%
    - Not Good: PRD ≥ 15.00%

    Args:
        clean: Clean ECG signal
        reconstructed: Reconstructed ECG signal
        eps: Small constant for numerical stability

    Returns:
        PRD value in percentage
    """
    clean = clean.flatten()
    reconstructed = reconstructed.flatten()

    numerator = np.sum((clean - reconstructed) ** 2)
    denominator = np.sum(clean ** 2) + eps
    prd = 100.0 * np.sqrt(numerator / denominator)

    return float(prd)


def compute_wwprd(
    clean: np.ndarray,
    reconstructed: np.ndarray,
    weights: Optional[np.ndarray] = None,
    eps: float = 1e-12
) -> float:
    """Compute Waveform-Weighted PRD (WWPRD).

    WWPRD = 100 * sqrt(sum(w * (clean - recon)^2) / sum(w * clean^2))

    where w are weights emphasizing clinically important features (e.g., QRS complexes).

    Clinical quality ranges:
    - Excellent: WWPRD < 7.4%
    - Very Good: 7.4% ≤ WWPRD < 14.8%
    - Good: 14.8% ≤ WWPRD < 24.7%
    - Not Good: WWPRD ≥ 24.7%

    Args:
        clean: Clean ECG signal
        reconstructed: Reconstructed ECG signal
        weights: Waveform weights (default: computed from signal derivative)
        eps: Small constant for numerical stability

    Returns:
        WWPRD value in percentage
    """
    clean = clean.flatten()
    reconstructed = reconstructed.flatten()

    if weights is None:
        # Default weights based on signal derivative (emphasizes QRS)
        weights = compute_derivative_weights(clean)
    else:
        weights = weights.flatten()

    # Ensure weights are non-negative
    weights = np.maximum(weights, 0.0)

    numerator = np.sum(weights * (clean - reconstructed) ** 2)
    denominator = np.sum(weights * clean ** 2) + eps
    wwprd = 100.0 * np.sqrt(numerator / denominator)

    return float(wwprd)


def compute_derivative_weights(signal: np.ndarray, alpha: float = 2.0) -> np.ndarray:
    """Compute weights based on signal derivative.

    Weights emphasize regions with high derivative (e.g., QRS complexes).

    w(n) = 1 + alpha * |x'(n)| / max(|x'|)

    Args:
        signal: Input ECG signal
        alpha: Scaling factor for derivative weights (default: 2.0)

    Returns:
        Weight array same shape as signal
    """
    # Flatten signal for processing
    original_shape = signal.shape
    signal_flat = signal.flatten()

    # Need at least 3 points for gradient
    if len(signal_flat) < 3:
        return np.ones_like(signal)

    # Compute derivative using finite differences
    derivative = np.abs(np.gradient(signal_flat))

    # Normalize to [0, 1] range
    max_deriv = np.max(derivative) + 1e-12
    normalized_deriv = derivative / max_deriv

    # Compute weights: higher derivative -> higher weight
    weights = 1.0 + alpha * normalized_deriv

    # Reshape to original shape
    weights = weights.reshape(original_shape)

    return weights.astype(np.float32)


def compute_snr(clean: np.ndarray, noisy: np.ndarray, eps: float = 1e-12) -> float:
    """Compute Signal-to-Noise Ratio (SNR) in dB.

    SNR = 10 * log10(sum(clean^2) / sum((clean - noisy)^2))

    Args:
        clean: Clean signal
        noisy: Noisy signal
        eps: Small constant for numerical stability

    Returns:
        SNR value in dB
    """
    clean = clean.flatten()
    noisy = noisy.flatten()

    signal_power = np.sum(clean ** 2) + eps
    noise_power = np.sum((clean - noisy) ** 2) + eps

    snr_db = 10.0 * np.log10(signal_power / noise_power)

    return float(snr_db)


def compute_compression_ratio(
    original_size: int,
    compressed_size: int
) -> float:
    """Compute compression ratio.

    CR = original_size / compressed_size

    Args:
        original_size: Size of original signal (in bits or samples)
        compressed_size: Size of compressed representation

    Returns:
        Compression ratio
    """
    return float(original_size) / (float(compressed_size) + 1e-12)


def evaluate_reconstruction(
    clean: np.ndarray,
    reconstructed: np.ndarray,
    noisy: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None,
    original_bits: int = 11,
    latent_dim: Optional[int] = None,
    latent_bits: int = 8,
) -> Dict[str, float]:
    """Comprehensive evaluation of reconstruction quality.

    Args:
        clean: Clean ECG signal
        reconstructed: Reconstructed ECG signal
        noisy: Noisy input signal (optional, for SNR improvement calculation)
        weights: WWPRD weights (optional)
        original_bits: Bits per sample in original signal
        latent_dim: Latent dimension (for CR calculation)
        latent_bits: Bits per latent variable

    Returns:
        Dictionary of metrics
    """
    metrics = {}

    # PRD
    metrics['PRD'] = compute_prd(clean, reconstructed)

    # WWPRD
    metrics['WWPRD'] = compute_wwprd(clean, reconstructed, weights)

    # SNR of reconstruction
    metrics['SNR_out'] = compute_snr(clean, reconstructed)

    # SNR improvement if noisy input provided
    if noisy is not None:
        snr_in = compute_snr(clean, noisy)
        metrics['SNR_in'] = snr_in
        metrics['SNR_improvement'] = metrics['SNR_out'] - snr_in

    # Compression ratio
    if latent_dim is not None:
        signal_length = len(clean.flatten())
        original_size = signal_length * original_bits
        compressed_size = latent_dim * latent_bits
        metrics['CR'] = compute_compression_ratio(original_size, compressed_size)

    # Quality classification based on PRD
    if metrics['PRD'] < 4.33:
        metrics['PRD_quality'] = 'Excellent'
    elif metrics['PRD'] < 9.00:
        metrics['PRD_quality'] = 'Very Good'
    elif metrics['PRD'] < 15.00:
        metrics['PRD_quality'] = 'Good'
    else:
        metrics['PRD_quality'] = 'Not Good'

    # Quality classification based on WWPRD
    if metrics['WWPRD'] < 7.4:
        metrics['WWPRD_quality'] = 'Excellent'
    elif metrics['WWPRD'] < 14.8:
        metrics['WWPRD_quality'] = 'Very Good'
    elif metrics['WWPRD'] < 24.7:
        metrics['WWPRD_quality'] = 'Good'
    else:
        metrics['WWPRD_quality'] = 'Not Good'

    return metrics


def batch_evaluate(
    clean_batch: torch.Tensor,
    recon_batch: torch.Tensor,
    noisy_batch: Optional[torch.Tensor] = None,
    weights: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """Evaluate a batch of signals and return averaged metrics.

    Args:
        clean_batch: Clean signals (B, C, T)
        recon_batch: Reconstructed signals (B, C, T)
        noisy_batch: Noisy signals (B, C, T), optional
        weights: Weights for WWPRD (B, C, T), optional

    Returns:
        Dictionary of averaged metrics including:
        - PRD: PRD vs clean signal (original definition)
        - PRD_vs_noisy: PRD vs noisy signal (denoising improvement)
        - WWPRD: WWPRD vs clean signal
        - WWPRD_vs_noisy: WWPRD vs noisy signal
    """
    batch_size = clean_batch.shape[0]

    all_prds = []  # PRD vs clean
    all_prds_vs_noisy = []  # PRD vs noisy
    all_wwprds = []  # WWPRD vs clean
    all_wwprds_vs_noisy = []  # WWPRD vs noisy
    all_snr_outs = []
    all_snr_ins = []

    for i in range(batch_size):
        clean = clean_batch[i].cpu().numpy()
        recon = recon_batch[i].cpu().numpy()

        w = None
        if weights is not None:
            w = weights[i].cpu().numpy()

        # Compute metrics vs clean (original definition)
        prd = compute_prd(clean, recon)
        wwprd = compute_wwprd(clean, recon, w)
        snr_out = compute_snr(clean, recon)

        all_prds.append(prd)
        all_wwprds.append(wwprd)
        all_snr_outs.append(snr_out)

        if noisy_batch is not None:
            noisy = noisy_batch[i].cpu().numpy()
            snr_in = compute_snr(clean, noisy)
            all_snr_ins.append(snr_in)

            # Compute metrics vs noisy (denoising improvement)
            prd_vs_noisy = compute_prd(noisy, recon)
            wwprd_vs_noisy = compute_wwprd(noisy, recon, w)
            all_prds_vs_noisy.append(prd_vs_noisy)
            all_wwprds_vs_noisy.append(wwprd_vs_noisy)

    metrics = {
        'PRD': np.mean(all_prds),
        'PRD_std': np.std(all_prds),
        'WWPRD': np.mean(all_wwprds),
        'WWPRD_std': np.std(all_wwprds),
        'SNR_out': np.mean(all_snr_outs),
        'SNR_out_std': np.std(all_snr_outs),
    }

    # Add PRD vs noisy if available
    if len(all_prds_vs_noisy) > 0:
        metrics['PRD_vs_noisy'] = np.mean(all_prds_vs_noisy)
        metrics['PRD_vs_noisy_std'] = np.std(all_prds_vs_noisy)
        metrics['WWPRD_vs_noisy'] = np.mean(all_wwprds_vs_noisy)
        metrics['WWPRD_vs_noisy_std'] = np.std(all_wwprds_vs_noisy)

    if len(all_snr_ins) > 0:
        metrics['SNR_in'] = np.mean(all_snr_ins)
        metrics['SNR_improvement'] = metrics['SNR_out'] - metrics['SNR_in']

    return metrics


def format_metrics(metrics: Dict[str, float], title: str = "Evaluation Metrics") -> str:
    """Format metrics dictionary as a readable string.

    Args:
        metrics: Dictionary of metrics
        title: Title for the metrics report

    Returns:
        Formatted string
    """
    lines = [f"\n{'='*60}", f"{title:^60}", f"{'='*60}"]

    for key, value in metrics.items():
        if isinstance(value, str):
            lines.append(f"{key:20s}: {value}")
        elif isinstance(value, float):
            if 'PRD' in key or 'SNR' in key or 'CR' in key:
                lines.append(f"{key:20s}: {value:8.2f}")
            else:
                lines.append(f"{key:20s}: {value:8.4f}")
        else:
            lines.append(f"{key:20s}: {value}")

    lines.append("=" * 60)

    return "\n".join(lines)

