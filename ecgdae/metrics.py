"""Evaluation metrics for ECG compression and denoising."""

import numpy as np
import torch
from typing import Tuple, Dict, Optional

# Optional dependency for wavelet-based WWPRD (per Kovács et al.)
try:
    import pywt  # type: ignore
    _HAS_PYWT = True
except Exception:  # pragma: no cover - optional
    _HAS_PYWT = False


def compute_prd(clean: np.ndarray, reconstructed: np.ndarray, eps: float = 1e-12) -> float:
    """Compute Percent Root-mean-square Difference (PRD).
    
    PRD = 100 * sqrt(sum((clean - recon)^2) / sum(clean^2))
    
    Lower is better.
    
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


def compute_prdn(clean: np.ndarray, reconstructed: np.ndarray, eps: float = 1e-12) -> float:
    """Compute normalized PRD (PRDN) with mean removed from reference.

    PRDN = 100 * sqrt( sum((clean - recon)^2) / sum( (clean - mean(clean))^2 ) )

    This matches the normalized PRD used in Kovács et al. (ECG compression).
    """
    clean = clean.flatten()
    reconstructed = reconstructed.flatten()

    numerator = np.sum((clean - reconstructed) ** 2)
    ref_zm = clean - np.mean(clean)
    denominator = np.sum(ref_zm ** 2) + eps
    prdn = 100.0 * np.sqrt(numerator / denominator)
    return float(prdn)


def compute_wwprd(
    clean: np.ndarray, 
    reconstructed: np.ndarray,
    weights: Optional[np.ndarray] = None,
    eps: float = 1e-12
) -> float:
    """Compute derivative-weighted PRD (legacy WWPRD variant).
    
    WWPRD ≈ 100 * sqrt(sum(w * (clean - recon)^2) / sum(w * clean^2))
    
    where w are time-domain weights emphasizing QRS via derivatives.
    
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


def compute_wwprd_wavelet(
    clean: np.ndarray,
    reconstructed: np.ndarray,
    wavelet: str = "db3",
    level: int = 5,
    weights: Optional[np.ndarray] = None,
    eps: float = 1e-12,
) -> float:
    """Compute wavelet-based WWPRD per Kovács et al. (Eq. 20).

    Steps:
      1) Remove mean from both signals.
      2) Perform DWT up to `level` with `wavelet`.
      3) Form coefficient groups [D1, D2, D3, D4, D5, A5] (6 groups).
      4) WWPRD = 100 * sqrt( sum_j w_j * ||c_j - ĉ_j||^2 / ||c_j||^2 ).

    Default weights (paper): [6/27, 9/27, 7/27, 3/27, 1/27, 1/27].
    """
    if not _HAS_PYWT:
        raise ImportError(
            "PyWavelets (pywt) is required for wavelet-based WWPRD. Please install 'PyWavelets'."
        )

    x = clean.flatten().astype(np.float64)
    y = reconstructed.flatten().astype(np.float64)

    # Zero-mean as per paper
    x = x - np.mean(x)
    y = y - np.mean(y)

    # Wavelet decomposition
    cA_x, details_x = _dwt_groups(x, wavelet=wavelet, level=level)
    cA_y, details_y = _dwt_groups(y, wavelet=wavelet, level=level)

    # Assemble groups in order [D1..D5, A5]
    coeff_groups_x = details_x + [cA_x]
    coeff_groups_y = details_y + [cA_y]

    # Default weights if not provided (6 groups)
    if weights is None:
        weights = np.array([6/27, 9/27, 7/27, 3/27, 1/27, 1/27], dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    assert len(weights) == 6, "Expected 6 weights for [D1..D5, A5]."

    num = 0.0
    den = 0.0
    for wj, cx, cy in zip(weights, coeff_groups_x, coeff_groups_y):
        if wj <= 0:
            continue
        diff = cx - cy
        num += wj * float(np.sum(diff * diff))
        den += wj * (float(np.sum(cx * cx)) + eps)

    wwprd = 100.0 * np.sqrt(num / (den + eps))
    return float(wwprd)


def _dwt_groups(signal: np.ndarray, wavelet: str = "db3", level: int = 5):
    """Helper: returns (A_level, [D1, D2, ..., D_level]) with consistent lengths."""
    coeffs = pywt.wavedec(signal, wavelet=wavelet, level=level, mode="periodization")
    # coeffs = [A_level, D_level, D_{level-1}, ..., D1]
    cA = coeffs[0]
    Ds = coeffs[1:][::-1]  # -> [D1, D2, ..., D_level]
    # Convert to numpy arrays (float64) for safety
    cA = np.asarray(cA, dtype=np.float64)
    Ds = [np.asarray(d, dtype=np.float64) for d in Ds]
    return cA, Ds


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

    # PRD (unnormalized) and PRDN (normalized, per paper)
    metrics['PRD'] = compute_prd(clean, reconstructed)
    metrics['PRDN'] = compute_prdn(clean, reconstructed)

    # WWPRD (wavelet-based, per paper) and derivative variant for reference
    try:
        metrics['WWPRD'] = compute_wwprd_wavelet(clean, reconstructed)
    except Exception:
        # Fallback if PyWavelets is missing
        metrics['WWPRD'] = float('nan')
    metrics['WWPRD_deriv'] = compute_wwprd(clean, reconstructed, weights)
    
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
    
    # Quality classification based on PRDN (Table III)
    prdn = metrics['PRDN']
    if prdn < 4.33:
        metrics['PRDN_quality'] = 'Excellent'
    elif prdn < 7.8:
        metrics['PRDN_quality'] = 'V. Good'
    elif prdn < 11.59:
        metrics['PRDN_quality'] = 'Good'
    elif prdn < 22.5:
        metrics['PRDN_quality'] = 'Not Bad'
    else:
        metrics['PRDN_quality'] = 'Bad'

    # Quality classification based on WWPRD (Table III)
    ww = metrics['WWPRD']
    if np.isfinite(ww):
        if ww < 7.4:
            metrics['WWPRD_quality'] = 'Excellent'
        elif ww < 15.45:
            metrics['WWPRD_quality'] = 'V. Good'
        elif ww < 25.18:
            metrics['WWPRD_quality'] = 'Good'
        elif ww < 37.4:
            metrics['WWPRD_quality'] = 'Not Bad'
        else:
            metrics['WWPRD_quality'] = 'Bad'
    
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
        Dictionary of averaged metrics
    """
    batch_size = clean_batch.shape[0]
    
    all_prds = []
    all_prdns = []
    all_wwprds = []  # wavelet-based
    all_wwprds_deriv = []
    all_snr_outs = []
    all_snr_ins = []
    
    for i in range(batch_size):
        clean = clean_batch[i].cpu().numpy()
        recon = recon_batch[i].cpu().numpy()
        
        w = None
        if weights is not None:
            w = weights[i].cpu().numpy()
        
        # Compute metrics
        prd = compute_prd(clean, recon)
        prdn = compute_prdn(clean, recon)
        try:
            wwprd = compute_wwprd_wavelet(clean, recon)
        except Exception:
            wwprd = float('nan')
        wwprd_deriv = compute_wwprd(clean, recon, w)
        snr_out = compute_snr(clean, recon)
        
        all_prds.append(prd)
        all_prdns.append(prdn)
        all_wwprds.append(wwprd)
        all_wwprds_deriv.append(wwprd_deriv)
        all_snr_outs.append(snr_out)
        
        if noisy_batch is not None:
            noisy = noisy_batch[i].cpu().numpy()
            snr_in = compute_snr(clean, noisy)
            all_snr_ins.append(snr_in)
    
    metrics = {
        'PRD': np.mean(all_prds),
        'PRD_std': np.std(all_prds),
        'PRDN': np.mean(all_prdns),
        'PRDN_std': np.std(all_prdns),
        'WWPRD': np.nanmean(all_wwprds),
        'WWPRD_std': np.nanstd(all_wwprds),
        'WWPRD_deriv': np.mean(all_wwprds_deriv),
        'WWPRD_deriv_std': np.std(all_wwprds_deriv),
        'SNR_out': np.mean(all_snr_outs),
        'SNR_out_std': np.std(all_snr_outs),
    }
    
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

