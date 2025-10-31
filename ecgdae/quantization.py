"""Quantization functions for ECG compression.

This module implements quantization and compression ratio calculation
for Week 2 deliverables.

Author: Person A
Date: October 2025
"""

import numpy as np
import torch
from typing import Union, Tuple, Optional


def uniform_quantize(
    values: Union[np.ndarray, torch.Tensor],
    n_bits: int,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
) -> Tuple[Union[np.ndarray, torch.Tensor], float, float]:
    """Uniform quantization to n_bits.

    Quantizes values to 2^n_bits discrete levels using uniform quantization.

    Args:
        values: Input values to quantize (numpy array or torch tensor)
        n_bits: Number of quantization bits (4, 6, or 8 typically)
        min_val: Minimum value for quantization range (if None, uses min of values)
        max_val: Maximum value for quantization range (if None, uses max of values)

    Returns:
        Tuple of (quantized_values, min_val, max_val)
        - quantized_values: Integer-quantized values [0, 2^n_bits - 1]
        - min_val: Minimum value used for quantization
        - max_val: Maximum value used for quantization

    Example:
        >>> values = np.array([-1.0, 0.0, 1.0, 2.0])
        >>> quantized, min_v, max_v = uniform_quantize(values, n_bits=4)
        >>> quantized
        array([0, 5, 10, 15])  # 16 levels (2^4)
    """
    is_torch = isinstance(values, torch.Tensor)

    if is_torch:
        values_np = values.detach().cpu().numpy()
    else:
        values_np = np.array(values)

    # Determine quantization range
    if min_val is None:
        min_val = float(np.min(values_np))
    if max_val is None:
        max_val = float(np.max(values_np))

    # Avoid division by zero
    if max_val == min_val:
        max_val = min_val + 1e-6

    # Number of quantization levels
    n_levels = 2 ** n_bits

    # Normalize to [0, 1]
    normalized = (values_np - min_val) / (max_val - min_val)

    # Clamp to [0, 1]
    normalized = np.clip(normalized, 0.0, 1.0)

    # Quantize to integers [0, n_levels - 1]
    quantized = np.round(normalized * (n_levels - 1)).astype(np.int32)

    # Convert back to original format
    if is_torch:
        return torch.from_numpy(quantized).to(values.device), min_val, max_val
    else:
        return quantized, min_val, max_val


def dequantize(
    quantized: Union[np.ndarray, torch.Tensor],
    min_val: float,
    max_val: float,
    n_bits: int,
) -> Union[np.ndarray, torch.Tensor]:
    """Dequantize integer-quantized values back to continuous range.

    Reverse operation of uniform_quantize. Reconstructs continuous values
    from quantized integers.

    Args:
        quantized: Integer-quantized values [0, 2^n_bits - 1]
        min_val: Minimum value of original range
        max_val: Maximum value of original range
        n_bits: Number of quantization bits used

    Returns:
        Dequantized values in continuous range [min_val, max_val]

    Example:
        >>> quantized = np.array([0, 5, 10, 15])
        >>> dequantized = dequantize(quantized, min_val=-1.0, max_val=2.0, n_bits=4)
        >>> dequantized
        array([-1.0, -0.333, 0.333, 2.0])  # Approximately
    """
    is_torch = isinstance(quantized, torch.Tensor)

    if is_torch:
        # Ensure we have the right dtype - convert to float32
        quantized_np = quantized.detach().cpu().numpy().astype(np.float32)
        original_dtype = quantized.dtype
        original_device = quantized.device
    else:
        quantized_np = quantized.astype(np.float32)

    n_levels = 2 ** n_bits

    # Normalize to [0, 1]
    normalized = quantized_np / (n_levels - 1)

    # Scale back to [min_val, max_val]
    dequantized = min_val + normalized * (max_val - min_val)

    if is_torch:
        # Convert back to tensor with matching dtype
        result = torch.from_numpy(dequantized).to(original_device)
        # Ensure dtype matches original latent dtype (float32)
        if original_dtype != result.dtype:
            result = result.to(original_dtype)
        return result
    else:
        return dequantized


def compute_compression_ratio(
    original_shape: Tuple[int, ...],
    latent_shape: Tuple[int, ...],
    quantization_bits: int = 8,
    original_bits_per_sample: int = 11,
) -> float:
    """Compute compression ratio (CR).

    CR = original_bits / compressed_bits

    where:
    - original_bits = original_length * original_bits_per_sample
    - compressed_bits = latent_dim * latent_length * quantization_bits

    Args:
        original_shape: Shape of original signal (e.g., (512,) or (1, 512))
        latent_shape: Shape of latent representation (e.g., (32, 32) for (channels, length))
        quantization_bits: Bits per latent variable after quantization
        original_bits_per_sample: Bits per sample in original signal (typically 11 for ECG)

    Returns:
        Compression ratio (float)

    Example:
        >>> original_shape = (512,)  # 512 samples
        >>> latent_shape = (32, 32)   # 32 channels, 32 length
        >>> cr = compute_compression_ratio(original_shape, latent_shape, quantization_bits=8)
        >>> cr  # (512 * 11) / (32 * 32 * 8) ≈ 5.5:1
    """
    # Original signal size (in bits)
    original_length = original_shape[-1]  # Last dimension is signal length
    original_bits = original_length * original_bits_per_sample

    # Compressed size (in bits)
    if len(latent_shape) == 1:
        # 1D latent: (length,)
        latent_size = latent_shape[0]
    elif len(latent_shape) == 2:
        # 2D latent: (channels, length)
        latent_size = latent_shape[0] * latent_shape[1]
    else:
        # Flatten all dimensions except first (batch)
        latent_size = int(np.prod(latent_shape[1:]))

    compressed_bits = latent_size * quantization_bits

    # Avoid division by zero
    if compressed_bits == 0:
        return 0.0

    cr = original_bits / compressed_bits

    return float(cr)


def quantize_latent(
    latent: torch.Tensor,
    quantization_bits: int,
    return_metadata: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
    """Quantize latent representation from autoencoder.

    Convenience function that quantizes the latent tensor from model.encode(),
    handling both per-channel and global quantization.

    Args:
        latent: Latent tensor from encoder (B, C, T) or (B, C*T)
        quantization_bits: Number of quantization bits
        return_metadata: If True, also return min/max for dequantization

    Returns:
        Quantized latent tensor, or (quantized, metadata) if return_metadata=True
        metadata contains: {'min_val': float, 'max_val': float, 'quantization_bits': int}
    """
    # Flatten spatial dimensions if needed
    original_shape = latent.shape
    latent_flat = latent.flatten(1)  # (B, C*T)

    batch_size = latent_flat.shape[0]
    quantized_batch = []
    metadata_batch = []

    for b in range(batch_size):
        # Quantize each sample independently (preserves sample-specific dynamic range)
        values = latent_flat[b].detach().cpu().numpy()
        quantized, min_val, max_val = uniform_quantize(values, n_bits=quantization_bits)

        # Store metadata for dequantization
        metadata = {
            'min_val': min_val,
            'max_val': max_val,
            'quantization_bits': quantization_bits
        }

        quantized_batch.append(torch.from_numpy(quantized).to(latent.device))
        metadata_batch.append(metadata)

    # Stack and reshape - keep as integer type for quantization representation
    quantized = torch.stack(quantized_batch)  # (B, C*T)
    quantized = quantized.reshape(original_shape)  # Restore original shape

    # Convert to float for dequantization (we'll use these values as integers conceptually)
    # But store as float so operations work smoothly
    quantized = quantized.float()

    if return_metadata:
        # Use first sample's metadata as representative (or average)
        avg_min = np.mean([m['min_val'] for m in metadata_batch])
        avg_max = np.mean([m['max_val'] for m in metadata_batch])
        metadata = {
            'min_val': avg_min,
            'max_val': avg_max,
            'quantization_bits': quantization_bits
        }
        return quantized, metadata
    else:
        return quantized


def dequantize_latent(
    quantized: torch.Tensor,
    metadata: dict,
) -> torch.Tensor:
    """Dequantize latent representation.

    Reverse of quantize_latent. Reconstructs continuous latent values
    from quantized integers using stored metadata.

    Args:
        quantized: Quantized latent tensor
        metadata: Dictionary with 'min_val', 'max_val', 'quantization_bits'

    Returns:
        Dequantized latent tensor (continuous values)
    """
    return dequantize(
        quantized,
        metadata['min_val'],
        metadata['max_val'],
        metadata['quantization_bits']
    )

