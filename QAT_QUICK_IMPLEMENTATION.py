"""
QUICK QAT IMPLEMENTATION - Copy-paste ready code for Day 1

Add these functions to ecgdae/quantization.py
Then modify scripts/train_mitbih.py as shown below
"""

# ============================================================================
# STEP 1: Add to ecgdae/quantization.py (at the end of the file)
# ============================================================================

def quantize_with_ste(
    latent: torch.Tensor,
    quantization_bits: int,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
) -> torch.Tensor:
    """Quantize latent with Straight-Through Estimator for training.

    Forward pass: quantizes the latent
    Backward pass: passes gradients through as if quantization was identity

    Args:
        latent: Latent tensor (B, C, T) or (B, C*T)
        quantization_bits: Number of quantization bits
        min_val: Optional min value (if None, computed from latent)
        max_val: Optional max value (if None, computed from latent)

    Returns:
        Dequantized latent tensor (same shape as input)
    """
    # Store original for STE
    latent_flat = latent.flatten(1)
    original_shape = latent.shape

    # Compute quantization range
    if min_val is None:
        min_val = latent_flat.min().item()
    if max_val is None:
        max_val = latent_flat.max().item()

    if max_val == min_val:
        max_val = min_val + 1e-6

    # Quantize (forward pass)
    n_levels = 2 ** quantization_bits
    normalized = (latent_flat - min_val) / (max_val - min_val)
    normalized = torch.clamp(normalized, 0.0, 1.0)
    quantized_int = torch.round(normalized * (n_levels - 1))

    # Dequantize
    quantized_normalized = quantized_int / (n_levels - 1)
    dequantized = min_val + quantized_normalized * (max_val - min_val)
    dequantized = dequantized.reshape(original_shape)

    # Straight-Through Estimator: forward uses quantized, backward uses identity
    return latent + (dequantized - latent).detach()


def add_quantization_noise(
    latent: torch.Tensor,
    quantization_bits: int,
    noise_scale: float = 1.0,
) -> torch.Tensor:
    """Add uniform quantization noise to simulate quantization during training.

    Alternative to exact quantization - adds noise with same variance as quantization error.
    Sometimes more stable during training.

    Args:
        latent: Latent tensor
        quantization_bits: Target quantization bits
        noise_scale: Scale of noise (1.0 = full quantization noise)

    Returns:
        Noisy latent tensor
    """
    n_levels = 2 ** quantization_bits
    quantization_step = 1.0 / n_levels

    # Uniform noise in range [-step/2, step/2]
    noise = (torch.rand_like(latent) - 0.5) * quantization_step * noise_scale

    return latent + noise


# ============================================================================
# STEP 2: Modify scripts/train_mitbih.py
# ============================================================================

# 2.1: Add to setup_args() function (around line 100):

    # QAT parameters
    parser.add_argument("--quantization_aware", action="store_true",
                        help="Enable quantization-aware training")
    parser.add_argument("--quantization_bits", type=int, default=4,
                        help="Quantization bits for QAT (default: 4)")
    parser.add_argument("--qat_probability", type=float, default=0.5,
                        help="Probability of applying QAT per batch (default: 0.5)")
    parser.add_argument("--qat_mode", type=str, default="ste", choices=["ste", "noise"],
                        help="QAT mode: 'ste' (straight-through) or 'noise' (quantization noise)")


# 2.2: Modify train_epoch() function (around line 247):

def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_weights: bool = False,
    weight_alpha: float = 2.0,
    quantization_aware: bool = False,  # ADD THIS
    quantization_bits: int = 4,        # ADD THIS
    qat_probability: float = 0.5,      # ADD THIS
    qat_mode: str = "ste",             # ADD THIS
) -> Dict[str, float]:
    """Train for one epoch."""
    import random  # ADD THIS at top of function

    model.train()
    total_loss = 0.0
    num_batches = 0

    requires_latent = getattr(loss_fn, "requires_latent", False)

    # Import QAT functions
    if quantization_aware:  # ADD THIS BLOCK
        from ecgdae.quantization import quantize_with_ste, add_quantization_noise

    for noisy, clean in train_loader:
        noisy = noisy.to(device)
        clean = clean.to(device)

        # Forward pass
        if requires_latent or quantization_aware:  # MODIFY THIS LINE
            recon, latents = model(noisy, return_latent=True)

            # Apply quantization-aware training
            if quantization_aware and random.random() < qat_probability:  # ADD THIS BLOCK
                if qat_mode == "ste":
                    latents = quantize_with_ste(latents, quantization_bits)
                else:  # noise mode
                    latents = add_quantization_noise(latents, quantization_bits)
                # Reconstruct with quantized latents
                recon = model.decode(latents)
        else:
            recon = model(noisy)
            latents = None

        # Compute loss (rest stays the same)
        weights = None
        if use_weights:
            weights = compute_wwprd_weights(clean, weight_alpha)
        if requires_latent:
            loss = loss_fn(clean, recon, latents, weights)
        elif use_weights:
            loss = loss_fn(clean, recon, weights)
        else:
            loss = loss_fn(clean, recon)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return {"train_loss": total_loss / num_batches}


# 2.3: Update main() function to pass QAT arguments (around line 400-450):

    # Find the train_epoch call and modify it:
    train_metrics = train_epoch(
        model, train_loader, loss_fn, optimizer, device,
        use_weights=(args.loss_type in ["wwprd", "combined", "stft_wwprd", "wwprd_l1"]),
        weight_alpha=args.weight_alpha,
        quantization_aware=args.quantization_aware,      # ADD THIS
        quantization_bits=args.quantization_bits,         # ADD THIS
        qat_probability=args.qat_probability,              # ADD THIS
        qat_mode=args.qat_mode,                           # ADD THIS
    )


# ============================================================================
# STEP 3: Test the implementation
# ============================================================================

# Quick test command:
# python scripts/train_mitbih.py --loss_type wwprd --latent_dim 4 --epochs 5 --quantization_aware --quantization_bits 4 --qat_probability 0.5 --output_dir outputs/test_qat

# If it runs without errors, you're good to go!


# ============================================================================
# STEP 4: Full training command (Day 1)
# ============================================================================

"""
python scripts/train_mitbih.py `
    --loss_type wwprd `
    --latent_dim 4 `
    --epochs 150 `
    --quantization_aware `
    --quantization_bits 4 `
    --qat_probability 0.5 `
    --qat_mode ste `
    --output_dir outputs/wwprd_latent4_qat_week1
"""

