"""Test script to verify Week 1 setup.

Quick smoke test to ensure all components are working correctly.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from rich.console import Console
from rich.panel import Panel

console = Console()


def test_imports():
    """Test that all required packages can be imported."""
    console.print("\n[bold cyan]Testing imports...[/bold cyan]")
    
    try:
        import torch
        import numpy
        import wfdb
        import matplotlib
        import scipy
        import seaborn
        console.print("[green]✓ All required packages imported successfully")
        return True
    except ImportError as e:
        console.print(f"[red]✗ Import failed: {e}")
        return False


def test_losses():
    """Test loss functions."""
    console.print("\n[bold cyan]Testing loss functions...[/bold cyan]")
    
    try:
        from ecgdae.losses import PRDLoss, WWPRDLoss, STFTWeightedWWPRDLoss
        
        # Create sample data
        x = torch.randn(4, 1, 720)
        x_hat = x + 0.1 * torch.randn_like(x)
        
        # Test PRD
        prd_loss = PRDLoss()
        prd_value = prd_loss(x, x_hat)
        console.print(f"  PRD Loss: {prd_value.item():.2f}")
        
        # Test WWPRD
        wwprd_loss = WWPRDLoss()
        wwprd_value = wwprd_loss(x, x_hat)
        console.print(f"  WWPRD Loss: {wwprd_value.item():.2f}")
        
        # Test STFT WWPRD
        stft_loss = STFTWeightedWWPRDLoss()
        stft_value = stft_loss(x, x_hat)
        console.print(f"  STFT WWPRD Loss: {stft_value.item():.2f}")
        
        console.print("[green]✓ All loss functions working correctly")
        return True
    except Exception as e:
        console.print(f"[red]✗ Loss function test failed: {e}")
        return False


def test_models():
    """Test model architectures."""
    console.print("\n[bold cyan]Testing models...[/bold cyan]")
    
    try:
        from ecgdae.models import ConvAutoEncoder, ResidualAutoEncoder, count_parameters
        
        # Test ConvAutoEncoder
        model1 = ConvAutoEncoder(in_channels=1, hidden_dims=(16, 32, 64), latent_dim=16)
        x = torch.randn(2, 1, 720)
        z = model1.encode(x)
        x_recon = model1(x)
        
        console.print(f"  ConvAutoEncoder:")
        console.print(f"    Input: {x.shape} → Latent: {z.shape} → Output: {x_recon.shape}")
        console.print(f"    Parameters: {count_parameters(model1):,}")
        
        # Test ResidualAutoEncoder
        model2 = ResidualAutoEncoder(in_channels=1, hidden_dims=(16, 32, 64), latent_dim=16)
        z2 = model2.encode(x)
        x_recon2 = model2(x)
        
        console.print(f"  ResidualAutoEncoder:")
        console.print(f"    Input: {x.shape} → Latent: {z2.shape} → Output: {x_recon2.shape}")
        console.print(f"    Parameters: {count_parameters(model2):,}")
        
        console.print("[green]✓ All models working correctly")
        return True
    except Exception as e:
        console.print(f"[red]✗ Model test failed: {e}")
        return False


def test_metrics():
    """Test evaluation metrics."""
    console.print("\n[bold cyan]Testing metrics...[/bold cyan]")
    
    try:
        from ecgdae.metrics import (
            compute_prd, compute_wwprd, compute_snr,
            compute_derivative_weights, evaluate_reconstruction
        )
        
        # Create sample signals
        clean = np.random.randn(1, 720).astype(np.float32)
        noisy = clean + 0.1 * np.random.randn(*clean.shape).astype(np.float32)
        recon = clean + 0.05 * np.random.randn(*clean.shape).astype(np.float32)
        
        # Compute metrics
        prd = compute_prd(clean, recon)
        wwprd = compute_wwprd(clean, recon)
        snr = compute_snr(clean, recon)
        
        console.print(f"  PRD: {prd:.2f}%")
        console.print(f"  WWPRD: {wwprd:.2f}%")
        console.print(f"  SNR: {snr:.2f} dB")
        
        # Test weights computation
        weights = compute_derivative_weights(clean, alpha=2.0)
        console.print(f"  Weights shape: {weights.shape}")
        console.print(f"  Weights range: [{weights.min():.2f}, {weights.max():.2f}]")
        
        console.print("[green]✓ All metrics working correctly")
        return True
    except Exception as e:
        console.print(f"[red]✗ Metrics test failed: {e}")
        return False


def test_data_loading():
    """Test data loading (without actually downloading)."""
    console.print("\n[bold cyan]Testing data loading modules...[/bold cyan]")
    
    try:
        from ecgdae.data import (
            WindowingConfig, ArrayECGDataset, gaussian_snr_mixer,
            MITBIHLoader, NSTDBNoiseMixer, MITBIHDataset
        )
        
        # Test windowing config
        config = WindowingConfig(sample_rate=360, window_seconds=2.0)
        console.print(f"  Window size: {config.window_size} samples")
        console.print(f"  Step size: {config.step_size} samples")
        
        # Test Gaussian noise mixer
        mixer = gaussian_snr_mixer(10.0)
        test_signal = np.random.randn(720).astype(np.float32)
        noisy_signal = mixer(test_signal)
        console.print(f"  Gaussian mixer created (SNR: 10 dB)")
        console.print(f"  Signal shape: {test_signal.shape} → {noisy_signal.shape}")
        
        # Test ArrayECGDataset
        signals = [np.random.randn(3600).astype(np.float32) for _ in range(2)]
        dataset = ArrayECGDataset(signals, config, noise_mixer=mixer)
        console.print(f"  ArrayECGDataset: {len(dataset)} windows")
        
        # Get one sample
        x, y = dataset[0]
        console.print(f"  Sample shape: noisy={x.shape}, clean={y.shape}")
        
        console.print("[green]✓ All data loading modules working correctly")
        console.print("[yellow]  Note: MIT-BIH and NSTDB loaders not tested (require network)")
        return True
    except Exception as e:
        console.print(f"[red]✗ Data loading test failed: {e}")
        return False


def test_device():
    """Test CUDA availability."""
    console.print("\n[bold cyan]Testing compute device...[/bold cyan]")
    
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        device_name = torch.cuda.get_device_name(0)
        console.print(f"[green]✓ CUDA available: {device_name}")
    else:
        console.print("[yellow]⚠ CUDA not available, will use CPU")
    
    return True


def main():
    """Run all tests."""
    console.print(Panel.fit(
        "[bold cyan]Week 1 Setup Test[/bold cyan]\n"
        "Testing all components for ECG denoising and compression",
        border_style="cyan"
    ))
    
    tests = [
        ("Imports", test_imports),
        ("Loss Functions", test_losses),
        ("Models", test_models),
        ("Metrics", test_metrics),
        ("Data Loading", test_data_loading),
        ("Device", test_device),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            console.print(f"[red]Unexpected error in {name}: {e}")
            results.append((name, False))
    
    # Summary
    console.print("\n" + "="*60)
    console.print("[bold]Test Summary:[/bold]")
    console.print("="*60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for name, success in results:
        status = "[green]✓ PASS" if success else "[red]✗ FAIL"
        console.print(f"{status}[/] {name}")
    
    console.print("="*60)
    console.print(f"[bold]Results: {passed}/{total} tests passed[/bold]")
    
    if passed == total:
        console.print("\n[bold green]✓ All tests passed! Ready to train.[/bold green]")
        console.print("\n[cyan]Next steps:[/cyan]")
        console.print("  1. Run: python scripts/train_mitbih.py --num_records 5 --epochs 10")
        console.print("  2. Check outputs in ./outputs/week1/")
        return 0
    else:
        console.print("\n[bold red]✗ Some tests failed. Please check the errors above.[/bold red]")
        return 1


if __name__ == "__main__":
    exit(main())

