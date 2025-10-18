"""
Training script for ECG denoising and compression with WWPRD loss.

This script trains a 1D convolutional autoencoder on MIT-BIH data with
waveform-weighted PRD loss for simultaneous denoising and compression.
"""

import argparse
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ecgdae.losses import PRDLoss, WWPRDLoss, STFTWeightedWWPRDLoss
from ecgdae.mitbih_loader import MITBIHConfig, create_mitbih_datasets
from ecgdae.visualization import create_overview_plot, plot_training_progress


class ECGConvAutoencoder(nn.Module):
    """
    1D Convolutional Autoencoder for ECG denoising and compression.
    
    This model compresses ECG signals through a bottleneck and reconstructs
    them, learning to both denoise and compress simultaneously.
    """
    
    def __init__(
        self,
        input_channels: int = 1,
        bottleneck_channels: int = 16,
        compression_ratio: int = 8,
    ):
        """
        Initialize the autoencoder.
        
        Args:
            input_channels: Number of input channels (1 for single-lead ECG)
            bottleneck_channels: Number of channels in the bottleneck
            compression_ratio: Target compression ratio (affects bottleneck size)
        """
        super().__init__()
        
        self.input_channels = input_channels
        self.bottleneck_channels = bottleneck_channels
        self.compression_ratio = compression_ratio
        
        # Encoder: compress the signal
        self.encoder = nn.Sequential(
            # First layer: 720 -> 360 samples
            nn.Conv1d(input_channels, 32, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(32),
            nn.GELU(),
            
            # Second layer: 360 -> 180 samples
            nn.Conv1d(32, 64, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(64),
            nn.GELU(),
            
            # Third layer: 180 -> 90 samples
            nn.Conv1d(64, bottleneck_channels, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(bottleneck_channels),
            nn.GELU(),
        )
        
        # Decoder: reconstruct the signal
        self.decoder = nn.Sequential(
            # First layer: 90 -> 180 samples
            nn.ConvTranspose1d(bottleneck_channels, 64, kernel_size=15, stride=2, padding=7, output_padding=1),
            nn.BatchNorm1d(64),
            nn.GELU(),
            
            # Second layer: 180 -> 360 samples
            nn.ConvTranspose1d(64, 32, kernel_size=15, stride=2, padding=7, output_padding=1),
            nn.BatchNorm1d(32),
            nn.GELU(),
            
            # Third layer: 360 -> 720 samples
            nn.ConvTranspose1d(32, input_channels, kernel_size=15, stride=2, padding=7, output_padding=1),
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the autoencoder.
        
        Args:
            x: Input ECG signal of shape (batch_size, channels, time)
            
        Returns:
            Reconstructed ECG signal of same shape as input
        """
        # Encode to bottleneck
        z = self.encoder(x)
        
        # Decode to reconstruction
        x_recon = self.decoder(z)
        
        return x_recon
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to bottleneck representation."""
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode bottleneck representation to reconstruction."""
        return self.decoder(z)


class ECGTrainer:
    """
    Trainer class for ECG denoising and compression model.
    
    This class handles training, validation, and evaluation of the autoencoder
    with different loss functions and metrics.
    """
    
    def __init__(
        self,
        model: ECGConvAutoencoder,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str = "cpu",
        save_dir: str = "checkpoints",
    ):
        """
        Initialize the trainer.
        
        Args:
            model: The autoencoder model to train
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            loss_fn: Loss function to use
            optimizer: Optimizer for training
            device: Device to run training on
            save_dir: Directory to save checkpoints
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Training history
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.best_val_loss = float('inf')
        
        # TensorBoard logging
        self.writer = SummaryWriter(self.save_dir / "logs")
        
        # Metrics
        self.prd_loss = PRDLoss()
        self.wwprd_loss = WWPRDLoss()
        self.stft_wwprd_loss = STFTWeightedWWPRDLoss()
    
    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (noisy, clean) in enumerate(self.train_loader):
            noisy = noisy.to(self.device)
            clean = clean.to(self.device)
            
            # Forward pass
            reconstructed = self.model(noisy)
            loss = self.loss_fn(clean, reconstructed)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Log progress
            if batch_idx % 100 == 0:
                print(f"Batch {batch_idx}/{len(self.train_loader)}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self) -> Dict[str, float]:
        """Validate the model and compute metrics."""
        self.model.eval()
        total_loss = 0.0
        total_prd = 0.0
        total_wwprd = 0.0
        total_stft_wwprd = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for noisy, clean in self.val_loader:
                noisy = noisy.to(self.device)
                clean = clean.to(self.device)
                
                # Forward pass
                reconstructed = self.model(noisy)
                
                # Compute losses
                loss = self.loss_fn(clean, reconstructed)
                prd = self.prd_loss(clean, reconstructed)
                wwprd = self.wwprd_loss(clean, reconstructed)
                stft_wwprd = self.stft_wwprd_loss(clean, reconstructed)
                
                total_loss += loss.item()
                total_prd += prd.item()
                total_wwprd += wwprd.item()
                total_stft_wwprd += stft_wwprd.item()
                num_batches += 1
        
        metrics = {
            'loss': total_loss / num_batches,
            'prd': total_prd / num_batches,
            'wwprd': total_wwprd / num_batches,
            'stft_wwprd': total_stft_wwprd / num_batches,
        }
        
        return metrics
    
    def train(self, num_epochs: int = 50, save_every: int = 10):
        """
        Train the model for specified number of epochs.
        
        Args:
            num_epochs: Number of epochs to train
            save_every: Save checkpoint every N epochs
        """
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            
            # Train
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Validate
            val_metrics = self.validate()
            val_loss = val_metrics['loss']
            self.val_losses.append(val_loss)
            
            # Log metrics
            self.writer.add_scalar('Loss/Train', train_loss, epoch)
            self.writer.add_scalar('Loss/Val', val_loss, epoch)
            self.writer.add_scalar('Metrics/PRD', val_metrics['prd'], epoch)
            self.writer.add_scalar('Metrics/WWPRD', val_metrics['wwprd'], epoch)
            self.writer.add_scalar('Metrics/STFT_WWPRD', val_metrics['stft_wwprd'], epoch)
            
            # Print progress
            epoch_time = time.time() - epoch_start
            print(f"Epoch {epoch+1}/{num_epochs}:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Val PRD: {val_metrics['prd']:.2f}%")
            print(f"  Val WWPRD: {val_metrics['wwprd']:.2f}%")
            print(f"  Val STFT WWPRD: {val_metrics['stft_wwprd']:.2f}%")
            print(f"  Time: {epoch_time:.1f}s")
            print()
            
            # Save checkpoint
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(epoch, is_best=True)
            
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(epoch)
        
        total_time = time.time() - start_time
        print(f"Training completed in {total_time:.1f}s")
        
        # Plot training progress
        plot_training_progress(
            self.train_losses,
            self.val_losses,
            title="ECG Autoencoder Training Progress",
            save_path=self.save_dir / "training_progress.png",
            show=False
        )
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
        }
        
        filename = f"checkpoint_epoch_{epoch+1}.pth"
        if is_best:
            filename = "best_model.pth"
        
        torch.save(checkpoint, self.save_dir / filename)
        print(f"Saved checkpoint: {filename}")
    
    def evaluate_and_visualize(self, num_samples: int = 5):
        """Evaluate model and create visualization plots."""
        self.model.eval()
        
        with torch.no_grad():
            for i, (noisy, clean) in enumerate(self.val_loader):
                if i >= num_samples:
                    break
                
                noisy = noisy.to(self.device)
                clean = clean.to(self.device)
                
                # Get reconstruction
                reconstructed = self.model(noisy)
                
                # Compute metrics
                prd = self.prd_loss(clean, reconstructed).item()
                wwprd = self.wwprd_loss(clean, reconstructed).item()
                stft_wwprd = self.stft_wwprd_loss(clean, reconstructed).item()
                
                # Create visualization
                create_overview_plot(
                    clean[0], noisy[0], reconstructed[0],
                    sample_rate=360,
                    prd=prd,
                    wwprd=wwprd,
                    snr=None,  # We'll compute SNR later
                    save_path=self.save_dir / f"sample_{i+1}_overview.png",
                    show=False
                )
                
                print(f"Sample {i+1}: PRD={prd:.2f}%, WWPRD={wwprd:.2f}%, STFT_WWPRD={stft_wwprd:.2f}%")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train ECG autoencoder with WWPRD loss")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--device", type=str, default="auto", help="Device to use")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Save directory")
    parser.add_argument("--loss", type=str, default="stft_wwprd", 
                       choices=["mse", "prd", "wwprd", "stft_wwprd"],
                       help="Loss function to use")
    
    args = parser.parse_args()
    
    # Set device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Create datasets
    config = MITBIHConfig()
    train_dataset, val_dataset = create_mitbih_datasets(config, add_noise=True)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Create model
    model = ECGConvAutoencoder(
        input_channels=1,
        bottleneck_channels=16,
        compression_ratio=8
    )
    
    # Create loss function
    if args.loss == "mse":
        loss_fn = nn.MSELoss()
    elif args.loss == "prd":
        loss_fn = PRDLoss()
    elif args.loss == "wwprd":
        loss_fn = WWPRDLoss()
    elif args.loss == "stft_wwprd":
        loss_fn = STFTWeightedWWPRDLoss()
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Create trainer
    trainer = ECGTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        save_dir=args.save_dir
    )
    
    # Train
    trainer.train(num_epochs=args.epochs)
    
    # Evaluate and visualize
    trainer.evaluate_and_visualize(num_samples=3)
    
    print("Training completed!")


if __name__ == "__main__":
    main()
