"""Deep autoencoder models for ECG compression and denoising."""

import torch
from torch import nn
from typing import Optional, Tuple


class ConvAutoEncoder(nn.Module):
    """1D Convolutional Autoencoder for ECG signals.
    
    Encoder progressively downsamples the signal through strided convolutions.
    Decoder upsamples through transposed convolutions.
    Bottleneck dimension controls compression ratio.
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        hidden_dims: Tuple[int, ...] = (16, 32, 64),
        latent_dim: int = 32,
        kernel_size: int = 9,
        activation: str = 'gelu',
    ):
        """Initialize ConvAutoEncoder.
        
        Args:
            in_channels: Number of input channels (typically 1 for single-lead ECG)
            hidden_dims: Tuple of hidden dimensions for encoder layers
            latent_dim: Dimension of bottleneck layer (controls compression)
            kernel_size: Kernel size for convolutions
            activation: Activation function ('relu', 'gelu', 'elu')
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        
        # Activation function
        if activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'gelu':
            self.act = nn.GELU()
        elif activation == 'elu':
            self.act = nn.ELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Encoder
        encoder_layers = []
        prev_dim = in_channels
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Conv1d(
                    prev_dim, h_dim,
                    kernel_size=kernel_size,
                    stride=2,
                    padding=kernel_size // 2
                ),
                nn.BatchNorm1d(h_dim),
                self.act,
            ])
            prev_dim = h_dim
        
        # Bottleneck
        encoder_layers.extend([
            nn.Conv1d(
                prev_dim, latent_dim,
                kernel_size=kernel_size,
                stride=2,
                padding=kernel_size // 2
            ),
            nn.BatchNorm1d(latent_dim),
            self.act,
        ])
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        
        # Reverse hidden dims for decoder
        reversed_dims = list(reversed(hidden_dims))
        
        for h_dim in reversed_dims:
            decoder_layers.extend([
                nn.ConvTranspose1d(
                    prev_dim, h_dim,
                    kernel_size=4,
                    stride=2,
                    padding=1
                ),
                nn.BatchNorm1d(h_dim),
                self.act,
            ])
            prev_dim = h_dim
        
        # Final layer to reconstruct signal
        decoder_layers.extend([
            nn.ConvTranspose1d(
                prev_dim, in_channels,
                kernel_size=4,
                stride=2,
                padding=1
            ),
        ])
        
        self.decoder = nn.Sequential(*decoder_layers)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent representation.
        
        Args:
            x: Input signal (B, C, T)
            
        Returns:
            Latent representation (B, latent_dim, T')
        """
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to signal.
        
        Args:
            z: Latent representation (B, latent_dim, T')
            
        Returns:
            Reconstructed signal (B, C, T)
        """
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through autoencoder.
        
        Args:
            x: Input signal (B, C, T)
            
        Returns:
            Reconstructed signal (B, C, T)
        """
        z = self.encode(x)
        x_recon = self.decode(z)
        
        # Ensure output matches input size
        if x_recon.shape[-1] != x.shape[-1]:
            x_recon = x_recon[..., :x.shape[-1]]
        
        return x_recon
    
    def get_latent_dim(self, input_length: int) -> int:
        """Calculate the latent sequence length for given input length.
        
        Args:
            input_length: Length of input signal
            
        Returns:
            Length of latent sequence
        """
        # Each encoder layer with stride 2 halves the length
        num_downsamples = len(self.hidden_dims) + 1  # +1 for bottleneck
        latent_length = input_length // (2 ** num_downsamples)
        return latent_length


class ResidualBlock(nn.Module):
    """Residual block for improved gradient flow."""
    
    def __init__(self, channels: int, kernel_size: int = 9):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm1d(channels)
        self.act = nn.GELU()
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm1d(channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.act(out)


class ResidualAutoEncoder(nn.Module):
    """Convolutional Autoencoder with residual connections.
    
    Similar to ConvAutoEncoder but with residual blocks for better
    gradient flow and reconstruction quality.
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        hidden_dims: Tuple[int, ...] = (32, 64, 128),
        latent_dim: int = 32,
        num_res_blocks: int = 2,
        kernel_size: int = 9,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        
        # Encoder
        encoder_layers = []
        prev_dim = in_channels
        
        for h_dim in hidden_dims:
            # Downsample
            encoder_layers.extend([
                nn.Conv1d(prev_dim, h_dim, kernel_size, stride=2, padding=kernel_size // 2),
                nn.BatchNorm1d(h_dim),
                nn.GELU(),
            ])
            
            # Residual blocks
            for _ in range(num_res_blocks):
                encoder_layers.append(ResidualBlock(h_dim, kernel_size))
            
            prev_dim = h_dim
        
        # Bottleneck
        encoder_layers.extend([
            nn.Conv1d(prev_dim, latent_dim, kernel_size, stride=2, padding=kernel_size // 2),
            nn.BatchNorm1d(latent_dim),
            nn.GELU(),
        ])
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        
        for h_dim in reversed(hidden_dims):
            # Upsample
            decoder_layers.extend([
                nn.ConvTranspose1d(prev_dim, h_dim, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm1d(h_dim),
                nn.GELU(),
            ])
            
            # Residual blocks
            for _ in range(num_res_blocks):
                decoder_layers.append(ResidualBlock(h_dim, kernel_size))
            
            prev_dim = h_dim
        
        # Final reconstruction
        decoder_layers.append(
            nn.ConvTranspose1d(prev_dim, in_channels, kernel_size=4, stride=2, padding=1)
        )
        
        self.decoder = nn.Sequential(*decoder_layers)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        x_recon = self.decode(z)
        
        # Match input size
        if x_recon.shape[-1] != x.shape[-1]:
            x_recon = x_recon[..., :x.shape[-1]]
        
        return x_recon


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_model(input_size: Tuple[int, int, int] = (4, 1, 720)) -> None:
    """Test model forward pass and print architecture info.
    
    Args:
        input_size: (batch_size, channels, length)
    """
    print("\n" + "="*60)
    print("Testing ConvAutoEncoder")
    print("="*60)
    
    model = ConvAutoEncoder(
        in_channels=1,
        hidden_dims=(16, 32, 64),
        latent_dim=32,
    )
    
    x = torch.randn(*input_size)
    z = model.encode(x)
    x_recon = model(x)
    
    print(f"Input shape:        {x.shape}")
    print(f"Latent shape:       {z.shape}")
    print(f"Reconstruction:     {x_recon.shape}")
    print(f"Parameters:         {count_parameters(model):,}")
    
    # Calculate compression ratio
    original_bits = input_size[-1] * 11  # 11 bits per sample (typical for ECG)
    latent_bits = z.shape[1] * z.shape[2] * 8  # 8 bits per latent variable
    cr = original_bits / latent_bits
    print(f"Compression ratio:  {cr:.2f}:1")
    
    print("\n" + "="*60)
    print("Testing ResidualAutoEncoder")
    print("="*60)
    
    model2 = ResidualAutoEncoder(
        in_channels=1,
        hidden_dims=(32, 64, 128),
        latent_dim=32,
        num_res_blocks=2,
    )
    
    z2 = model2.encode(x)
    x_recon2 = model2(x)
    
    print(f"Input shape:        {x.shape}")
    print(f"Latent shape:       {z2.shape}")
    print(f"Reconstruction:     {x_recon2.shape}")
    print(f"Parameters:         {count_parameters(model2):,}")
    
    latent_bits2 = z2.shape[1] * z2.shape[2] * 8
    cr2 = original_bits / latent_bits2
    print(f"Compression ratio:  {cr2:.2f}:1")


if __name__ == "__main__":
    test_model()

