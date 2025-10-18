import math
from typing import Optional

import torch
from torch import nn


class PRDLoss(nn.Module):
    """Percent root-mean-square difference (PRD) loss.

    Computes 100 * sqrt( sum((x - x_hat)^2) / (sum(x^2) + eps) ).
    Differentiable w.r.t. x_hat.
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
        num = torch.sum((x - x_hat) ** 2, dim=(-1,))
        den = torch.sum(x ** 2, dim=(-1,)) + self.eps
        prd = 100.0 * torch.sqrt(num / den)
        # Reduce over feature dims if needed
        while prd.dim() > 1:
            prd = prd.mean(dim=-1)
        return prd.mean()


class WWPRDLoss(nn.Module):
    """Waveform-Weighted PRD (time-domain weights).

    w should be broadcastable to x shape and non-negative. If None, falls back to PRD.
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        x: torch.Tensor,
        x_hat: torch.Tensor,
        w: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if w is None:
            w = torch.ones_like(x)
        # Ensure non-negativity and differentiability
        w = torch.relu(w) + 1e-12
        num = torch.sum(w * (x - x_hat) ** 2, dim=(-1,))
        den = torch.sum(w * x ** 2, dim=(-1,)) + self.eps
        wwprd = 100.0 * torch.sqrt(num / den)
        while wwprd.dim() > 1:
            wwprd = wwprd.mean(dim=-1)
        return wwprd.mean()


class STFTWeightedWWPRDLoss(nn.Module):
    """Frequency-weighted WWPRD using differentiable STFT magnitudes.

    We compute PRD in the STFT magnitude domain with frequency weights `freq_weights`.
    This approximates clinically-weighted distortion focusing on bands around QRS.
    """

    def __init__(
        self,
        n_fft: int = 256,
        hop_length: Optional[int] = None,
        win_length: Optional[int] = None,
        window: Optional[torch.Tensor] = None,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length or (n_fft // 4)
        self.win_length = win_length or n_fft
        self.register_buffer(
            "window",
            window if window is not None else torch.hann_window(self.win_length),
            persistent=False,
        )
        self.eps = eps

    def forward(
        self,
        x: torch.Tensor,  # shape: (B, C, T) or (B, T)
        x_hat: torch.Tensor,
        freq_weights: Optional[torch.Tensor] = None,  # shape: (F,) or (1,F,1)
    ) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        if x_hat.dim() == 2:
            x_hat = x_hat.unsqueeze(1)

        def stft_mag(z: torch.Tensor) -> torch.Tensor:
            B, C, T = z.shape
            z_flat = z.reshape(B * C, T)
            Z = torch.stft(
                z_flat,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=self.window.to(z_flat.device),
                return_complex=True,
            )
            mag = torch.abs(Z)  # (B*C, F, S)
            return mag.view(B, C, mag.size(-2), mag.size(-1))

        X = stft_mag(x)
        Xh = stft_mag(x_hat)

        if freq_weights is None:
            # Emphasize 5–40 Hz typical QRS energy region approximately
            # Build triangular weights over frequency bins (0..fs/2 unknown → relative)
            F = X.size(-2)
            device = X.device
            freqs = torch.linspace(0, 1.0, F, device=device)
            peak = 0.2  # relative peak around low-mid band
            width = 0.2
            w = torch.exp(-0.5 * ((freqs - peak) / (width + 1e-12)) ** 2)
            freq_weights = w  # (F,)

        if freq_weights.dim() == 1:
            freq_weights = freq_weights.view(1, 1, -1, 1)

        # Weight magnitudes before PRD computation
        num = torch.sum(freq_weights * (X - Xh) ** 2, dim=(-1, -2))
        den = torch.sum(freq_weights * X ** 2, dim=(-1, -2)) + self.eps
        wwprd = 100.0 * torch.sqrt(num / den)
        while wwprd.dim() > 1:
            wwprd = wwprd.mean(dim=-1)
        return wwprd.mean()


def unit_test_losses(device: str = "cpu") -> None:
    torch.manual_seed(0)
    B, C, T = 4, 1, 1024
    x = torch.randn(B, C, T, device=device)
    x_hat = x + 0.5 * torch.randn_like(x)

    prd = PRDLoss().to(device)
    wwprd = WWPRDLoss().to(device)
    fwwprd = STFTWeightedWWPRDLoss().to(device)

    for loss_fn in [prd, wwprd, fwwprd]:
        x_hat_param = nn.Parameter(x_hat.clone())
        opt = torch.optim.SGD([x_hat_param], lr=1e-1)
        initial = loss_fn(x, x_hat_param)
        for _ in range(5):
            opt.zero_grad()
            loss = loss_fn(x, x_hat_param)
            loss.backward()
            opt.step()
        final = loss_fn(x, x_hat_param)
        # Print to demonstrate optimization progress
        print(loss_fn.__class__.__name__, float(initial), "→", float(final))


if __name__ == "__main__":
    unit_test_losses("cpu")



