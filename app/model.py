"""
model.py — BhuVistaar Core Super-Resolution Model.

Architecture:
  - Residual CNN backbone (ResidualBlock with Spatial Dropout2d)
  - Sub-pixel PixelShuffle upsampling (2x or 4x) without checkerboard artifacts
  - Multi-band input support (4 bands: Blue, Green, Red, NIR)
  - Dropout enabled at inference for Monte Carlo epistemic uncertainty sampling
  - Global bicubic residual skip connection for stable, fast convergence

Team: The Outliers | Smart India Hackathon 2026
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ResidualBlock(nn.Module):
    """
    Residual block with Spatial Dropout2d for Monte Carlo uncertainty estimation.
    Structure: Conv -> PReLU -> Dropout2d -> Conv -> Residual Add
    """
    def __init__(self, num_features: int = 64, dropout_rate: float = 0.2):
        super().__init__()
        self.conv1 = nn.Conv2d(num_features, num_features, 3, 1, 1, bias=True)
        self.act   = nn.PReLU()
        # Spatial Dropout drops full feature channels — better uncertainty signal in CNNs
        self.drop  = nn.Dropout2d(p=dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(num_features, num_features, 3, 1, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        out = self.conv1(x)
        out = self.act(out)
        out = self.drop(out)
        out = self.conv2(out)
        return out + res


class UpsampleBlock(nn.Module):
    """
    PixelShuffle sub-pixel convolution upsampler.
    2x: one PixelShuffle(2) stage.
    4x: two cascaded PixelShuffle(2) stages for stability.
    """
    def __init__(self, num_features: int = 64, scale_factor: int = 2):
        super().__init__()
        if scale_factor not in (2, 4):
            raise ValueError(f"scale_factor must be 2 or 4, got {scale_factor}")
        layers = []
        for _ in range(1 if scale_factor == 2 else 2):
            layers += [
                nn.Conv2d(num_features, num_features * 4, 3, 1, 1, bias=True),
                nn.PixelShuffle(2),
                nn.PReLU()
            ]
        self.upsample = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.upsample(x)


class ResidualSR(nn.Module):
    """
    BhuVistaar Super-Resolution Network.

    Args:
        in_channels  (int): Input spectral bands. Default 4 (Blue, Green, Red, NIR).
        out_channels (int): Output spectral bands. Defaults to in_channels.
        num_features (int): Feature map channels in residual body. Default 64.
        num_blocks   (int): Number of stacked residual blocks. Default 4.
        scale_factor (int): Upsampling factor — 2 or 4. Default 2.
        dropout_rate (float): Spatial Dropout probability. Default 0.2.
    """
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: Optional[int] = None,
        num_features: int = 64,
        num_blocks: int = 4,
        scale_factor: int = 2,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        self.scale_factor = scale_factor
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.dropout_rate = dropout_rate

        # Head: shallow feature extraction
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, num_features, 3, 1, 1, bias=True),
            nn.PReLU()
        )
        # Body: deep residual feature learning
        self.body  = nn.Sequential(*[
            ResidualBlock(num_features, dropout_rate) for _ in range(num_blocks)
        ])
        self.body_conv = nn.Conv2d(num_features, num_features, 3, 1, 1, bias=True)
        # Upsampler + Tail
        self.upsample = UpsampleBlock(num_features, scale_factor)
        self.tail     = nn.Conv2d(num_features, out_channels, 3, 1, 1, bias=True)
        # Global bicubic skip for stable residual learning
        self.bicubic  = nn.Upsample(scale_factor=scale_factor, mode="bicubic", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip    = self.bicubic(x)
        feat    = self.head(x)
        res     = self.body(feat)
        res     = self.body_conv(res) + feat
        hr_feat = self.upsample(res)
        return (skip + self.tail(hr_feat)).clamp(0.0, 1.0)

    def enable_mc_dropout(self):
        """Reactivate Dropout layers during eval mode for stochastic MC sampling."""
        for m in self.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout2d)):
                m.train()

    @torch.no_grad()
    def monte_carlo_inference(
        self,
        x: torch.Tensor,
        num_passes: int = 15,
        return_variance: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Bayesian Monte Carlo Dropout Inference.

        Performs T stochastic forward passes with Dropout active, approximating
        a Bayesian posterior over network weights:

            mu(x)     = (1/T) * sum_t [ y_t ]            <- predictive mean (SR output)
            sigma(x)  = sqrt( Var_t [ y_t ] + eps )      <- epistemic uncertainty

        High-uncertainty pixels correspond to complex textures, shadows, cloud boundaries,
        or out-of-distribution regions.

        Args:
            x            : LR input tensor (B, C, H, W)
            num_passes   : Number of stochastic MC passes. Default 15.
            return_variance: If True returns variance; else standard deviation.

        Returns:
            (mean_sr, uncertainty_map) — both tensors of shape (B, C, H_sr, W_sr)
        """
        self.eval()
        self.enable_mc_dropout()

        preds = []
        for _ in range(num_passes):
            preds.append(self.forward(x).unsqueeze(0))

        stacked = torch.cat(preds, dim=0)                          # (T, B, C, H, W)
        mean_sr = torch.mean(stacked, dim=0)                       # (B, C, H, W)
        var_map = torch.var(stacked, dim=0, unbiased=False)        # (B, C, H, W)
        unc_map = var_map if return_variance else torch.sqrt(var_map + 1e-8)

        return mean_sr, unc_map


def load_model(
    checkpoint_path: str,
    scale_factor: int = 2,
    in_channels: int = 4,
    device: Optional[torch.device] = None,
) -> ResidualSR:
    """
    Loads a trained ResidualSR checkpoint from disk.
    Falls back to untrained weights if the file is not found.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ResidualSR(
        in_channels=in_channels,
        out_channels=in_channels,
        num_features=64,
        num_blocks=4,
        scale_factor=scale_factor,
        dropout_rate=0.2,
    )

    import os
    if os.path.isfile(checkpoint_path):
        try:
            ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(checkpoint_path, map_location=device)

        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
            # Remap old checkpoint key naming (res_blocks.*) → new naming (body.*)
            remapped = {}
            for k, v in state.items():
                new_k = k.replace("res_blocks.", "body.")
                remapped[new_k] = v
            model.load_state_dict(remapped)
            print(f"[BhuVistaar] Loaded checkpoint: epoch={ckpt.get('epoch', '?')} "
                  f"val_loss={ckpt.get('val_loss', '?'):.4f}")
        else:
            model.load_state_dict(ckpt)
            print("[BhuVistaar] Loaded state dict.")
    else:
        print(f"[BhuVistaar] Checkpoint not found at '{checkpoint_path}'. Using random weights.")

    return model.to(device).eval()
