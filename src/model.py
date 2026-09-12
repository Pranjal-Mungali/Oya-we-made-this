"""
Sentinel-2 Super Resolution Model with Monte Carlo Dropout for Uncertainty Estimation.

Architecture:
- Deep Residual CNN backbone (ResidualSR)
- Sub-pixel convolution via PixelShuffle for efficient 2x and 4x upsampling
- Multi-band input support (e.g. 4 bands: RGB + NIR or all 12/13 Sentinel-2 bands)
- Dropout layers enabling Monte Carlo Dropout (Gal & Ghahramani, 2016)
  to quantify epistemic (model) uncertainty and flag reconstruction outliers.
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """
    Residual Block with integrated Dropout for Monte Carlo uncertainty sampling.
    
    Structure:
        Conv2d -> PReLU -> Dropout2d -> Conv2d -> Add Residual
    """

    def __init__(self, num_features: int = 64, dropout_rate: float = 0.2):
        super().__init__()
        self.conv1 = nn.Conv2d(num_features, num_features, kernel_size=3, stride=1, padding=1, bias=True)
        self.act = nn.PReLU()
        # Dropout2d (Spatial Dropout) drops entire feature channels,
        # providing effective regularization and meaningful epistemic uncertainty in CNNs.
        self.dropout = nn.Dropout2d(p=dropout_rate) if dropout_rate > 0.0 else nn.Identity()
        self.conv2 = nn.Conv2d(num_features, num_features, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.act(out)
        out = self.dropout(out)
        out = self.conv2(out)
        return out + residual


class UpsampleBlock(nn.Module):
    """
    PixelShuffle Upsampling Block.
    
    Uses sub-pixel convolution (PixelShuffle) to efficiently reconstruct
    high-resolution feature maps without checkerboard artifacts.
    """

    def __init__(self, num_features: int = 64, scale_factor: int = 2):
        super().__init__()
        if scale_factor not in (2, 4):
            raise ValueError(f"scale_factor must be 2 or 4, got {scale_factor}")

        layers = []
        if scale_factor == 2:
            layers.extend([
                nn.Conv2d(num_features, num_features * 4, kernel_size=3, stride=1, padding=1, bias=True),
                nn.PixelShuffle(upscale_factor=2),
                nn.PReLU()
            ])
        elif scale_factor == 4:
            # 4x upsampling using two cascaded 2x PixelShuffle stages for better stability
            for _ in range(2):
                layers.extend([
                    nn.Conv2d(num_features, num_features * 4, kernel_size=3, stride=1, padding=1, bias=True),
                    nn.PixelShuffle(upscale_factor=2),
                    nn.PReLU()
                ])

        self.upsample = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.upsample(x)


class ResidualSR(nn.Module):
    """
    Residual Super-Resolution Network for Sentinel-2 Multispectral Imagery.

    Parameters:
        in_channels (int): Number of input channels (e.g., 4 for RGB+NIR, or 12/13 for full Sentinel-2).
        out_channels (int): Number of target output channels (default: matches in_channels).
        num_features (int): Number of intermediate feature channels (default: 64).
        num_blocks (int): Number of stacked residual blocks (default: 4 for fast prototyping).
        scale_factor (int): Upsampling factor, either 2 or 4 (default: 2).
        dropout_rate (float): Dropout probability for Monte Carlo inference (default: 0.2).
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: Optional[int] = None,
        num_features: int = 64,
        num_blocks: int = 4,
        scale_factor: int = 2,
        dropout_rate: float = 0.2
    ):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = scale_factor
        self.dropout_rate = dropout_rate

        # 1. Shallow feature extraction (Head)
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, num_features, kernel_size=3, stride=1, padding=1, bias=True),
            nn.PReLU()
        )

        # 2. Deep feature extraction with residual backbone (Body)
        self.res_blocks = nn.Sequential(*[
            ResidualBlock(num_features=num_features, dropout_rate=dropout_rate)
            for _ in range(num_blocks)
        ])
        self.body_conv = nn.Conv2d(num_features, num_features, kernel_size=3, stride=1, padding=1, bias=True)

        # 3. Upsampling reconstruction (PixelShuffle)
        self.upsample = UpsampleBlock(num_features=num_features, scale_factor=scale_factor)

        # 4. Final output layer (Tail)
        self.tail = nn.Conv2d(num_features, out_channels, kernel_size=3, stride=1, padding=1, bias=True)

        # 5. Global baseline bicubic skip connection
        self.bicubic_upsample = nn.Upsample(scale_factor=scale_factor, mode="bicubic", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Low-resolution input tensor of shape (B, C, H, W).
            
        Returns:
            torch.Tensor: High-resolution output tensor of shape (B, C, H * scale, W * scale).
        """
        # Global residual learning: model learns residual details on top of bicubic upsampling
        bicubic = self.bicubic_upsample(x)

        head_feat = self.head(x)
        body_feat = self.res_blocks(head_feat)
        body_feat = self.body_conv(body_feat) + head_feat  # Long skip connection
        hr_feat = self.upsample(body_feat)
        sr_residual = self.tail(hr_feat)

        return bicubic + sr_residual

    def enable_mc_dropout(self):
        """
        Force all Dropout layers to remain active during evaluation/inference.
        
        This enables Monte Carlo Dropout sampling without affecting other layers
        like BatchNorm (if present).
        """
        for module in self.modules():
            if isinstance(module, (nn.Dropout, nn.Dropout2d)):
                module.train()

    @torch.no_grad()
    def monte_carlo_inference(
        self,
        x: torch.Tensor,
        num_passes: int = 15,
        return_variance: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run Monte Carlo Dropout inference to predict super-resolved imagery and uncertainty.

        === How Monte Carlo Dropout Works ===
        Standard neural networks output deterministic point estimates with no measure of
        confidence. In Monte Carlo (MC) Dropout (Gal & Ghahramani, 2016), dropout is kept
        active at inference time.
        
        By performing T independent stochastic forward passes:
            {y_1, y_2, ..., y_T} = {f(x; W_1), f(x; W_2), ..., f(x; W_T)}
            
        We approximate a Bayesian posterior distribution over network weights:
        1. Predictive Mean (Super-Resolved Output):
               mu(x) = (1 / T) * sum(y_t)
        2. Epistemic Uncertainty Map (Pixel-wise Variance or Std Dev):
               sigma^2(x) = (1 / T) * sum((y_t - mu(x))^2)
               sigma(x)   = sqrt(sigma^2(x))
               
        High uncertainty regions typically correspond to:
        - Fine edges / high-frequency textures that are hard to resolve
        - Sensor anomalies, cloud borders, or out-of-distribution outliers
        
        Args:
            x (torch.Tensor): Low-resolution input tensor of shape (B, C, H, W).
            num_passes (int): Number of stochastic forward passes (default: 15).
            return_variance (bool): If True, returns variance map; if False, returns standard deviation map.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - mean_sr: Predictive mean super-resolution tensor (B, C, H*scale, W*scale)
                - uncertainty_map: Pixel-wise uncertainty tensor (B, C, H*scale, W*scale) or aggregated.
        """
        # Put entire network in eval mode first
        self.eval()
        # Reactivate only Dropout layers for stochastic sampling
        self.enable_mc_dropout()

        preds = []
        for _ in range(num_passes):
            pred = self.forward(x)
            preds.append(pred.unsqueeze(0))

        # Stack passes: Shape -> (num_passes, B, C, H_sr, W_sr)
        stacked_preds = torch.cat(preds, dim=0)

        # 1. Predictive Mean: Expected high-resolution image
        mean_sr = torch.mean(stacked_preds, dim=0)

        # 2. Epistemic Uncertainty: Variance across Monte Carlo passes
        variance_map = torch.var(stacked_preds, dim=0, unbiased=False)

        if return_variance:
            uncertainty_map = variance_map
        else:
            uncertainty_map = torch.sqrt(variance_map + 1e-8)

        return mean_sr, uncertainty_map


def monte_carlo_inference(
    model: ResidualSR,
    x: torch.Tensor,
    num_passes: int = 15,
    return_variance: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convenience function to run Monte Carlo Dropout inference on a given model.
    """
    return model.monte_carlo_inference(x=x, num_passes=num_passes, return_variance=return_variance)


if __name__ == "__main__":
    # Quick sanity test
    print("Testing ResidualSR architecture...")
    batch_size = 2
    in_channels = 4  # RGB + NIR
    lr_height, lr_width = 32, 32

    # 1. Test 2x Super-Resolution
    model_2x = ResidualSR(in_channels=in_channels, scale_factor=2, num_blocks=4, dropout_rate=0.2)
    sample_input = torch.randn(batch_size, in_channels, lr_height, lr_width)
    
    output_2x = model_2x(sample_input)
    print(f"2x Input shape:  {sample_input.shape}")
    print(f"2x Output shape: {output_2x.shape} (Expected: ({batch_size}, {in_channels}, {lr_height * 2}, {lr_width * 2}))")
    assert output_2x.shape == (batch_size, in_channels, lr_height * 2, lr_width * 2)

    # 2. Test 4x Super-Resolution
    model_4x = ResidualSR(in_channels=in_channels, scale_factor=4, num_blocks=4, dropout_rate=0.2)
    output_4x = model_4x(sample_input)
    print(f"4x Output shape: {output_4x.shape} (Expected: ({batch_size}, {in_channels}, {lr_height * 4}, {lr_width * 4}))")
    assert output_4x.shape == (batch_size, in_channels, lr_height * 4, lr_width * 4)

    # 3. Test Monte Carlo Inference
    print("\nRunning Monte Carlo Dropout inference (15 passes)...")
    mean_img, uncertainty_map = model_2x.monte_carlo_inference(sample_input, num_passes=15)
    print(f"Mean SR shape:        {mean_img.shape}")
    print(f"Uncertainty map shape:{uncertainty_map.shape}")
    print(f"Uncertainty min/max:  {uncertainty_map.min().item():.6f} / {uncertainty_map.max().item():.6f}")

    print("\nAll sanity checks passed successfully!")
