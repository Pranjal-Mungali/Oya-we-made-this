"""
Sentinel-2 Super-Resolution Inference with Monte Carlo Dropout Uncertainty Estimation.

Features:
- Loads trained ResidualSR checkpoints from models/ (or initializes on the fly).
- Supports single image (.tif/.npy/synthetic) and batch processing.
- Performs Monte Carlo Dropout inference over T stochastic forward passes.
- Computes:
  1. Predictive Mean: High-resolution reconstructed output.
  2. Epistemic Uncertainty Map: Standard deviation across MC passes (quantifying model confidence and anomalies).
- Computes PSNR and SSIM if a High-Resolution ground truth reference is provided.
- Saves multi-band tensors, RGB visuals, uncertainty heatmaps, and side-by-side comparison panels into outputs/.
"""

import argparse
import glob
import os
from typing import Optional, Tuple, Union, List

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim_fn
from skimage.metrics import peak_signal_noise_ratio as psnr_fn

from src.model import ResidualSR, monte_carlo_inference
from src.dataset import Sentinel2SRDataset, generate_dummy_sentinel2_data

# Optional geospatial reader
try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False


def load_model(
    model_path: str,
    scale_factor: int = 2,
    channels: int = 4,
    device: torch.device = torch.device("cpu")
) -> ResidualSR:
    """
    Loads ResidualSR model weights from checkpoint or instantiates a fresh model.
    """
    model = ResidualSR(
        in_channels=channels,
        out_channels=channels,
        num_features=64,
        num_blocks=4,
        scale_factor=scale_factor,
        dropout_rate=0.2
    )

    if os.path.isfile(model_path):
        print(f"Loading checkpoint from: '{model_path}'")
        try:
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            print(f"  Loaded trained model state (Val Loss: {checkpoint.get('val_loss', 'N/A')}, Epoch: {checkpoint.get('epoch', 'N/A')})")
        elif isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint)
            print("  Loaded state dict successfully.")
    else:
        print(f"[Notice] Model checkpoint '{model_path}' not found. Using initialized model weights.")

    model.to(device)
    model.eval()
    return model


def to_rgb_visual(img_tensor_or_np: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """
    Converts a multi-band image (C, H, W) in range [0, 1] to an RGB (H, W, 3) visualization array.
    """
    if isinstance(img_tensor_or_np, torch.Tensor):
        arr = img_tensor_or_np.detach().cpu().numpy()
    else:
        arr = img_tensor_or_np.copy()

    if arr.ndim == 4:  # (1, C, H, W)
        arr = arr[0]

    c, h, w = arr.shape
    if c >= 3:
        # Channels 0, 1, 2 as RGB visual
        rgb = np.transpose(arr[:3], (1, 2, 0))
    elif c == 1:
        rgb = np.repeat(np.transpose(arr, (1, 2, 0)), 3, axis=2)
    else:
        rgb = np.zeros((h, w, 3), dtype=np.float32)
        rgb[:, :, :c] = np.transpose(arr, (1, 2, 0))

    return np.clip(rgb, 0.0, 1.0)


def save_geotiff_or_numpy(array: np.ndarray, file_path: str):
    """Saves multi-band array (C, H, W) as .tif or .npy."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    c, h, w = array.shape
    if HAS_RASTERIO:
        with rasterio.open(
            file_path,
            "w",
            driver="GTiff",
            height=h,
            width=w,
            count=c,
            dtype=np.float32
        ) as dst:
            dst.write(array.astype(np.float32))
    elif HAS_TIFFFILE:
        tifffile.imwrite(file_path, array.astype(np.float32))
    else:
        np.save(file_path.replace(".tif", ".npy"), array.astype(np.float32))


def run_inference_on_sample(
    model: ResidualSR,
    lr_tensor: torch.Tensor,
    hr_gt: Optional[torch.Tensor] = None,
    num_mc_samples: int = 15,
    sample_name: str = "sample",
    output_dir: str = "outputs",
    device: torch.device = torch.device("cpu")
) -> dict:
    """
    Executes Monte Carlo Dropout inference on a single low-resolution input tensor.
    """
    os.makedirs(output_dir, exist_ok=True)

    if lr_tensor.ndim == 3:
        lr_tensor = lr_tensor.unsqueeze(0)  # (1, C, H, W)

    lr_tensor = lr_tensor.to(device)

    # 1. Run Monte Carlo Dropout Inference
    print(f"Running Monte Carlo Dropout ({num_mc_samples} forward passes)...")
    mean_sr, uncertainty_map = model.monte_carlo_inference(
        lr_tensor,
        num_passes=num_mc_samples,
        return_variance=False  # Returns standard deviation map
    )

    # Convert to numpy for export and visualization
    lr_np = lr_tensor.squeeze(0).detach().cpu().numpy()
    sr_mean_np = mean_sr.squeeze(0).detach().cpu().numpy()
    unc_np = uncertainty_map.squeeze(0).detach().cpu().numpy()

    # Aggregate uncertainty across bands for 2D spatial heatmap
    spatial_uncertainty = np.mean(unc_np, axis=0)  # (H, W)

    # 2. Compute Metrics if HR Ground Truth is available
    metrics = {}
    hr_np = None
    if hr_gt is not None:
        if isinstance(hr_gt, torch.Tensor):
            hr_np = hr_gt.squeeze(0).detach().cpu().numpy() if hr_gt.ndim == 4 else hr_gt.detach().cpu().numpy()
        else:
            hr_np = hr_gt

        # Calculate multi-channel PSNR & SSIM
        psnr_val = psnr_fn(hr_np, sr_mean_np, data_range=1.0)
        # SSIM with channel axis
        ssim_val = ssim_fn(
            np.transpose(hr_np, (1, 2, 0)),
            np.transpose(sr_mean_np, (1, 2, 0)),
            channel_axis=-1,
            data_range=1.0
        )
        metrics["PSNR"] = psnr_val
        metrics["SSIM"] = ssim_val

    # 3. Save Output Artifacts
    # A. Super-resolved multi-band image
    sr_tif_path = os.path.join(output_dir, f"{sample_name}_sr_mean.tif")
    save_geotiff_or_numpy(sr_mean_np, sr_tif_path)

    # B. Uncertainty map multi-band file
    unc_tif_path = os.path.join(output_dir, f"{sample_name}_uncertainty.tif")
    save_geotiff_or_numpy(unc_np, unc_tif_path)

    # C. RGB Super-Resolved Visual (.png)
    sr_rgb = to_rgb_visual(sr_mean_np)
    sr_png_path = os.path.join(output_dir, f"{sample_name}_sr_visual.png")
    plt.imsave(sr_png_path, sr_rgb)

    # D. Uncertainty Heatmap (.png)
    unc_png_path = os.path.join(output_dir, f"{sample_name}_uncertainty_heatmap.png")
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(spatial_uncertainty, cmap="inferno")
    ax.set_title(f"Epistemic Uncertainty Heatmap\n(Std Dev across {num_mc_samples} MC Passes)")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pixel Standard Deviation")
    plt.tight_layout()
    plt.savefig(unc_png_path, dpi=200)
    plt.close(fig)

    # E. Side-by-Side Comparison Panel (.png)
    comparison_png_path = os.path.join(output_dir, f"{sample_name}_comparison.png")
    num_cols = 4 if hr_np is not None else 3
    fig, axes = plt.subplots(1, num_cols, figsize=(4 * num_cols, 4))

    # Panel 1: Low-Resolution Input
    lr_rgb = to_rgb_visual(lr_np)
    axes[0].imshow(lr_rgb)
    axes[0].set_title(f"Low-Resolution Input\n({lr_np.shape[1]}x{lr_np.shape[2]})")
    axes[0].axis("off")

    # Panel 2: Super-Resolved Output (Predictive Mean)
    axes[1].imshow(sr_rgb)
    axes[1].set_title(f"Super-Resolved (Mean)\n({sr_mean_np.shape[1]}x{sr_mean_np.shape[2]})")
    axes[1].axis("off")

    # Panel 3: Uncertainty Heatmap
    im_unc = axes[2].imshow(spatial_uncertainty, cmap="inferno")
    axes[2].set_title(f"MC Uncertainty Map\n(Mean Std: {spatial_uncertainty.mean():.4f})")
    axes[2].axis("off")
    fig.colorbar(im_unc, ax=axes[2], fraction=0.046, pad=0.04)

    # Panel 4: Ground Truth (if provided)
    if hr_np is not None:
        hr_rgb = to_rgb_visual(hr_np)
        axes[3].imshow(hr_rgb)
        axes[3].set_title(f"High-Resolution GT\nPSNR: {metrics['PSNR']:.2f}dB | SSIM: {metrics['SSIM']:.3f}")
        axes[3].axis("off")

    plt.tight_layout()
    plt.savefig(comparison_png_path, dpi=200)
    plt.close(fig)

    results = {
        "sample_name": sample_name,
        "lr_shape": list(lr_np.shape),
        "sr_shape": list(sr_mean_np.shape),
        "uncertainty_shape": list(unc_np.shape),
        "uncertainty_min": float(spatial_uncertainty.min()),
        "uncertainty_max": float(spatial_uncertainty.max()),
        "uncertainty_mean": float(spatial_uncertainty.mean()),
        "metrics": metrics,
        "saved_files": {
            "sr_image": sr_tif_path,
            "sr_visual": sr_png_path,
            "uncertainty_tif": unc_tif_path,
            "uncertainty_heatmap": unc_png_path,
            "comparison": comparison_png_path
        }
    }
    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Sentinel-2 Super-Resolution Inference with Monte Carlo Dropout."
    )
    parser.add_argument("--model-path", type=str, default="models/residual_sr_x2_best.pt",
                        help="Path to trained PyTorch checkpoint. Default: 'models/residual_sr_x2_best.pt'.")
    parser.add_argument("--input", type=str, default=None,
                        help="Path to input .tif image or folder. If None or 'synthetic', runs on synthetic sample.")
    parser.add_argument("--scale", type=int, default=2, choices=[2, 4],
                        help="Upscaling factor (2 or 4). Default: 2.")
    parser.add_argument("--channels", type=int, default=4,
                        help="Number of spectral bands (RGB+NIR). Default: 4.")
    parser.add_argument("--num-mc-samples", type=int, default=15,
                        help="Number of stochastic forward passes for Monte Carlo Dropout. Default: 15.")
    parser.add_argument("--output-dir", type=str, default="outputs",
                        help="Directory to save generated outputs. Default: 'outputs'.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"],
                        help="Computation device ('auto', 'cuda', or 'cpu'). Default: 'auto'.")
    return parser.parse_args()


def main():
    args = parse_args()

    # Hardware device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print("=" * 65)
    print(" Sentinel-2 Super-Resolution Monte Carlo Inference")
    print("=" * 65)
    print(f"Device:             {device}")
    print(f"Model Checkpoint:   {args.model_path}")
    print(f"Scale Factor:       {args.scale}x")
    print(f"MC Dropout Passes:  {args.num_mc_samples}")
    print(f"Output Directory:   {args.output_dir}")
    print("=" * 65)

    # 1. Load Model
    model = load_model(
        model_path=args.model_path,
        scale_factor=args.scale,
        channels=args.channels,
        device=device
    )

    # 2. Prepare Sample(s)
    if args.input is None or args.input.lower() == "synthetic":
        print("\nNo custom input provided. Generating a paired synthetic Sentinel-2 sample for test...")
        # Create dataset sample with GT for metric computation
        dataset = Sentinel2SRDataset(
            scale_factor=args.scale,
            hr_patch_size=128,
            channels=args.channels,
            num_samples=1,
            is_train=False
        )
        lr_tensor, hr_gt = dataset[0]
        sample_name = f"synthetic_sentinel2_x{args.scale}"
        
        # Run inference
        results = run_inference_on_sample(
            model=model,
            lr_tensor=lr_tensor,
            hr_gt=hr_gt,
            num_mc_samples=args.num_mc_samples,
            sample_name=sample_name,
            output_dir=args.output_dir,
            device=device
        )

        print("\nInference Results Summary:")
        print(f"  - LR Input Shape:       {results['lr_shape']}")
        print(f"  - Super-Resolved Shape: {results['sr_shape']}")
        print(f"  - Uncertainty Map Shape:{results['uncertainty_shape']}")
        print(f"  - Uncertainty Range:    [{results['uncertainty_min']:.6f}, {results['uncertainty_max']:.6f}] (Mean: {results['uncertainty_mean']:.6f})")
        if results['metrics']:
            print(f"  - PSNR:                 {results['metrics']['PSNR']:.2f} dB")
            print(f"  - SSIM:                 {results['metrics']['SSIM']:.4f}")
        print("\nSaved Output Artifacts in 'outputs/':")
        for key, path in results["saved_files"].items():
            print(f"  - {key:20s}: {path}")

    else:
        # Check if single file or folder
        if os.path.isfile(args.input):
            input_files = [args.input]
        elif os.path.isdir(args.input):
            input_files = sorted(
                glob.glob(os.path.join(args.input, "*.tif")) +
                glob.glob(os.path.join(args.input, "*.tiff"))
            )
        else:
            raise FileNotFoundError(f"Input path '{args.input}' not found.")

        print(f"\nProcessing {len(input_files)} input image(s)...")
        dataset = Sentinel2SRDataset(
            image_paths=input_files,
            scale_factor=args.scale,
            hr_patch_size=128,
            channels=args.channels,
            num_samples=len(input_files),
            is_train=False
        )

        for idx in range(len(dataset)):
            lr_tensor, hr_gt = dataset[idx]
            base_name = os.path.splitext(os.path.basename(input_files[idx]))[0]
            print(f"\nProcessing [{idx+1}/{len(dataset)}]: {base_name}...")
            res = run_inference_on_sample(
                model=model,
                lr_tensor=lr_tensor,
                hr_gt=hr_gt,
                num_mc_samples=args.num_mc_samples,
                sample_name=base_name,
                output_dir=args.output_dir,
                device=device
            )
            print(f"  SR Shape: {res['sr_shape']}, Mean Uncertainty: {res['uncertainty_mean']:.5f}")

    print("\n" + "=" * 65)
    print("Inference completed successfully!")
    print("=" * 65)


if __name__ == "__main__":
    main()
