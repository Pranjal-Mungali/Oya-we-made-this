"""
Training Script for Sentinel-2 Super-Resolution (ResidualSR).

Supports:
- Multi-band super-resolution (e.g., 4 bands: RGB + NIR)
- 2x and 4x upsampling factors
- L1 reconstruction loss + PSNR metric tracking
- Validation evaluation and automatic best checkpoint saving in models/
- Automatic CUDA/GPU acceleration with CPU fallback
- In-memory synthetic prototyping or real GeoTIFF (.tif) datasets
"""

import argparse
import glob
import os
import time
from typing import Tuple, List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.model import ResidualSR
from src.dataset import Sentinel2SRDataset


def calculate_psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    """
    Computes Peak Signal-to-Noise Ratio (PSNR) in decibels (dB).

    Args:
        pred (torch.Tensor): Reconstructed high-resolution tensor [B, C, H, W] in [0, max_val].
        target (torch.Tensor): Ground-truth high-resolution tensor [B, C, H, W] in [0, max_val].
        max_val (float): Maximum possible pixel value (1.0 for normalized reflectance).

    Returns:
        float: PSNR value in dB.
    """
    with torch.no_grad():
        mse = F.mse_loss(pred, target).item()
        if mse <= 1e-10:
            return 100.0  # Cap perfect reconstruction
        return 10.0 * np.log10((max_val ** 2) / mse)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> Tuple[float, float]:
    """
    Runs one training epoch.

    Returns:
        Tuple[float, float]: (average_loss, average_psnr)
    """
    model.train()
    running_loss = 0.0
    running_psnr = 0.0
    total_batches = len(dataloader)

    for lr_imgs, hr_imgs in dataloader:
        lr_imgs = lr_imgs.to(device)
        hr_imgs = hr_imgs.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        sr_imgs = model(lr_imgs)

        # Compute L1 Loss
        loss = criterion(sr_imgs, hr_imgs)

        # Backward pass & optimization step
        loss.backward()
        optimizer.step()

        # Track metrics
        running_loss += loss.item()
        running_psnr += calculate_psnr(sr_imgs, hr_imgs)

    avg_loss = running_loss / total_batches
    avg_psnr = running_psnr / total_batches
    return avg_loss, avg_psnr


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """
    Evaluates the model on validation data.

    Returns:
        Tuple[float, float]: (val_loss, val_psnr)
    """
    model.eval()
    val_loss = 0.0
    val_psnr = 0.0
    total_batches = len(dataloader)

    with torch.no_grad():
        for lr_imgs, hr_imgs in dataloader:
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)

            sr_imgs = model(lr_imgs)
            loss = criterion(sr_imgs, hr_imgs)

            val_loss += loss.item()
            val_psnr += calculate_psnr(sr_imgs, hr_imgs)

    avg_loss = val_loss / total_batches
    avg_psnr = val_psnr / total_batches
    return avg_loss, avg_psnr


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train ResidualSR model on Sentinel-2 satellite imagery."
    )

    # Dataset & Scaling Parameters
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Path to folder containing .tif satellite files. Defaults to synthetic data if None.")
    parser.add_argument("--scale", type=int, default=2, choices=[2, 4],
                        help="Super-resolution upscale factor (2 or 4). Default: 2.")
    parser.add_argument("--channels", type=int, default=4,
                        help="Number of spectral bands (e.g. 4 for RGB+NIR). Default: 4.")
    parser.add_argument("--patch-size", type=int, default=128,
                        help="High-resolution patch size for training crops. Default: 128.")
    parser.add_argument("--num-train-samples", type=int, default=100,
                        help="Number of training patch samples per epoch. Default: 100.")
    parser.add_argument("--num-val-samples", type=int, default=20,
                        help="Number of validation patch samples per epoch. Default: 20.")

    # Model Architecture Parameters
    parser.add_argument("--num-features", type=int, default=64,
                        help="Number of feature channels in residual blocks. Default: 64.")
    parser.add_argument("--num-blocks", type=int, default=4,
                        help="Number of residual blocks. Default: 4.")
    parser.add_argument("--dropout-rate", type=float, default=0.2,
                        help="Dropout rate for Monte Carlo inference. Default: 0.2.")

    # Optimization & Training Parameters
    parser.add_argument("--epochs", type=int, default=10,
                        help="Total number of training epochs. Default: 10.")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size per iteration. Default: 8.")
    parser.add_argument("--lr", "--learning-rate", dest="lr", type=float, default=1e-4,
                        help="Adam optimizer learning rate. Default: 0.0001.")
    parser.add_argument("--save-dir", type=str, default="models",
                        help="Directory where model checkpoints will be saved. Default: 'models'.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"],
                        help="Device to train on ('cuda', 'cpu', or 'auto'). Default: 'auto'.")

    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Hardware device selection
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print("=" * 65)
    print(" Sentinel-2 Super-Resolution Training Pipeline (ResidualSR)")
    print("=" * 65)
    print(f"Device:               {device}")
    print(f"Scale Factor:         {args.scale}x")
    print(f"Spectral Bands:       {args.channels} (e.g. RGB + NIR)")
    print(f"HR Patch Size:        {args.patch_size}x{args.patch_size} -> LR: {args.patch_size // args.scale}x{args.patch_size // args.scale}")
    print(f"Batch Size:           {args.batch_size}")
    print(f"Learning Rate:        {args.lr}")
    print(f"Epochs:               {args.epochs}")
    print(f"Checkpoint Save Dir:  {args.save_dir}")
    print("=" * 65)

    # 2. Prepare Data Sources
    image_paths: Optional[List[str]] = None
    if args.data_dir and os.path.isdir(args.data_dir):
        image_paths = sorted(
            glob.glob(os.path.join(args.data_dir, "*.tif")) +
            glob.glob(os.path.join(args.data_dir, "*.tiff"))
        )
        print(f"Found {len(image_paths)} GeoTIFF files in '{args.data_dir}'.")
        if not image_paths:
            print("No .tif files found in specified directory; switching to synthetic imagery mode.")
            image_paths = None
    else:
        print("No data directory provided; using synthetic Sentinel-2 multi-band data generator.")

    # 3. Create Datasets and DataLoaders
    train_dataset = Sentinel2SRDataset(
        image_paths=image_paths,
        scale_factor=args.scale,
        hr_patch_size=args.patch_size,
        channels=args.channels,
        num_samples=args.num_train_samples,
        is_train=True
    )
    val_dataset = Sentinel2SRDataset(
        image_paths=image_paths,
        scale_factor=args.scale,
        hr_patch_size=args.patch_size,
        channels=args.channels,
        num_samples=args.num_val_samples,
        is_train=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0
    )

    # 4. Initialize ResidualSR Model
    model = ResidualSR(
        in_channels=args.channels,
        out_channels=args.channels,
        num_features=args.num_features,
        num_blocks=args.num_blocks,
        scale_factor=args.scale,
        dropout_rate=args.dropout_rate
    ).to(device)

    # 5. Loss Function and Optimizer
    criterion = nn.L1Loss()  # L1 loss provides sharp, robust reconstruction gradients
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))

    # 6. Checkpoint Directory Setup
    os.makedirs(args.save_dir, exist_ok=True)
    best_val_loss = float("inf")
    best_model_path = os.path.join(args.save_dir, f"residual_sr_x{args.scale}_best.pt")
    latest_model_path = os.path.join(args.save_dir, f"residual_sr_x{args.scale}_latest.pt")

    # 7. Training Loop
    print("\nStarting training loop...")
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # Train one epoch
        train_loss, train_psnr = train_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device
        )

        # Validation evaluation
        val_loss, val_psnr = validate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device
        )

        epoch_duration = time.time() - epoch_start

        # Check if this is the best model so far
        is_best = val_loss < best_val_loss
        checkpoint_status = ""

        if is_best:
            best_val_loss = val_loss
            # Save checkpoint state dict with metadata
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scale_factor": args.scale,
                "channels": args.channels,
                "num_features": args.num_features,
                "num_blocks": args.num_blocks,
                "dropout_rate": args.dropout_rate,
                "val_loss": val_loss,
                "val_psnr": val_psnr,
            }, best_model_path)
            checkpoint_status = " [Saved Best Model]"

        # Always save latest
        torch.save(model.state_dict(), latest_model_path)

        # Print progress summary
        print(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] "
            f"Time: {epoch_duration:.2f}s | "
            f"Train Loss: {train_loss:.4f} | Train PSNR: {train_psnr:.2f} dB | "
            f"Val Loss: {val_loss:.4f} | Val PSNR: {val_psnr:.2f} dB"
            f"{checkpoint_status}"
        )

    total_time = time.time() - start_time
    print("-" * 65)
    print(f"Training completed in {total_time:.2f}s.")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"Best model checkpoint saved to: '{best_model_path}'")
    print("=" * 65)


if __name__ == "__main__":
    main()
