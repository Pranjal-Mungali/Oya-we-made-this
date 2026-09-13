"""
Sentinel-2 Super Resolution Dataset Pipeline with Synthetic LR-HR Pair Generation.

Features:
- Reads 4-band Sentinel-2 imagery (RGB + NIR) or synthetic multi-band tensors.
- Degrades HR ground truth with Gaussian blur + downsampling to produce paired LR inputs.
- Random cropping into configurable patch sizes (e.g. 128x128 HR -> 64x64 LR for 2x scale).
- Data augmentations (horizontal flip, vertical flip, random 90-degree rotation).
- Works with GeoTIFF (.tif/.tiff) files or in-memory synthetic arrays for rapid testing.
- Includes utility functions to generate dummy multi-band satellite data.
"""

from typing import List, Optional, Tuple, Union
import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

# Optional geospatial reader dependencies with fallback support
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


def _create_gaussian_kernel2d(kernel_size: int = 5, sigma: float = 1.0) -> torch.Tensor:
    """Creates a 2D Gaussian blur kernel normalized to sum to 1."""
    coords = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel_1d = g / g.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    return kernel_2d


class Sentinel2SRDataset(Dataset):
    """
    PyTorch Dataset for Sentinel-2 Super-Resolution.
    
    Generates Low-Resolution (LR) and High-Resolution (HR) paired patches.
    
    Args:
        image_paths (List[str], optional): List of file paths to .tif satellite images.
        synthetic_images (List[np.ndarray], optional): List of in-memory (C, H, W) numpy images.
        scale_factor (int): Downsampling / super-resolution factor (2 or 4). Default: 2.
        hr_patch_size (int): Size of HR square patches (e.g., 128 or 64). Default: 128.
        channels (int): Expected number of spectral bands (e.g., 4 for RGB+NIR). Default: 4.
        num_samples (int): Virtual length of dataset (number of random crops per epoch). Default: 100.
        is_train (bool): If True, applies random geometric augmentations. Default: True.
        blur_sigma (float): Sigma for Gaussian blur degradation modeling. Default: 1.0.
        normalize (bool): If True, normalizes 16-bit DN values [0, 10000] to reflectance [0.0, 1.0]. Default: True.
    """

    def __init__(
        self,
        image_paths: Optional[List[str]] = None,
        synthetic_images: Optional[List[np.ndarray]] = None,
        scale_factor: int = 2,
        hr_patch_size: int = 128,
        channels: int = 4,
        num_samples: int = 100,
        is_train: bool = True,
        blur_sigma: float = 1.0,
        normalize: bool = True
    ):
        super().__init__()
        if scale_factor not in (2, 4):
            raise ValueError(f"scale_factor must be 2 or 4, got {scale_factor}")
        if hr_patch_size % scale_factor != 0:
            raise ValueError(f"hr_patch_size ({hr_patch_size}) must be divisible by scale_factor ({scale_factor})")

        self.scale_factor = scale_factor
        self.hr_patch_size = hr_patch_size
        self.lr_patch_size = hr_patch_size // scale_factor
        self.channels = channels
        self.num_samples = num_samples
        self.is_train = is_train
        self.blur_sigma = blur_sigma
        self.normalize = normalize

        # Prepare image sources
        self.image_paths = image_paths or []
        self.synthetic_images = synthetic_images or []

        # If neither provided, automatically initialize with synthetic imagery
        if not self.image_paths and not self.synthetic_images:
            self.synthetic_images = generate_dummy_sentinel2_data(
                num_images=4,
                height=max(256, hr_patch_size * 2),
                width=max(256, hr_patch_size * 2),
                channels=channels
            )

        # Pre-compute Gaussian blur kernel for LR degradation
        kernel_2d = _create_gaussian_kernel2d(kernel_size=5, sigma=self.blur_sigma)
        # Reshape to (C, 1, K, K) for depthwise 2D convolution
        self.blur_kernel = kernel_2d.repeat(self.channels, 1, 1, 1)

    def __len__(self) -> int:
        return self.num_samples

    def _load_image(self, index: int) -> np.ndarray:
        """Loads a multi-band image from disk or in-memory list. Returns shape (C, H, W)."""
        if self.image_paths:
            path = self.image_paths[index % len(self.image_paths)]
            return self._read_tif(path)
        else:
            return self.synthetic_images[index % len(self.synthetic_images)]

    def _read_tif(self, path: str) -> np.ndarray:
        """Reads a GeoTIFF image file into a float32 numpy array with shape (C, H, W)."""
        if HAS_RASTERIO:
            with rasterio.open(path) as src:
                img = src.read()  # (C, H, W)
                img = img.astype(np.float32)
        elif HAS_TIFFFILE:
            img = tifffile.imread(path).astype(np.float32)
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            elif img.ndim == 3 and img.shape[2] <= 13:  # (H, W, C) -> (C, H, W)
                img = np.transpose(img, (2, 0, 1))
        else:
            # Fallback loader for .npy / raw binary or simple images
            try:
                from PIL import Image
                pil_img = Image.open(path)
                img = np.array(pil_img, dtype=np.float32)
                if img.ndim == 2:
                    img = img[np.newaxis, ...]
                elif img.ndim == 3:
                    img = np.transpose(img, (2, 0, 1))
            except Exception as e:
                raise RuntimeError(
                    f"Failed to read image at '{path}'. Install rasterio or tifffile: {e}"
                )

        # Ensure channel count matches
        if img.shape[0] < self.channels:
            # Replicate channels if fewer
            repeat_count = int(np.ceil(self.channels / img.shape[0]))
            img = np.repeat(img, repeat_count, axis=0)[:self.channels]
        elif img.shape[0] > self.channels:
            img = img[:self.channels]

        # Normalize Sentinel-2 digital numbers (DN) to [0.0, 1.0] reflectance
        if self.normalize:
            if img.max() > 1.0:
                # Sentinel-2 L1C/L2A typical reflectance scale factor is 10,000
                img = img / 10000.0
            img = np.clip(img, 0.0, 1.0)

        return img

    def _apply_augmentations(self, hr_patch: torch.Tensor) -> torch.Tensor:
        """Applies random flips and 90-degree rotations to the HR patch."""
        # Horizontal flip
        if random.random() > 0.5:
            hr_patch = torch.flip(hr_patch, dims=[-1])
        # Vertical flip
        if random.random() > 0.5:
            hr_patch = torch.flip(hr_patch, dims=[-2])
        # Random 90-degree rotation (k in [0, 1, 2, 3])
        k = random.randint(0, 3)
        if k > 0:
            hr_patch = torch.rot90(hr_patch, k, dims=[-2, -1])
        return hr_patch

    def _degrade_hr_to_lr(self, hr_patch: torch.Tensor) -> torch.Tensor:
        """
        Degrades an HR patch (C, H, W) to an LR patch (C, H/scale, W/scale)
        using Gaussian blur followed by bicubic/area downsampling.
        """
        # Add batch dimension: (1, C, H, W)
        x = hr_patch.unsqueeze(0)

        # 1. Apply depthwise Gaussian blur
        padding = (self.blur_kernel.shape[-1] - 1) // 2
        kernel = self.blur_kernel.to(x.device, dtype=x.dtype)
        blurred = F.conv2d(x, kernel, padding=padding, groups=self.channels)

        # 2. Downsample spatially by scale_factor
        lr = F.interpolate(
            blurred,
            size=(self.lr_patch_size, self.lr_patch_size),
            mode="bicubic",
            align_corners=False
        )

        return lr.squeeze(0).clamp(0.0, 1.0)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (lr_patch, hr_patch)
                - lr_patch: (C, lr_patch_size, lr_patch_size)
                - hr_patch: (C, hr_patch_size, hr_patch_size)
        """
        img_np = self._load_image(idx)  # (C, H, W)
        c, h, w = img_np.shape

        # Ensure image is large enough for patch size
        if h < self.hr_patch_size or w < self.hr_patch_size:
            pad_h = max(0, self.hr_patch_size - h)
            pad_w = max(0, self.hr_patch_size - w)
            img_np = np.pad(img_np, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")
            _, h, w = img_np.shape

        # Random Crop
        top = random.randint(0, h - self.hr_patch_size)
        left = random.randint(0, w - self.hr_patch_size)
        hr_crop_np = img_np[:, top:top + self.hr_patch_size, left:left + self.hr_patch_size]

        # Convert to Tensor (C, H, W)
        hr_patch = torch.from_numpy(hr_crop_np.copy()).float()

        # Apply Augmentations if training
        if self.is_train:
            hr_patch = self._apply_augmentations(hr_patch)

        # Create LR counterpart via Blur + Downsampling
        lr_patch = self._degrade_hr_to_lr(hr_patch)

        return lr_patch, hr_patch


def generate_dummy_sentinel2_data(
    num_images: int = 4,
    height: int = 256,
    width: int = 256,
    channels: int = 4,
    save_dir: Optional[str] = None
) -> Union[List[np.ndarray], List[str]]:
    """
    Generates synthetic multispectral Sentinel-2-like images for unit testing and prototyping.

    Each synthetic image features multi-frequency patterns simulating terrain,
    vegetation (high NIR response in band 4), water, and edge structures.

    Args:
        num_images (int): Number of synthetic images to generate.
        height (int): Image height in pixels. Default: 256.
        width (int): Image width in pixels. Default: 256.
        channels (int): Number of bands (e.g., 4: Blue, Green, Red, NIR). Default: 4.
        save_dir (str, optional): If provided, writes GeoTIFF files (.tif) to this directory.

    Returns:
        Union[List[np.ndarray], List[str]]: List of numpy arrays or file paths to saved .tif files.
    """
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    generated_arrays = []
    saved_paths = []

    for i in range(num_images):
        np.random.seed(42 + i)

        # Base gradient / terrain map
        y, x = np.ogrid[:height, :width]
        base_grid = (np.sin(x / 16.0) + np.cos(y / 16.0)) * 0.2 + 0.5

        bands = []
        for c in range(channels):
            # Band-specific reflectance variations
            noise = np.random.normal(0, 0.05, (height, width))
            channel_shift = 0.1 * (c + 1)
            band = base_grid * (1.0 + 0.2 * c) + channel_shift + noise

            # Simulate Sentinel-2 NIR band (higher reflectance in vegetation scenes)
            if c == 3:  # NIR band
                band = band * 1.5 + 0.1

            band = np.clip(band, 0.0, 1.0).astype(np.float32)
            bands.append(band)

        img_array = np.stack(bands, axis=0)  # Shape: (channels, height, width)

        if save_dir:
            file_path = os.path.join(save_dir, f"synthetic_sentinel2_{i+1}.tif")
            if HAS_RASTERIO:
                with rasterio.open(
                    file_path,
                    "w",
                    driver="GTiff",
                    height=height,
                    width=width,
                    count=channels,
                    dtype=np.float32
                ) as dst:
                    dst.write(img_array)
            elif HAS_TIFFFILE:
                # Save as (H, W, C) or (C, H, W)
                tifffile.imwrite(file_path, img_array)
            else:
                # Fallback: save as numpy binary if tiff writer not found
                np.save(file_path.replace(".tif", ".npy"), img_array)
                file_path = file_path.replace(".tif", ".npy")

            saved_paths.append(file_path)
        else:
            generated_arrays.append(img_array)

    return saved_paths if save_dir else generated_arrays
