"""
utils.py — BhuVistaar Preprocessing, Postprocessing & Visualization Utilities.

Handles:
- GeoTIFF (multi-band) and standard RGB image loading
- Multiple bit-depth normalization (uint8, uint16, float32)
- Aspect-ratio-preserving resize + center crop to model patch size
- Correct tensor formatting for the model
- RGB visualization from multi-band tensors
- Uncertainty heatmap rendering
- Side-by-side matplotlib comparison panel

Team: The Outliers | Smart India Hackathon 2026
"""

from typing import Tuple, Optional, Union
import io
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
import torch
import torch.nn.functional as F

# Optional Sentinel-2 / GeoTIFF readers
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


# ─── Constants ────────────────────────────────────────────────────────────────

# Sentinel-2 L1C/L2A DN scale factor (TOAR reflectance)
SENTINEL2_SCALE = 10000.0

# Accepted .tif reader fallback order
_READER_ORDER = ["rasterio", "tifffile"]


# ─── Loading ──────────────────────────────────────────────────────────────────

def _read_geotiff(file_path: str) -> np.ndarray:
    """
    Reads a GeoTIFF file into a float32 (C, H, W) array.
    Tries rasterio → tifffile → PIL in order.
    """
    if HAS_RASTERIO:
        with rasterio.open(file_path) as src:
            arr = src.read().astype(np.float32)        # (C, H, W)
        return arr
    if HAS_TIFFFILE:
        arr = tifffile.imread(file_path).astype(np.float32)
        if arr.ndim == 2:
            arr = arr[np.newaxis]
        elif arr.ndim == 3 and arr.shape[-1] <= 13:    # (H, W, C) layout
            arr = np.transpose(arr, (2, 0, 1))
        return arr
    # Fallback: PIL (reads 8-bit or 16-bit single-band / RGB)
    img = Image.open(file_path)
    arr = np.array(img, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[np.newaxis]
    elif arr.ndim == 3:
        arr = np.transpose(arr, (2, 0, 1))
    return arr


def _read_standard_image(file_path_or_bytes) -> np.ndarray:
    """
    Reads a standard RGB image (PNG / JPEG) into a float32 (3, H, W) array in [0, 1].
    """
    if isinstance(file_path_or_bytes, (str, bytes)):
        img = Image.open(file_path_or_bytes).convert("RGB")
    else:
        img = file_path_or_bytes.convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0     # (H, W, 3)
    return np.transpose(arr, (2, 0, 1))                # (3, H, W)


def _normalize_bit_depth(arr: np.ndarray) -> np.ndarray:
    """
    Normalizes a raw satellite image array to [0.0, 1.0] float32.

    Handles:
      - uint8  → divide by 255
      - uint16 → Sentinel-2 DN: divide by 10000, clamp to [0, 1]
      - float  → if range > 1.0, assume DN; else already normalized
    """
    if arr.dtype == np.uint8:
        return arr.astype(np.float32) / 255.0

    arr = arr.astype(np.float32)
    max_val = float(arr.max())

    if max_val <= 1.0:
        return arr.clip(0.0, 1.0)                      # already in [0, 1]

    if max_val > 1.0:
        # Assume Sentinel-2 DN (typical range 0 – 10000, can go up to ~16000)
        return (arr / SENTINEL2_SCALE).clip(0.0, 1.0)

    return arr.clip(0.0, 1.0)


def _ensure_four_bands(arr: np.ndarray) -> np.ndarray:
    """
    Ensures the array has exactly 4 channels.
    - 1-band  → replicate to 4
    - 2-band  → B, G synthetic
    - 3-band  → synthesize NIR proxy from 0.6*R + 0.4*G + 0.1
    - 4-band  → return as-is
    - >4-band → use first 4
    """
    c = arr.shape[0]
    if c == 4:
        return arr
    if c > 4:
        return arr[:4]
    if c == 1:
        return np.repeat(arr, 4, axis=0)
    if c == 2:
        nir = np.clip(arr[0] * 0.6 + arr[1] * 0.4 + 0.1, 0.0, 1.0)[np.newaxis]
        return np.concatenate([arr, arr, nir], axis=0)[:4]
    if c == 3:
        nir = np.clip(arr[0] * 0.6 + arr[1] * 0.4 + 0.1, 0.0, 1.0)[np.newaxis]
        return np.concatenate([arr, nir], axis=0)
    return arr


def _resize_to_patch(arr: np.ndarray, patch_size: int = 128) -> np.ndarray:
    """
    Resizes (C, H, W) to at least patch_size on both axes while preserving aspect ratio,
    then center-crops to exactly (C, patch_size, patch_size).
    """
    c, h, w = arr.shape
    # Scale so shortest side >= patch_size
    scale = max(patch_size / h, patch_size / w)
    if scale > 1.0:
        new_h, new_w = int(np.ceil(h * scale)), int(np.ceil(w * scale))
        t = torch.from_numpy(arr).unsqueeze(0)                        # (1, C, H, W)
        t = F.interpolate(t, size=(new_h, new_w), mode="bicubic", align_corners=False)
        arr = t.squeeze(0).numpy()
        h, w = new_h, new_w

    # Center crop
    top  = (h - patch_size) // 2
    left = (w - patch_size) // 2
    return arr[:, top:top + patch_size, left:left + patch_size]


def load_input_image(
    file_path: str,
    patch_size: int = 128,
    target_channels: int = 4,
) -> np.ndarray:
    """
    Full preprocessing pipeline for uploaded satellite images.

    Steps:
      1. Auto-detect GeoTIFF vs. RGB and read with appropriate backend.
      2. Normalize bit-depth (uint8 / uint16 / float DN) to [0, 1].
      3. Ensure exactly 4 spectral channels.
      4. Resize to >= patch_size and center-crop to (4, patch_size, patch_size).

    Args:
        file_path      : Path to uploaded image.
        patch_size     : Target spatial size fed to the model. Default 128.
        target_channels: Number of output bands. Default 4.

    Returns:
        Preprocessed float32 numpy array of shape (4, patch_size, patch_size) in [0, 1].
    """
    ext = file_path.lower().split(".")[-1]

    if ext in ("tif", "tiff"):
        arr = _read_geotiff(file_path)
    else:
        arr = _read_standard_image(file_path)

    arr = _normalize_bit_depth(arr)
    arr = _ensure_four_bands(arr)
    arr = _resize_to_patch(arr, patch_size=patch_size)

    return arr.astype(np.float32)


def numpy_to_tensor(arr: np.ndarray, device: torch.device) -> torch.Tensor:
    """Converts (C, H, W) numpy array to (1, C, H, W) float32 tensor on device."""
    return torch.from_numpy(arr).float().unsqueeze(0).to(device)


# ─── Visualization ────────────────────────────────────────────────────────────

def tensor_to_rgb(
    t: Union[torch.Tensor, np.ndarray],
    gamma: float = 1.0,
) -> np.ndarray:
    """
    Converts (C, H, W) or (1, C, H, W) multi-band float tensor to (H, W, 3) RGB [0, 1].

    Uses bands 0, 1, 2 as Blue, Green, Red composited to RGB visual.
    Applies optional gamma stretch for visual contrast enhancement.
    """
    if isinstance(t, torch.Tensor):
        arr = t.detach().cpu().float().numpy()
    else:
        arr = np.array(t, dtype=np.float32)

    if arr.ndim == 4:
        arr = arr[0]

    c, h, w = arr.shape
    if c >= 3:
        # BGR -> RGB convention for natural color composites
        rgb = np.stack([arr[2], arr[1], arr[0]], axis=-1)  # (H, W, 3)
    elif c == 2:
        rgb = np.stack([arr[0], arr[1], arr[0]], axis=-1)
    else:
        rgb = np.repeat(arr[0:1].transpose(1, 2, 0), 3, axis=2)

    rgb = np.clip(rgb, 0.0, 1.0)

    # Optional gamma stretch to improve visual contrast
    if gamma != 1.0:
        rgb = np.power(rgb, 1.0 / gamma)

    return np.clip(rgb, 0.0, 1.0)


def render_uncertainty_heatmap(
    unc_map: Union[torch.Tensor, np.ndarray],
    cmap: str = "inferno",
    title: str = "Uncertainty Heatmap",
) -> np.ndarray:
    """
    Renders the per-pixel epistemic uncertainty map as a colormapped RGB image.

    Aggregates across all channels via mean to produce a 2D spatial map,
    then applies the Inferno colormap (black = confident, yellow = uncertain).

    Returns:
        (H, W, 3) uint8 RGB array ready for display.
    """
    if isinstance(unc_map, torch.Tensor):
        unc_np = unc_map.detach().cpu().numpy()
    else:
        unc_np = np.asarray(unc_map, dtype=np.float32)

    if unc_np.ndim == 4:
        unc_np = unc_np[0]

    # Aggregate across bands
    spatial = np.mean(unc_np, axis=0)  # (H, W)

    # Normalize to [0, 1] for colormap
    vmin, vmax = spatial.min(), spatial.max()
    if vmax > vmin:
        norm = (spatial - vmin) / (vmax - vmin)
    else:
        norm = spatial * 0.0

    cmap_fn = plt.get_cmap(cmap)
    rgb = (cmap_fn(norm)[:, :, :3] * 255).astype(np.uint8)   # (H, W, 3)
    return rgb


def build_comparison_figure(
    lr_rgb: np.ndarray,
    sr_rgb: np.ndarray,
    unc_rgb: np.ndarray,
    lr_shape: Tuple[int, int],
    sr_shape: Tuple[int, int],
    psnr: Optional[float] = None,
    ssim: Optional[float] = None,
    unc_mean: float = 0.0,
    duration: float = 0.0,
) -> np.ndarray:
    """
    Builds a professional 4-panel side-by-side comparison figure for display.

    Panels:
        1. Original Low-Resolution Input
        2. BhuVistaar Super-Resolved Output
        3. MC Uncertainty Heatmap
        4. Inferno colorbar legend

    Returns:
        (H, W, 3) uint8 numpy array of the rendered figure.
    """
    fig = plt.figure(figsize=(18, 5.5), facecolor="#0b1426")
    gs  = gridspec.GridSpec(1, 4, figure=fig, width_ratios=[1, 1, 1, 0.06], wspace=0.08)

    axes = [fig.add_subplot(gs[i]) for i in range(3)]
    cax  = fig.add_subplot(gs[3])

    panel_data = [
        (lr_rgb, f"Original Input\n{lr_shape[0]} x {lr_shape[1]} px  |  10m Resolution",   "#94a3b8"),
        (sr_rgb, f"BhuVistaar SR Output\n{sr_shape[0]} x {sr_shape[1]} px  |  <4m Resolution", "#00f2fe"),
        (unc_rgb, f"Uncertainty Heatmap\nMean σ = {unc_mean:.4f}  |  MC Passes = 15",         "#f43f5e"),
    ]

    for ax, (img, title, color) in zip(axes, panel_data):
        ax.imshow(img)
        ax.set_facecolor("#0b1426")
        ax.set_title(title, color=color, fontsize=10, fontweight="bold", pad=8)
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(1.2)
            spine.set_visible(True)

    # Colorbar for uncertainty
    import matplotlib.cm as cm
    norm_cb = matplotlib.colors.Normalize(vmin=0, vmax=1)
    cb = fig.colorbar(
        cm.ScalarMappable(norm=norm_cb, cmap="inferno"),
        cax=cax, orientation="vertical"
    )
    cb.set_ticks([0.0, 0.5, 1.0])
    cb.set_ticklabels(["Low\nσ", "", "High\nσ"])
    cb.ax.yaxis.set_tick_params(color="white", labelsize=8)
    plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color="white")
    cb.outline.set_edgecolor("rgba(255,255,255,0.2)")

    # Footer metrics bar
    metrics_parts = []
    if psnr is not None:
        metrics_parts.append(f"PSNR = {psnr:.2f} dB")
    if ssim is not None:
        metrics_parts.append(f"SSIM = {ssim:.4f}")
    metrics_parts.append(f"Inference = {duration:.2f}s")
    metrics_parts.append("Team: The Outliers  |  SIH 2026")

    fig.text(
        0.5, 0.01, "    |    ".join(metrics_parts),
        ha="center", va="bottom", fontsize=9,
        color="#64748b", style="italic"
    )

    fig.tight_layout(rect=[0, 0.04, 1, 1])

    # Render to numpy
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    img_pil = Image.open(buf).convert("RGB")
    return np.array(img_pil)


def compute_metrics(
    sr_np: np.ndarray,
    hr_ref: np.ndarray,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Computes PSNR (dB) and SSIM against a ground-truth reference.

    Args:
        sr_np  : Super-resolved output (C, H, W) float32 in [0, 1].
        hr_ref : Ground-truth HR reference (C, H, W) float32 in [0, 1].

    Returns:
        (psnr, ssim) — None if computation fails.
    """
    try:
        from skimage.metrics import peak_signal_noise_ratio as psnr_fn
        from skimage.metrics import structural_similarity as ssim_fn

        psnr = psnr_fn(hr_ref, sr_np, data_range=1.0)
        ssim = ssim_fn(
            np.transpose(hr_ref, (1, 2, 0)),
            np.transpose(sr_np,  (1, 2, 0)),
            channel_axis=-1,
            data_range=1.0
        )
        return float(psnr), float(ssim)
    except Exception:
        return None, None
