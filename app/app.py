"""
app.py — BhuVistaar Gradio Web Application.

BhuVistaar: AI-Powered Super Resolution Mapping from Sentinel-2 Imagery.
Team: The Outliers | Smart India Hackathon 2026.

Converts 10m Sentinel-2 imagery to <4m equivalent resolution using a
deep Residual CNN with PixelShuffle upsampling and Monte Carlo Dropout
for pixel-level epistemic uncertainty quantification.

Designed for deployment on Hugging Face Spaces (ZERO-GPU / CPU).
"""

import os
import sys
import time
from typing import Optional, Tuple
import numpy as np
import torch
import gradio as gr

# Ensure project root is in path (works in both local dev and HF Spaces)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import from co-located app modules
from app.model import ResidualSR, load_model
from app.utils import (
    load_input_image,
    numpy_to_tensor,
    tensor_to_rgb,
    render_uncertainty_heatmap,
    build_comparison_figure,
    compute_metrics,
)
from src.dataset import Sentinel2SRDataset


# ─── Model Configuration ──────────────────────────────────────────────────────

MODEL_PATH   = os.path.join(PROJECT_ROOT, "models", "residual_sr_x2_best.pt")
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MC_PASSES    = 15
PATCH_SIZE   = 128
SCALE_FACTOR = 2


# ─── Model Singleton ──────────────────────────────────────────────────────────

_model: Optional[ResidualSR] = None


def get_model() -> ResidualSR:
    """Lazily loads and caches the BhuVistaar model (thread-safe singleton)."""
    global _model
    if _model is None:
        _model = load_model(
            checkpoint_path=MODEL_PATH,
            scale_factor=SCALE_FACTOR,
            in_channels=4,
            device=DEVICE,
        )
    return _model


# ─── Sample Scene Generator ───────────────────────────────────────────────────

def _make_structured_scene(scene_type: str = "agriculture") -> np.ndarray:
    """
    Generates a high-contrast structured Sentinel-2 4-band scene for demo use.
    Includes synthetic road grids (urban) or parcel boundaries & canals (agriculture).
    """
    h = w = PATCH_SIZE * SCALE_FACTOR   # HR size — will be degraded to LR
    np.random.seed(888 if scene_type == "agriculture" else 999)
    y, x = np.ogrid[:h, :w]

    if scene_type == "agriculture":
        field = (np.sin(x / 9.0) * np.cos(y / 9.0) > 0.05).astype(float)
        canal = (np.abs(x - y - 12) < 2.5).astype(float)
        base  = 0.32 + 0.35 * field - 0.20 * canal
    else:
        gx    = (x % 14 < 2.5).astype(float)
        gy    = (y % 14 < 2.5).astype(float)
        roads = np.clip(gx + gy, 0.0, 1.0)
        blks  = (((x // 14) % 2 == 0) & ((y // 14) % 2 == 0)).astype(float) * 0.4
        base  = 0.34 + 0.28 * roads + blks

    bands = []
    for c in range(4):
        noise = np.random.normal(0, 0.015, (h, w))
        if   c == 0: b = base * 0.82 + 0.06 + noise      # Blue  B02
        elif c == 1: b = base * 0.92 + 0.08 + noise      # Green B03
        elif c == 2: b = base * 1.00 + 0.05 + noise      # Red   B04
        else:        b = base * 1.35 + (0.28 if scene_type == "agriculture" else 0.08) + noise  # NIR B08
        bands.append(np.clip(b, 0.0, 1.0).astype(np.float32))

    hr_np = np.stack(bands)   # (4, 256, 256)

    # Degrade HR → LR using dataset pipeline for consistency with training
    ds = Sentinel2SRDataset(
        synthetic_images=[hr_np],
        scale_factor=SCALE_FACTOR,
        hr_patch_size=PATCH_SIZE,
        num_samples=1,
        blur_sigma=1.2,
        is_train=False,
    )
    lr_tensor, _ = ds[0]
    return lr_tensor.numpy()   # (4, 64, 64)


SAMPLES = {
    "Agriculture: Crop Fields & Waterway (Sentinel-2 10m)": _make_structured_scene("agriculture"),
    "Urban: Road Grid & Settlement Zone (Sentinel-2 10m)":  _make_structured_scene("urban"),
}


# ─── Core Inference Function ──────────────────────────────────────────────────

def run_super_resolution(
    upload_img,
    sample_choice: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    """
    Main inference function called by the Gradio interface.

    Args:
        upload_img    : Uploaded image path from gr.Image (file path or None).
        sample_choice : Key into SAMPLES dict for preset benchmark scenes.

    Returns:
        (lr_display, sr_display, unc_display, comparison_panel, metrics_text)
    """
    model = get_model()

    # 1. Determine and preprocess input ─────────────────────────────────────
    if upload_img is not None:
        # Uploaded file: full preprocessing pipeline (bit-depth, bands, resize)
        lr_np = load_input_image(
            file_path=upload_img,
            patch_size=PATCH_SIZE,
            target_channels=4,
        )
        source_label = os.path.basename(upload_img)
    else:
        # Preset benchmark sample
        lr_np = SAMPLES[sample_choice].copy()
        source_label = sample_choice

    # 2. Build LR input tensor ───────────────────────────────────────────────
    lr_tensor = numpy_to_tensor(lr_np, device=DEVICE)    # (1, 4, 64, 64)

    # 3. Monte Carlo Dropout Inference (15 passes) ────────────────────────────
    t_start = time.time()
    sr_tensor, unc_tensor = model.monte_carlo_inference(
        lr_tensor,
        num_passes=MC_PASSES,
        return_variance=False,   # returns std dev map
    )
    duration = time.time() - t_start

    # 4. Extract numpy arrays ─────────────────────────────────────────────────
    sr_np  = sr_tensor.squeeze(0).detach().cpu().numpy()     # (4, 128, 128)
    unc_np = unc_tensor.squeeze(0).detach().cpu().numpy()    # (4, 128, 128)

    spatial_unc = float(np.mean(unc_np))

    # 5. Visual conversions ───────────────────────────────────────────────────
    lr_rgb  = (tensor_to_rgb(lr_np,  gamma=1.2) * 255).astype(np.uint8)
    sr_rgb  = (tensor_to_rgb(sr_np,  gamma=1.2) * 255).astype(np.uint8)
    unc_rgb = render_uncertainty_heatmap(unc_np, cmap="inferno")

    # 6. Build side-by-side comparison panel ─────────────────────────────────
    comp = build_comparison_figure(
        lr_rgb=lr_rgb,
        sr_rgb=sr_rgb,
        unc_rgb=unc_rgb,
        lr_shape=(lr_np.shape[1], lr_np.shape[2]),
        sr_shape=(sr_np.shape[1], sr_np.shape[2]),
        psnr=None,
        ssim=None,
        unc_mean=spatial_unc,
        duration=duration,
    )

    # 7. Metrics text ─────────────────────────────────────────────────────────
    metrics_md = f"""
### Inference Report

| Metric | Value |
|--------|-------|
| Input Dimensions | `{lr_np.shape[1]} x {lr_np.shape[2]} px` |
| Output Dimensions | `{sr_np.shape[1]} x {sr_np.shape[2]} px` |
| Scale Factor | `{SCALE_FACTOR}x  ({SCALE_FACTOR*SCALE_FACTOR}x pixel density)` |
| MC Dropout Passes | `{MC_PASSES}` |
| Mean Epistemic Uncertainty (σ) | `{spatial_unc:.5f}` |
| Inference Latency | `{duration:.2f}s` |
| Source | `{source_label}` |
| Compute Device | `{str(DEVICE).upper()}` |

> **Uncertainty Heatmap:** Shows which pixels the model is less confident about. Brighter/warmer colours indicate lower reliability — often at high-frequency edges, cloud boundaries, or anomalous sensor regions.
"""
    return lr_rgb, sr_rgb, unc_rgb, comp, metrics_md


# ─── Gradio UI ────────────────────────────────────────────────────────────────

CSS = """
/* Global background */
body, .gradio-container {
    background: #070d18 !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif;
}

/* Card panels */
.gr-panel, .gr-box, .gr-form {
    background: rgba(15, 23, 42, 0.7) !important;
    border: 1px solid rgba(0, 242, 254, 0.12) !important;
    border-radius: 12px !important;
}

/* Primary button — BhuVistaar teal-cyan */
.gr-button-primary {
    background: linear-gradient(135deg, #0284c7 0%, #00f2fe 60%, #10b981 100%) !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 1.05rem !important;
    border: none !important;
    border-radius: 10px !important;
    box-shadow: 0 4px 24px rgba(0, 242, 254, 0.3) !important;
    transition: transform 0.2s ease, box-shadow 0.2s ease !important;
}
.gr-button-primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 32px rgba(0, 242, 254, 0.45) !important;
}

/* Image display captions */
.gr-image-label, label span {
    color: #94a3b8 !important;
    font-size: 0.8rem !important;
}

/* Tab styling */
.tab-nav button {
    color: #94a3b8 !important;
    font-weight: 600 !important;
    border-radius: 8px 8px 0 0 !important;
}
.tab-nav button.selected {
    color: #00f2fe !important;
    border-bottom: 2px solid #00f2fe !important;
}

/* Headers */
h1 { color: #f8fafc !important; font-family: 'Space Grotesk', sans-serif; }
h2, h3 { color: #e2e8f0 !important; }
"""

TITLE = """
<div style="
    background: linear-gradient(135deg, #0b1426, #0a1e30);
    border: 1px solid rgba(0, 242, 254, 0.2);
    border-radius: 14px;
    padding: 24px 32px 18px;
    margin-bottom: 16px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
">
    <h1 style="
        margin: 0 0 6px;
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00f2fe 0%, #4facfe 45%, #10b981 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: 0.5px;
    ">BhuVistaar</h1>
    <p style="margin: 0 0 10px; font-size: 1.05rem; color: #94a3b8; font-weight: 500;">
        AI-Powered Super Resolution Mapping from Sentinel-2 Imagery
    </p>
    <p style="margin: 0 0 12px; font-size: 0.9rem; color: #cbd5e1; line-height: 1.6; max-width: 860px;">
        BhuVistaar is an AI-based Super Resolution framework that converts 10m Sentinel-2 imagery
        to &lt;4m resolution, producing sharper imagery while preserving spectral and geographic
        consistency. It enables fine-scale crop, urban, and disaster analysis with built-in
        Uncertainty Quantification.
    </p>
    <div style="display: flex; gap: 10px; flex-wrap: wrap;">
        <span style="
            background: rgba(0, 242, 254, 0.12);
            color: #00f2fe;
            border: 1px solid rgba(0, 242, 254, 0.3);
            font-size: 0.73rem; font-weight: 700;
            padding: 4px 10px; border-radius: 20px; letter-spacing: 0.6px;
        ">MC DROPOUT UQ — OUR USP</span>
        <span style="
            background: rgba(16, 185, 129, 0.12);
            color: #10b981;
            border: 1px solid rgba(16, 185, 129, 0.3);
            font-size: 0.73rem; font-weight: 700;
            padding: 4px 10px; border-radius: 20px; letter-spacing: 0.6px;
        ">SPECTRAL + GEOSPATIAL CONSISTENCY</span>
        <span style="
            background: rgba(79, 172, 254, 0.12);
            color: #4facfe;
            border: 1px solid rgba(79, 172, 254, 0.3);
            font-size: 0.73rem; font-weight: 700;
            padding: 4px 10px; border-radius: 20px; letter-spacing: 0.6px;
        ">ISRO CARTOSAT ALIGNED</span>
        <span style="
            background: rgba(248, 113, 113, 0.1);
            color: #f87171;
            border: 1px solid rgba(248, 113, 113, 0.3);
            font-size: 0.73rem; font-weight: 700;
            padding: 4px 10px; border-radius: 20px; letter-spacing: 0.6px;
        ">TEAM: THE OUTLIERS — SIH 2026</span>
    </div>
</div>
"""


def build_interface() -> gr.Blocks:
    """Constructs and returns the BhuVistaar Gradio Blocks interface."""
    with gr.Blocks(css=CSS, title="BhuVistaar | SIH 2026") as demo:

        gr.HTML(TITLE)

        with gr.Row():
            # Left: Input Column
            with gr.Column(scale=1):
                gr.Markdown("### Input Configuration")

                upload_input = gr.Image(
                    label="Upload Satellite Image (.tif, .png, .jpg)",
                    type="filepath",
                    sources=["upload"],
                    height=220,
                )

                gr.Markdown(
                    "<div style='text-align: center; color: #64748b; font-size: 0.82rem;'>or choose a preloaded benchmark sample</div>"
                )

                sample_dd = gr.Dropdown(
                    label="Benchmark Sample Scene",
                    choices=list(SAMPLES.keys()),
                    value=list(SAMPLES.keys())[0],
                )

                run_btn = gr.Button(
                    "Generate Super Resolution with Uncertainty",
                    variant="primary",
                )

                gr.Markdown("""
<div style="
    background: rgba(15, 23, 42, 0.7);
    border: 1px solid rgba(0, 242, 254, 0.12);
    border-left: 4px solid #00f2fe;
    border-radius: 10px;
    padding: 12px 16px;
    margin-top: 14px;
    font-size: 0.84rem;
    color: #94a3b8;
    line-height: 1.5;
">
<strong style="color: #00f2fe;">Supported inputs:</strong><br>
• Sentinel-2 GeoTIFF (4-band: B02 B03 B04 B08)<br>
• Standard RGB images (PNG / JPEG)<br>
• Auto-handles uint8 / uint16 / float DN normalization<br>
• Auto-synthesizes NIR band for RGB inputs
</div>
""")

            # Right: Output Column
            with gr.Column(scale=2):
                gr.Markdown("### Results")

                with gr.Tabs():
                    with gr.TabItem("Side-by-Side Comparison"):
                        comparison_out = gr.Image(
                            label="Full Comparison Panel",
                            height=340,
                            show_download_button=True,
                        )

                    with gr.TabItem("Individual Outputs"):
                        with gr.Row():
                            lr_out  = gr.Image(label="Original Low-Resolution Input",       height=230)
                            sr_out  = gr.Image(label="BhuVistaar Super-Resolved Output",    height=230)
                            unc_out = gr.Image(label="Uncertainty Heatmap (Epistemic, σ)", height=230)

                    with gr.TabItem("Inference Report"):
                        metrics_out = gr.Markdown()

        # Footer
        gr.HTML("""
        <div style="
            text-align: center;
            color: #334155;
            font-size: 0.8rem;
            margin-top: 24px;
            padding: 12px;
            border-top: 1px solid rgba(255, 255, 255, 0.05);
        ">
            BhuVistaar &nbsp;|&nbsp; Team: The Outliers &nbsp;|&nbsp;
            Smart India Hackathon 2026 &nbsp;|&nbsp;
            Residual CNN + PixelShuffle + Monte Carlo Dropout
        </div>
        """)

        # Wire up
        run_btn.click(
            fn=run_super_resolution,
            inputs=[upload_input, sample_dd],
            outputs=[lr_out, sr_out, unc_out, comparison_out, metrics_out],
            show_progress="full",
        )

    return demo


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo = build_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_api=False,
    )
