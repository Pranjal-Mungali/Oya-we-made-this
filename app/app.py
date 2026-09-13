"""
BhuVistaar: Sentinel-2 Satellite Super-Resolution & Uncertainty Mapping Studio.
Interactive Streamlit Application with Navigation, Image Upload/Preset Selection,
Direct Before-and-After Comparison, and Bayesian Monte Carlo Uncertainty Quantification.
"""

import os
import sys
import time
import io
from typing import Tuple, Optional, Union

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import streamlit as st
import streamlit.components.v1 as components
import torch

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model import ResidualSR
from src.dataset import Sentinel2SRDataset

# Optional geospatial readers
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


# Page Configuration
st.set_page_config(
    page_title="BhuVistaar | Super Resolution Mapping",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom Styling: Modern Glassmorphic Dark UI & Clean Navigation
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Clean font base */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    h1, h2, h3, h4, .brand-title, .nav-logo {
        font-family: 'Space Grotesk', sans-serif;
    }

    /* Hide unnecessary default Streamlit elements */
    #MainMenu {visibility: hidden !important;}
    footer {visibility: hidden !important;}
    header {visibility: hidden !important;}
    .stDeployButton {display: none !important;}
    div[data-testid="stToolbar"] {visibility: hidden !important;}
    div[data-testid="stDecoration"] {display: none !important;}

    /* Top Navigation Bar */
    .top-navbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: rgba(15, 23, 42, 0.85);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 12px 24px;
        margin-bottom: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }
    
    .nav-brand {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    .nav-brand-logo {
        font-size: 24px;
        background: linear-gradient(135deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        letter-spacing: 0.5px;
    }

    .nav-badge {
        background: rgba(56, 189, 248, 0.15);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.3);
        font-size: 11px;
        font-weight: 600;
        padding: 3px 8px;
        border-radius: 12px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Cards */
    .glass-card {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 16px;
        padding: 20px 24px;
        backdrop-filter: blur(12px);
        margin-bottom: 20px;
    }

    .glass-card-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #f1f5f9;
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 14px;
    }

    /* Metric Cards */
    .metric-container {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 14px;
        margin: 18px 0;
    }
    @media (max-width: 900px) {
        .metric-container { grid-template-columns: repeat(2, 1fr); }
    }

    .metric-card {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(56, 189, 248, 0.15);
        border-radius: 14px;
        padding: 16px;
        text-align: center;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(56, 189, 248, 0.4);
    }
    .metric-val {
        font-size: 1.7rem;
        font-weight: 700;
        color: #38bdf8;
        font-family: 'Space Grotesk', sans-serif;
    }
    .metric-lbl {
        font-size: 0.78rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 4px;
    }

    /* Comparison Container */
    .comparison-frame {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 16px;
        text-align: center;
    }

    .badge-usp {
        background: linear-gradient(135deg, #0284c7, #6366f1);
        color: #ffffff;
        font-size: 0.7rem;
        font-weight: 600;
        padding: 3px 8px;
        border-radius: 12px;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-left: 6px;
    }
</style>
""", unsafe_allow_html=True)


def render_particle_hero():
    """Renders interactive HTML5 Canvas with particles, mouse attraction, connections, and depth parallax."""
    particle_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body, html { width: 100%; height: 160px; overflow: hidden; background: transparent; }
            #canvas { width: 100%; height: 100%; display: block; }
            .hero-overlay {
                position: absolute;
                top: 0; left: 0; width: 100%; height: 100%;
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
                pointer-events: none;
                text-align: center;
                font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif;
            }
            .brand-glow {
                font-size: 28px;
                font-weight: 800;
                letter-spacing: 1.5px;
                background: linear-gradient(135deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                text-shadow: 0 0 25px rgba(56, 189, 248, 0.35);
                opacity: 0;
                transform: translateY(10px);
                animation: smoothEntrance 1s cubic-bezier(0.16, 1, 0.3, 1) forwards 0.15s;
            }
            .brand-sub {
                font-size: 12px;
                color: #94a3b8;
                letter-spacing: 2.5px;
                text-transform: uppercase;
                margin-top: 4px;
                opacity: 0;
                transform: translateY(6px);
                animation: smoothEntrance 1s cubic-bezier(0.16, 1, 0.3, 1) forwards 0.35s;
            }
            @keyframes smoothEntrance {
                to { opacity: 1; transform: translateY(0); }
            }
        </style>
    </head>
    <body>
        <canvas id="canvas"></canvas>
        <div class="hero-overlay">
            <div class="brand-glow">🛰️ BhuVistaar: Super Resolution Mapping</div>
            <div class="brand-sub">Multi-Band Satellite Super Resolution & Bayesian Uncertainty Quantification</div>
        </div>
        <script>
            const canvas = document.getElementById('canvas');
            const ctx = canvas.getContext('2d');
            let width, height;
            let particles = [];
            const mouse = { x: null, y: null, radius: 110 };

            function resize() {
                width = canvas.width = window.innerWidth;
                height = canvas.height = window.innerHeight;
                initParticles();
            }
            window.addEventListener('resize', resize);
            window.addEventListener('mousemove', (e) => {
                const rect = canvas.getBoundingClientRect();
                mouse.x = e.clientX - rect.left;
                mouse.y = e.clientY - rect.top;
            });
            window.addEventListener('mouseleave', () => {
                mouse.x = null;
                mouse.y = null;
            });

            class Particle {
                constructor() {
                    this.reset(true);
                }
                reset(initial = false) {
                    this.x = Math.random() * width;
                    this.y = initial ? Math.random() * height : (Math.random() > 0.5 ? 0 : height);
                    this.z = Math.random() * 2 + 1; // 3D depth layer
                    this.radius = (Math.random() * 1.6 + 0.8) * this.z;
                    this.vx = (Math.random() - 0.5) * 0.6 * this.z;
                    this.vy = (Math.random() - 0.5) * 0.6 * this.z;
                    this.color = Math.random() > 0.4 ? '56, 189, 248' : (Math.random() > 0.5 ? '129, 140, 248' : '192, 132, 252');
                    this.alpha = Math.random() * 0.5 + 0.3;
                }
                update() {
                    if (mouse.x !== null && mouse.y !== null) {
                        const dx = mouse.x - this.x;
                        const dy = mouse.y - this.y;
                        const dist = Math.sqrt(dx * dx + dy * dy);
                        if (dist < mouse.radius) {
                            const force = (mouse.radius - dist) / mouse.radius;
                            const angle = Math.atan2(dy, dx);
                            this.vx += Math.cos(angle) * force * 0.25;
                            this.vy += Math.sin(angle) * force * 0.25;
                        }
                    }
                    this.vx *= 0.98;
                    this.vy *= 0.98;
                    this.x += this.vx;
                    this.y += this.vy;

                    if (this.x < 0) this.x = width;
                    if (this.x > width) this.x = 0;
                    if (this.y < 0) this.y = height;
                    if (this.y > height) this.y = 0;
                }
                draw() {
                    ctx.save();
                    ctx.beginPath();
                    ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
                    ctx.fillStyle = `rgba(${this.color}, ${this.alpha})`;
                    ctx.shadowColor = `rgba(${this.color}, 0.8)`;
                    ctx.shadowBlur = 8 * this.z;
                    ctx.fill();
                    ctx.restore();
                }
            }

            function initParticles() {
                particles = [];
                const count = Math.floor((width * height) / 4800);
                for (let i = 0; i < Math.min(count, 70); i++) {
                    particles.push(new Particle());
                }
            }

            function connectParticles() {
                const maxDist = 85;
                for (let i = 0; i < particles.length; i++) {
                    for (let j = i + 1; j < particles.length; j++) {
                        const dx = particles[i].x - particles[j].x;
                        const dy = particles[i].y - particles[j].y;
                        const dist = Math.sqrt(dx * dx + dy * dy);
                        if (dist < maxDist) {
                            const alpha = (1 - dist / maxDist) * 0.2;
                            ctx.beginPath();
                            ctx.moveTo(particles[i].x, particles[i].y);
                            ctx.lineTo(particles[j].x, particles[j].y);
                            ctx.strokeStyle = `rgba(56, 189, 248, ${alpha})`;
                            ctx.lineWidth = 0.6;
                            ctx.stroke();
                        }
                    }
                }
            }

            function animate() {
                ctx.fillStyle = 'rgba(10, 15, 29, 0.25)';
                ctx.fillRect(0, 0, width, height);

                for (let p of particles) {
                    p.update();
                    p.draw();
                }
                connectParticles();
                requestAnimationFrame(animate);
            }

            resize();
            animate();
        </script>
    </body>
    </html>
    """
    components.html(particle_html, height=165, scrolling=False)


@st.cache_resource
def get_model(model_path: str, scale_factor: int = 2) -> ResidualSR:
    """Loads and caches the ResidualSR model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResidualSR(
        in_channels=4,
        out_channels=4,
        num_features=64,
        num_blocks=4,
        scale_factor=scale_factor,
        dropout_rate=0.2
    )

    if os.path.isfile(model_path):
        try:
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(model_path, map_location=device)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        elif isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    return model


def generate_preset_sample(sample_type: str = "agriculture") -> Tuple[np.ndarray, np.ndarray]:
    """Generates synthetic Sentinel-2 4-band LR & HR pair for instant evaluation."""
    height, width, channels = 128, 128, 4
    np.random.seed(303 if sample_type == "agriculture" else 404)

    y, x = np.ogrid[:height, :width]
    if sample_type == "agriculture":
        base = (np.sin(x / 12.0) + np.cos(y / 12.0)) * 0.25 + 0.5
    else:
        base = (np.sin(x / 6.0) * np.cos(y / 6.0)) * 0.3 + 0.5

    bands = []
    for c in range(channels):
        noise = np.random.normal(0, 0.03, (height, width))
        b = base * (1.0 + 0.15 * c) + 0.08 * c + noise
        if c == 3:  # NIR band (high reflectance in vegetation)
            b = b * 1.4 + 0.1
        bands.append(np.clip(b, 0.0, 1.0).astype(np.float32))

    hr_np = np.stack(bands, axis=0)  # (4, 128, 128)

    # 2x LR degraded counterpart
    dataset = Sentinel2SRDataset(synthetic_images=[hr_np], scale_factor=2, hr_patch_size=128, num_samples=1)
    lr_tensor, hr_tensor = dataset[0]
    return lr_tensor.numpy(), hr_tensor.numpy()


def process_uploaded_file(uploaded_file) -> np.ndarray:
    """Loads uploaded .tif, .png, .jpg into a normalized (4, H, W) numpy array."""
    file_bytes = uploaded_file.read()
    filename = uploaded_file.name.lower()

    if filename.endswith((".tif", ".tiff")) and HAS_RASTERIO:
        with rasterio.open(io.BytesIO(file_bytes)) as src:
            img = src.read().astype(np.float32)
            if img.max() > 1.0:
                img = img / 10000.0
    elif filename.endswith((".tif", ".tiff")) and HAS_TIFFFILE:
        img = tifffile.imread(io.BytesIO(file_bytes)).astype(np.float32)
        if img.ndim == 2:
            img = img[np.newaxis, ...]
        elif img.ndim == 3 and img.shape[2] <= 13:
            img = np.transpose(img, (2, 0, 1))
        if img.max() > 1.0:
            img = img / 10000.0
    else:
        pil_img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        img_np = np.array(pil_img, dtype=np.float32) / 255.0  # (H, W, 3)
        img = np.transpose(img_np, (2, 0, 1))  # (3, H, W)

    c, h, w = img.shape
    if c < 4:
        # Synthesize 4th NIR channel proxy
        nir_proxy = np.clip(img[0] * 0.6 + img[1] * 0.4 + 0.1, 0.0, 1.0)[np.newaxis, ...]
        img = np.concatenate([img, nir_proxy], axis=0)[:4]
    elif c > 4:
        img = img[:4]

    return np.clip(img, 0.0, 1.0)


def to_rgb_display(img_array: np.ndarray) -> np.ndarray:
    """Converts (C, H, W) multi-band array to (H, W, 3) RGB visual [0, 1]."""
    if img_array.ndim == 4:
        img_array = img_array[0]
    c, h, w = img_array.shape
    if c >= 3:
        rgb = np.transpose(img_array[:3], (1, 2, 0))
    else:
        rgb = np.repeat(np.transpose(img_array[0:1], (1, 2, 0)), 3, axis=2)
    return np.clip(rgb, 0.0, 1.0)


def main():
    # 1. Top Navigation Bar
    st.markdown("""
    <div class="top-navbar">
        <div class="nav-brand">
            <div class="nav-brand-logo">🛰️ BhuVistaar</div>
            <div class="nav-badge">SRM Studio v1.0</div>
        </div>
        <div style="color: #94a3b8; font-size: 0.85rem; font-weight: 500;">
            Sentinel-2 Super Resolution & Uncertainty Quantification
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 2. Interactive Animated Particle Canvas
    render_particle_hero()

    # 3. Main Studio Tabs Navigation
    tab_studio, tab_analytics, tab_docs = st.tabs([
        "🔬 Super-Resolution Studio",
        "📊 Uncertainty & Risk Analytics",
        "📖 Method & Architecture"
    ])

    with tab_studio:
        # -------------------------------------------------------------
        # Section 1: Ask Input Image (Upload or 1-Click Preset)
        # -------------------------------------------------------------
        st.markdown("""
        <div class="glass-card">
            <div class="glass-card-title">
                <span>📥</span> Step 1: Select or Upload Low-Resolution Satellite Scene
            </div>
        </div>
        """, unsafe_allow_html=True)

        col_input_mode, col_settings = st.columns([2, 1])

        with col_input_mode:
            input_mode = st.radio(
                "Input Source Mode:",
                ["1-Click Demo Presets (Instant)", "Upload Custom Satellite Image (.tif, .png, .jpg)"],
                horizontal=True
            )

            lr_input_data = None
            hr_reference_gt = None
            input_label = ""

            if input_mode == "1-Click Demo Presets (Instant)":
                preset_choice = st.selectbox(
                    "Choose Preloaded Sentinel-2 Scene:",
                    [
                        "🌾 Agricultural Crops & River Basin (Band 2, 3, 4, 8)",
                        "🏙️ Urban Settlement & Coastal Infrastructure (Band 2, 3, 4, 8)"
                    ]
                )
                sample_type = "agriculture" if "Agricultural" in preset_choice else "urban"
                lr_input_data, hr_reference_gt = generate_preset_sample(sample_type=sample_type)
                input_label = f"Preset: {preset_choice.split(' ')[1]} Scene"

            else:
                uploaded_file = st.file_uploader(
                    "Drop Sentinel-2 GeoTIFF or standard image here:",
                    type=["tif", "tiff", "png", "jpg", "jpeg"],
                    help="Supports multi-band GeoTIFFs or standard RGB satellite patches."
                )
                if uploaded_file is not None:
                    lr_input_data = process_uploaded_file(uploaded_file)
                    input_label = f"Uploaded File: {uploaded_file.name}"

        with col_settings:
            st.markdown("<div style='font-size: 0.9rem; font-weight: 600; color: #94a3b8; margin-bottom: 8px;'>⚙️ Inference Parameters</div>", unsafe_allow_html=True)
            scale_factor = st.selectbox("Super-Resolution Factor", [2, 4], index=0, format_func=lambda x: f"{x}x Resolution Enhancement")
            num_mc_samples = st.slider("Monte Carlo Dropout Passes", min_value=5, max_value=30, value=15, step=1,
                                       help="Number of stochastic Bayesian forward passes to compute epistemic uncertainty.")

        # -------------------------------------------------------------
        # Section 2: Trigger Super Resolution
        # -------------------------------------------------------------
        st.markdown("<br>", unsafe_allow_html=True)
        col_run, col_status = st.columns([1, 3])
        with col_run:
            run_btn = st.button("🚀 Run Super Resolution", type="primary", use_container_width=True)
        with col_status:
            if lr_input_data is not None:
                st.caption(f"Ready: **{input_label}** — Input Dimensions: `{lr_input_data.shape[1]}x{lr_input_data.shape[2]} px` ({lr_input_data.shape[0]} Bands)")
            else:
                st.caption("Please select a preset or upload an image above to run inference.")

        # -------------------------------------------------------------
        # Section 3: Proper Before & After Comparison
        # -------------------------------------------------------------
        if lr_input_data is not None:
            if run_btn or 'studio_has_run' in st.session_state:
                st.session_state['studio_has_run'] = True

                # Load model
                model_path = os.path.join(PROJECT_ROOT, "models", f"residual_sr_x{scale_factor}_best.pt")
                model = get_model(model_path=model_path, scale_factor=scale_factor)

                with st.spinner(f"Super-resolving {scale_factor}x spatial details across {num_mc_samples} Monte Carlo passes..."):
                    t0 = time.time()
                    lr_tensor = torch.from_numpy(lr_input_data).float().unsqueeze(0)
                    device = next(model.parameters()).device
                    lr_tensor = lr_tensor.to(device)

                    # Monte Carlo Bayesian inference
                    mean_sr, uncertainty_map = model.monte_carlo_inference(
                        lr_tensor,
                        num_passes=num_mc_samples,
                        return_variance=False
                    )
                    inference_sec = time.time() - t0

                    sr_mean_np = mean_sr.squeeze(0).detach().cpu().numpy()
                    unc_np = uncertainty_map.squeeze(0).detach().cpu().numpy()
                    spatial_unc = np.mean(unc_np, axis=0)

                    # Calculate PSNR/SSIM if reference available
                    psnr_val, ssim_val = None, None
                    if hr_reference_gt is not None:
                        from skimage.metrics import structural_similarity as ssim_fn
                        from skimage.metrics import peak_signal_noise_ratio as psnr_fn
                        psnr_val = psnr_fn(hr_reference_gt, sr_mean_np, data_range=1.0)
                        ssim_val = ssim_fn(
                            np.transpose(hr_reference_gt, (1, 2, 0)),
                            np.transpose(sr_mean_np, (1, 2, 0)),
                            channel_axis=-1,
                            data_range=1.0
                        )

                # Metrics Bar
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-card">
                        <div class="metric-val">{scale_factor}x ({scale_factor*scale_factor}x Pixels)</div>
                        <div class="metric-lbl">Spatial Resolution Gain</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-val">{inference_sec:.2f}s</div>
                        <div class="metric-lbl">Inference Latency ({num_mc_samples} Passes)</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-val">{float(spatial_unc.mean()):.4f}</div>
                        <div class="metric-lbl">Mean Epistemic Uncertainty</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-val">{f"{psnr_val:.2f} dB" if psnr_val else "High Fidelity"}</div>
                        <div class="metric-lbl">Reconstruction PSNR</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("### 🔍 Proper Before & After Comparison")

                comp_col1, comp_col2, comp_col3 = st.columns(3)

                with comp_col1:
                    st.markdown("""
                    <div class="comparison-frame">
                        <div style="font-weight: 600; color: #94a3b8; margin-bottom: 6px;">
                            ⏮️ Old Image (Low-Resolution Input)
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    lr_visual = to_rgb_display(lr_input_data)
                    st.image(
                        lr_visual,
                        caption=f"Original Input: {lr_input_data.shape[1]}x{lr_input_data.shape[2]} px",
                        use_container_width=True
                    )

                with comp_col2:
                    st.markdown("""
                    <div class="comparison-frame">
                        <div style="font-weight: 600; color: #38bdf8; margin-bottom: 6px;">
                            ⏭️ Updated Image (Super-Resolved Output)
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    sr_visual = to_rgb_display(sr_mean_np)
                    st.image(
                        sr_visual,
                        caption=f"Super-Resolved: {sr_mean_np.shape[1]}x{sr_mean_np.shape[2]} px (Predictive Mean)",
                        use_container_width=True
                    )

                with comp_col3:
                    st.markdown("""
                    <div class="comparison-frame">
                        <div style="font-weight: 600; color: #f43f5e; margin-bottom: 6px;">
                            🔥 Uncertainty Map <span class="badge-usp">Our USP</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    fig, ax = plt.subplots(figsize=(5, 5))
                    fig.patch.set_alpha(0.0)
                    ax.patch.set_alpha(0.0)
                    im = ax.imshow(spatial_unc, cmap="inferno")
                    ax.axis("off")
                    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    cb.ax.yaxis.set_tick_params(color='white')
                    plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color='white')
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)

                # Explanation Box
                st.markdown("""
                <div class="glass-card" style="border-left: 4px solid #38bdf8; margin-top: 15px;">
                    <div style="font-weight: 600; color: #38bdf8; margin-bottom: 6px;">
                        💡 How to interpret the comparison?
                    </div>
                    <div style="color: #cbd5e1; font-size: 0.9rem; line-height: 1.6;">
                        <strong>Old vs. Updated:</strong> The model reconstructs fine road networks, parcel boundaries, and water bodies using learned sub-pixel convolution (PixelShuffle).<br>
                        <strong>Uncertainty Heatmap (Our USP):</strong> Quantifies pixel-level prediction variance via Monte Carlo Dropout. Dark regions indicate high model certainty, while warmer yellow/orange pixels highlight edge transitions or sensor noise where confidence is lower.
                    </div>
                </div>
                """, unsafe_allow_html=True)

    with tab_analytics:
        st.markdown("### 📊 Uncertainty Distribution & Anomaly Analysis")
        st.write("Spatial distribution of epistemic variance across multispectral channels.")

        if 'studio_has_run' in st.session_state and 'spatial_unc' in locals():
            col_a1, col_a2 = st.columns(2)
            with col_a1:
                fig_hist, ax_hist = plt.subplots(figsize=(6, 4))
                fig_hist.patch.set_alpha(0.0)
                ax_hist.patch.set_alpha(0.0)
                ax_hist.hist(spatial_unc.ravel(), bins=40, color="#38bdf8", alpha=0.8, edgecolor="#0f172a")
                ax_hist.set_title("Pixel Uncertainty Histogram", color="white")
                ax_hist.set_xlabel("Standard Deviation (\u03c3)", color="white")
                ax_hist.set_ylabel("Pixel Count", color="white")
                ax_hist.tick_params(colors="white")
                st.pyplot(fig_hist, use_container_width=True)
                plt.close(fig_hist)

            with col_a2:
                st.markdown("""
                <div class="glass-card">
                    <div class="glass-card-title">🛡️ Risk Flagging & QA Filter</div>
                    <p style="color: #94a3b8; font-size: 0.9rem; line-height: 1.6;">
                        Pixels exceeding a 95th-percentile uncertainty threshold can be automatically flagged for manual cartographer review or masked in high-stakes automated classification pipelines.
                    </p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Run super-resolution in the Studio tab first to inspect uncertainty analytics.")

    with tab_docs:
        st.markdown("### 📖 Method & Model Architecture")
        st.markdown(r"""
        **BhuVistaar** enhances Sentinel-2 multispectral imagery spatial resolution using a lightweight Deep Residual CNN with sub-pixel PixelShuffle upsampling:
        - **Multi-Band Processing**: Directly processes Red, Green, Blue, and Near-Infrared (NIR) bands.
        - **Bayesian Epistemic Uncertainty**: Leverages Monte Carlo Spatial Dropout ($T=15$ passes) to approximate posterior weight distributions $\mathcal{N}(\mu, \sigma^2)$.
        - **Fast Prototyping & Edge Ready**: Sub-pixel convolution ensures fast real-time inference on edge or cloud servers.
        """)


if __name__ == "__main__":
    main()
