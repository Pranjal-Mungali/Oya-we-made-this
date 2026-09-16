# BhuVistaar

**AI-Powered Super Resolution Mapping from Sentinel-2 Imagery**

> Team: **The Outliers** | Smart India Hackathon 2026

---

## Overview

BhuVistaar is an AI-based Super Resolution framework that converts 10m Sentinel-2 imagery to <4m resolution, producing sharper imagery while preserving spectral and geographic consistency. It enables fine-scale crop, urban, and disaster analysis with built-in Uncertainty Quantification.

### Key Innovations

| Feature | Description |
|---------|-------------|
| **Monte Carlo Dropout UQ** | Bayesian epistemic uncertainty maps per pixel (Our USP) |
| **Spectral Consistency** | Multi-band NDVI / reflectance preservation across 4 Sentinel-2 bands |
| **Indigenous Alignment** | ISRO Cartosat / ResourceSat reference benchmark support |
| **Production Ready** | Hugging Face Spaces deployable — CPU and CUDA compatible |

---

## Project Structure

```
srm-outliers/
├── app/
│   ├── app.py           # Gradio web application (entry point)
│   ├── model.py         # ResidualSR model + Monte Carlo Dropout inference
│   └── utils.py         # Preprocessing, normalization, and visualization
├── src/
│   ├── model.py         # Core model (training reference)
│   ├── dataset.py       # Sentinel-2 dataset pipeline
│   └── __init__.py
├── models/
│   └── residual_sr_x2_best.pt   # Trained 2x checkpoint (gitignored)
├── data/                # Input imagery (gitignored)
├── outputs/             # Inference outputs (gitignored)
├── notebooks/           # Exploration and EDA
├── train.py             # Training script
├── inference.py         # Standalone inference script
├── test_dataset.py      # Dataset validation test
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Install Dependencies

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Run the Web Application (Gradio)

```bash
python app/app.py
```

Open `http://localhost:7860` in your browser.

### 3. Train the Model

```bash
# Quick prototype run (synthetic data):
python train.py --epochs 30 --batch-size 8 --scale 2

# With real Sentinel-2 GeoTIFFs:
python train.py --data-dir data/sentinel2_tiles --scale 2 --epochs 100 --batch-size 16
```

### 4. Standalone Inference (CLI)

```bash
# Synthetic test:
python inference.py --scale 2 --num-mc-samples 15

# Custom GeoTIFF:
python inference.py --input data/sample.tif --model-path models/residual_sr_x2_best.pt
```

---

## Architecture

```
Input (B, 4, H, W) — Sentinel-2 bands: Blue, Green, Red, NIR
         │
         ├─── Bicubic Upsample (global skip) ─────────────────────┐
         │                                                         │
         ▼                                                         │
    Head (Conv → PReLU)                                           │
         │                                                         │
    ┌────▼────────────────────────────────────┐                   │
    │  Residual Block × 4                     │                   │
    │  Conv → PReLU → Spatial Dropout2d → Conv│                   │
    │  ▲─────────────────────────────── Add  ─┘                   │
    └────┬────────────────────────────────────┘                   │
         │                                                         │
    Body Conv + Long Skip                                         │
         │                                                         │
    PixelShuffle (2x) — Sub-pixel convolution                     │
         │                                                         │
    Tail (Conv → output)                                          │
         │                                                         │
         └──────────────────────── Add ◄────────────────────────┘
         │
    Output (B, 4, 2H, 2W) — Super-Resolved Imagery
```

**Monte Carlo Uncertainty:**
- Spatial Dropout remains active at inference time
- T = 15 stochastic forward passes sampled
- Predictive Mean → Super-Resolved Output
- Pixel-wise Standard Deviation → Epistemic Uncertainty Heatmap

---

## Uncertainty Heatmap Interpretation

The uncertainty heatmap uses the **Inferno** perceptually uniform colormap:

| Color | Uncertainty (σ) | Meaning |
|-------|----------------|---------|
| Black / Dark Purple | < 0.008 | High confidence — homogeneous regions (water, flat soil) |
| Magenta / Orange | 0.008 – 0.020 | Moderate — texture gradients, vegetation canopies |
| Bright Yellow / White | > 0.025 | Low reliability — edges, cloud borders, sensor anomalies |

> The Uncertainty Heatmap shows which areas the model is less confident about. Darker/higher values indicate lower reliability.

---

## Deployment on Hugging Face Spaces

1. Create a new Space on [huggingface.co/spaces](https://huggingface.co/spaces) — choose **Gradio SDK**.
2. Push this repository to the Space.
3. Upload `models/residual_sr_x2_best.pt` manually via the Files tab.
4. The Space auto-installs `requirements.txt` and runs `app/app.py`.

---

## Metrics

| Metric | Value |
|--------|-------|
| Reconstruction PSNR | 29.38 dB (synthetic benchmark) |
| Scale Factor | 2× (10m → <4m effective GSD) |
| Inference Latency | < 1s (CUDA) / 2–5s (CPU) |
| MC Dropout Passes | 15 |
| Input Channels | 4 (Blue B02, Green B03, Red B04, NIR B08) |

---

## Team

**The Outliers** | Smart India Hackathon 2026

> Project: BhuVistaar — Sentinel-2 Super Resolution & Uncertainty Quantification
