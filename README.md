# srm-outliers: Sentinel-2 Super Resolution

A prototype pipeline for super-resolution and anomaly/outlier detection in Sentinel-2 multispectral satellite imagery.

## Overview

The goal of this project is to enhance the spatial resolution of Sentinel-2 multispectral bands (e.g., bringing 20m/60m bands to 10m resolution or performing 2x/4x super-resolution) while handling outliers and edge cases in remote sensing scenes.

## Project Structure

```text
srm-outliers/
├── app/            # Interactive demo or web interface (e.g. Streamlit)
├── data/           # Dataset storage (raw and processed, gitignored)
├── models/         # Trained model checkpoints and weights (gitignored)
├── notebooks/      # Exploratory notebooks for EDA and evaluation
├── outputs/        # Inferred images, visual comparisons, and metrics (gitignored)
├── src/            # Source code and core modules
│   └── __init__.py
├── .gitignore      # Git ignore patterns
├── README.md       # Project documentation
└── requirements.txt# Project dependencies
```

## Getting Started

### 1. Installation

Set up a Python virtual environment and install dependencies:

```bash
python -m venv .venv
# On Linux/macOS:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Usage

- **Exploration & Prototyping**: Run `jupyter lab` and open notebooks in `notebooks/`.
- **Source Modules**: Build models and data loaders under `src/`.
- **Demo Application**: Launch the application via `streamlit run app/app.py`.
