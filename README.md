# ☀️ Attention-Enhanced Spatio-Temporal Deep Learning Model for Solar Irradiance Forecasting

> A hybrid deep learning framework for short-term Global Horizontal Irradiance (GHI) forecasting using INSAT satellite imagery and physics-based clear-sky modelling.

---

## 📄 Paper

**"Attention-Enhanced Spatio-Temporal Deep Learning Model for Solar Irradiance Forecasting"**

---

## 🧠 Overview

This project proposes a novel hybrid forecasting framework that combines:
- A **ConvLSTM Autoencoder** for spatiotemporal cloud pattern learning
- A **Multi-Head Attention** mechanism for temporal dependency weighting
- The **Reduced Solis clear-sky model** (via `pvlib`) for physics-based GHI baseline

The model takes **6 sequential INSAT satellite frames (3 hours of history)** as input and predicts the **next 4 GHI values (2-hour horizon)** in 30-minute steps.

---

## 🏗️ Architecture

```
INSAT Satellite Images (HDF5)
        ↓
   Preprocessing & Clipping (256×256 AOI around Tirupati)
        ↓
   Conv2D Autoencoder  ←→  Cloud Mask Generation
        ↓
   Cloud Index (CI) Computation
        ↓
   ConvLSTM + LayerNorm + MultiHeadAttention
        ↓
   Dense Output (4-step GHI forecast)
        ↓
   GHI_estimated = GHI_clearsky × Cloud_Index
```

---

## 📊 Results

| Metric | Score |
|--------|-------|
| MAE    | 110.98 W/m² |
| RMSE   | 142.38 W/m² |
| R²     | 0.74 |

---

## 📁 Project Structure

```
solar-ghi-forecasting/
├── notebooks/
│   └── solar_ghi_forecasting.ipynb   # Main Colab notebook (full pipeline)
├── src/
│   ├── data_preprocessing.py         # INSAT HDF5 loading, clipping, normalization
│   ├── autoencoder.py                # Conv2D Autoencoder + cloud mask generation
│   ├── model.py                      # ConvLSTM + MultiHeadAttention model
│   ├── features.py                   # Cloud index, clear-sky GHI, sequence creation
│   └── evaluate.py                   # Metrics, plots, comparison table
├── data/
│   └── README.md                     # Data download instructions
├── results/
│   └── README.md                     # How to interpret results
├── docs/
│   └── paper.pdf                     # Research paper
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/solar-ghi-forecasting.git
cd solar-ghi-forecasting
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare Data

- Download INSAT visible band HDF5 files from [MOSDAC](https://www.mosdac.gov.in/)
- Place ground-truth GHI data (Excel) at `data/tirupati_ground_data.xlsx`
- Update file paths in the notebook if running locally

### 4. Run the Notebook

Open `notebooks/solar_ghi_forecasting.ipynb` in **Google Colab** or Jupyter.  
Mount your Google Drive with the data folder structured as:

```
MyDrive/solar_project/
├── 2019_Data/          # Raw INSAT HDF5 files
├── tirupati_ground_data.xlsx
```

---

## 🔧 Key Components

### Data Sources
- **INSAT VIS Band** — 30-min temporal resolution, HDF5 format (ISRO/MOSDAC)
- **Ground Truth GHI** — Pyranometer at Tirupati, India (13.627°N, 79.397°E)
- **Clear-sky GHI** — Reduced Solis model via `pvlib`

### Model Highlights
- ConvLSTM at autoencoder bottleneck captures **temporal cloud evolution**
- **Multi-Head Attention** (4 heads, key_dim=16) learns time-step importance
- **Layer Normalization** for training stability
- **Huber loss** for robustness to GHI outliers
- **Sliding window**: 6 input steps → 4 predicted steps

---

## 📦 Requirements

See [`requirements.txt`](requirements.txt)

---

## 📌 Citation

If you use this work, please cite:

```
"Attention-Enhanced Spatio-Temporal Deep Learning Model for Solar Irradiance Forecasting." 2024.
```



