# Data

This folder is for storing datasets. Due to size constraints, raw data is **not included** in this repository.

## How to Obtain the Data

### 1. INSAT Satellite Imagery (HDF5)
- Visit [MOSDAC (Meteorological & Oceanographic Satellite Data Archival Centre)](https://www.mosdac.gov.in/)
- Register for a free account
- Download **INSAT-3D/3DR visible band (VIS)** images in HDF5 format
- Dataset name inside HDF5: `IMG_VIS`
- Temporal resolution: **30 minutes**
- Place files in: `MyDrive/solar_project/2019_Data/` (for Colab) or update path in notebook

### 2. Ground Truth GHI Data
- Source: Ground-based pyranometer at **Tirupati, India** (13.627°N, 79.397°E)
- Format: Excel (`.xlsx`) with columns `Date` and `Ground_data`
- File: `tirupati_ground_data.xlsx`

### 3. Clear-Sky GHI
- Generated programmatically using `pvlib` — no download needed
- Model: **Reduced Solis**
- Location: Tirupati, India

## Expected Folder Structure (Google Drive)

```
MyDrive/solar_project/
├── 2019_Data/                    # Raw INSAT .h5 files
│   ├── 3DIMG_01SEP2019_0000_VIS.h5
│   └── ...
└── tirupati_ground_data.xlsx     # Ground truth GHI
```
