# Results

After running the full pipeline notebook, the following output files are generated:

## Output Files

| File | Description |
|------|-------------|
| `best_model.h5` | Best saved Keras model (via ModelCheckpoint) |
| `ghi_comparison_tirupati_september.csv` | Timestamped table of Actual vs Predicted vs Clear-sky GHI |

## Evaluation Metrics (Test Set)

| Metric | Value |
|--------|-------|
| MAE    | 110.98 W/m² |
| RMSE   | 142.38 W/m² |
| R²     | 0.74 |

## Plots Generated

1. **Training vs Validation Loss** — Convergence curve over 80 epochs
2. **2-Hour GHI Forecast** — Predicted vs Actual vs Clear-sky for top samples
3. **Full-Month GHI Forecast** — Time-series comparison across September 2019
4. **Cloud Mask Overlay** — Autoencoder reconstruction error visualized on satellite frames
