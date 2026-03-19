"""
evaluate.py
-----------
Metrics computation, result visualisation, and output CSV export.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    """
    Compute MAE, RMSE, and R² for flattened prediction arrays.

    Args:
        y_true: Ground truth GHI values (W/m²).
        y_pred: Predicted GHI values (W/m²).

    Returns:
        Dictionary with keys 'MAE', 'RMSE', 'R2'.
    """
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)

    print("\n📊 Evaluation Metrics:")
    print(f"  MAE  : {mae:.2f} W/m²")
    print(f"  RMSE : {rmse:.2f} W/m²")
    print(f"  R²   : {r2:.4f}")

    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def plot_loss_curves(history) -> None:
    """Plot training and validation loss curves."""
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"],     label="Training Loss",   linewidth=2)
    plt.plot(history.history["val_loss"], label="Validation Loss", linewidth=2)
    plt.xlabel("Epochs")
    plt.ylabel("Loss (Huber)")
    plt.title("Model Loss Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_forecast_samples(
    y_val_denorm: np.ndarray,
    y_pred_denorm: np.ndarray,
    clear_ghi_val: np.ndarray,
    time_val: np.ndarray,
    masks_val: np.ndarray,
    pred_len: int,
    n_samples: int = 3,
) -> None:
    """
    Plot cloud mask + GHI forecast for the best-performing samples.

    Args:
        y_val_denorm: Actual GHI (N, pred_len).
        y_pred_denorm: Predicted GHI (N, pred_len).
        clear_ghi_val: Clear-sky GHI flattened values.
        time_val: Timestamps (N, pred_len).
        masks_val: Cloud masks (N, H, W).
        pred_len: Forecast horizon.
        n_samples: Number of samples to plot.
    """
    mse_per_sample = np.mean((y_val_denorm - y_pred_denorm) ** 2, axis=1)
    top_indices = np.argsort(mse_per_sample)[:n_samples]

    for idx in top_indices:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].imshow(masks_val[idx], cmap="gray")
        axes[0].set_title(f"Cloud Mask @ {time_val[idx][0].strftime('%Y-%m-%d %H:%M')}")
        axes[0].axis("off")

        axes[1].plot(range(pred_len), y_val_denorm[idx],  label="Actual GHI",    marker="o")
        axes[1].plot(range(pred_len), y_pred_denorm[idx], label="Predicted GHI", marker="x")
        axes[1].plot(
            range(pred_len),
            clear_ghi_val[idx * pred_len : (idx + 1) * pred_len],
            label="Clear-sky GHI",
            linestyle="--",
        )
        axes[1].set_title(f"GHI Forecast (Next {pred_len} steps)")
        axes[1].set_xlabel("Future time steps (30-min each)")
        axes[1].set_ylabel("GHI (W/m²)")
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        plt.show()


def plot_full_month_forecast(
    y_val_flat: np.ndarray,
    y_pred_flat: np.ndarray,
    clear_ghi_flat: np.ndarray,
    time_flat,
) -> None:
    """
    Plot the full time-series of actual vs predicted vs clear-sky GHI.

    Args:
        y_val_flat: Flattened actual GHI.
        y_pred_flat: Flattened predicted GHI.
        clear_ghi_flat: Flattened clear-sky GHI.
        time_flat: Corresponding flattened timestamps.
    """
    plot_df = pd.DataFrame({
        "timestamp":    time_flat,
        "actual_GHI":   y_val_flat,
        "predicted_GHI": y_pred_flat,
        "clear_sky_GHI": clear_ghi_flat,
    })
    plot_df["timestamp"] = plot_df["timestamp"].dt.tz_convert("Asia/Kolkata")
    plot_df = plot_df.sort_values("timestamp")

    plt.figure(figsize=(18, 6))
    plt.plot(plot_df["timestamp"], plot_df["actual_GHI"],    label="Actual GHI",    linewidth=2, color="blue")
    plt.plot(plot_df["timestamp"], plot_df["predicted_GHI"], label="Predicted GHI", linewidth=2, color="orange")
    plt.plot(plot_df["timestamp"], plot_df["clear_sky_GHI"], label="Clear-sky GHI", linewidth=1.5, color="green", linestyle="--")
    plt.xlabel("Timestamp (IST)")
    plt.ylabel("GHI (W/m²)")
    plt.title("Full-Month GHI Forecast vs Actual vs Clear-Sky (Tirupati, September 2019)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def save_comparison_csv(
    time_flat,
    y_val_flat: np.ndarray,
    y_pred_flat: np.ndarray,
    clear_ghi_flat: np.ndarray,
    output_path: str = "results/ghi_comparison_tirupati_september.csv",
) -> pd.DataFrame:
    """
    Save a comparison table of actual, predicted, and clear-sky GHI to CSV.

    Args:
        time_flat: Flattened timestamps.
        y_val_flat: Actual GHI.
        y_pred_flat: Predicted GHI.
        clear_ghi_flat: Clear-sky GHI.
        output_path: File path for the output CSV.

    Returns:
        The saved DataFrame.
    """
    df = pd.DataFrame({
        "timestamp":    time_flat,
        "actual_GHI":   y_val_flat,
        "predicted_GHI": y_pred_flat,
        "clear_sky_GHI": clear_ghi_flat,
    })
    df["timestamp"] = df["timestamp"].dt.tz_convert("Asia/Kolkata")
    df = df.sort_values("timestamp")
    df.to_csv(output_path, index=False)
    print(f"Saved comparison CSV: {output_path}")
    return df
