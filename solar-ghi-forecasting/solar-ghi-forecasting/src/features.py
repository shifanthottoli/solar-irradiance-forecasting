"""
features.py
-----------
Cloud index computation, clear-sky GHI generation, and
sliding-window sequence creation for model training.
"""

import numpy as np
import pandas as pd
from datetime import datetime
from pvlib.location import Location
from sklearn.preprocessing import MinMaxScaler


def compute_clearsky_ghi(
    lat: float,
    lon: float,
    altitude: float,
    start: str,
    end: str,
    freq: str = "1min",
    model: str = "simplified_solis",
) -> pd.DataFrame:
    """
    Generate a clear-sky GHI time-series using pvlib.

    Args:
        lat: Latitude of the site.
        lon: Longitude of the site.
        altitude: Altitude in metres.
        start: Start datetime string (e.g. '2019-09-01 00:00:00').
        end: End datetime string.
        freq: Pandas frequency string for the time index.
        model: pvlib clear-sky model name.

    Returns:
        DataFrame with UTC-indexed clear-sky irradiance columns.
    """
    date_range = pd.date_range(start=start, end=end, freq=freq)
    location = Location(lat, lon, altitude=altitude)
    clearsky = location.get_clearsky(date_range, model=model)
    clearsky.index = clearsky.index.tz_localize("UTC")
    return clearsky


def compute_cloud_index(
    cloud_mask_values: np.ndarray,
    cloud_mask_times: pd.DatetimeIndex,
    rolling_window: str = "15min",
    percentile_low: int = 5,
    percentile_high: int = 95,
) -> np.ndarray:
    """
    Convert raw cloud mask mean values to a smoothed Cloud Index (CI).

    CI = 1 - normalised(cloud_fraction), clipped to [0, 1].
    A rolling mean is applied for temporal smoothing.

    Args:
        cloud_mask_values: Per-frame mean cloud mask values.
        cloud_mask_times: Corresponding UTC timestamps.
        rolling_window: Pandas rolling window string.
        percentile_low: Lower percentile for normalisation.
        percentile_high: Upper percentile for normalisation.

    Returns:
        Smoothed cloud index array.
    """
    nmin = np.percentile(cloud_mask_values, percentile_low)
    nmax = np.percentile(cloud_mask_values, percentile_high)
    ci = 1 - (cloud_mask_values - nmin) / (nmax - nmin)
    ci = np.clip(ci, 0, 1)
    ci_series = pd.Series(ci, index=cloud_mask_times).rolling(rolling_window).mean().bfill()
    return ci_series.values


def build_feature_matrix(
    actual_ghi: np.ndarray,
    clear_ghi: np.ndarray,
    cloud_index: np.ndarray,
    timestamps: np.ndarray,
) -> tuple[np.ndarray, MinMaxScaler]:
    """
    Normalise GHI signals and stack with time-based cyclic features.

    Feature columns: [ghi_norm, clear_ghi_norm, cloud_index, sin_hour, sin_minute]

    Args:
        actual_ghi: Ground truth GHI values.
        clear_ghi: Clear-sky GHI values.
        cloud_index: Smoothed cloud index values.
        timestamps: Array of pandas Timestamps.

    Returns:
        (feature_matrix, fitted_scaler)
    """
    scaler = MinMaxScaler()
    all_vals = np.concatenate([actual_ghi.reshape(-1, 1), clear_ghi.reshape(-1, 1)], axis=0)
    scaler.fit(all_vals)

    ghi_norm       = scaler.transform(actual_ghi.reshape(-1, 1)).flatten()
    clear_ghi_norm = scaler.transform(clear_ghi.reshape(-1, 1)).flatten()

    hour   = np.array([np.sin(2 * np.pi * ts.hour   / 24) for ts in timestamps])
    minute = np.array([np.sin(2 * np.pi * ts.minute / 60) for ts in timestamps])

    features = np.stack([ghi_norm, clear_ghi_norm, cloud_index, hour, minute], axis=1)
    return features, scaler


def create_sequences(
    features: np.ndarray,
    ghi_norm: np.ndarray,
    cloud_masks: np.ndarray,
    timestamps: np.ndarray,
    seq_len: int = 6,
    pred_len: int = 4,
) -> tuple:
    """
    Build sliding-window sequences for ConvLSTM input.

    Args:
        features: Feature matrix of shape (T, num_features).
        ghi_norm: Normalised GHI values.
        cloud_masks: Cloud mask images for each timestep.
        timestamps: Array of timestep timestamps.
        seq_len: Number of input time steps (lookback window).
        pred_len: Number of output time steps (forecast horizon).

    Returns:
        (X, y, time_y, masks_seq) as NumPy arrays.
        X shape: (N, seq_len, 1, 1, num_features)  — ConvLSTM-compatible
        y shape: (N, pred_len)
    """
    X, y, time_y, masks_seq = [], [], [], []

    for i in range(len(features) - seq_len - pred_len + 1):
        output_seq = ghi_norm[i + seq_len : i + seq_len + pred_len]
        if np.isnan(output_seq).any():
            continue
        X.append(features[i : i + seq_len])
        y.append(output_seq)
        time_y.append(timestamps[i + seq_len : i + seq_len + pred_len])
        masks_seq.append(cloud_masks[i + seq_len - 1])

    num_features = features.shape[1]
    X         = np.array(X, dtype="float32").reshape(-1, seq_len, 1, 1, num_features)
    y         = np.array(y, dtype="float32")
    time_y    = np.array(time_y)
    masks_seq = np.array(masks_seq, dtype="float32")

    return X, y, time_y, masks_seq
