"""
data_preprocessing.py
---------------------
Handles loading, clipping, and normalizing INSAT HDF5 satellite imagery.
"""

import os
import numpy as np
import h5py


def load_insat_data(hdf5_file: str, dataset_name: str = "IMG_VIS") -> np.ndarray | None:
    """
    Load a dataset from an INSAT HDF5 file.

    Args:
        hdf5_file: Path to the .h5 file.
        dataset_name: Name of the dataset inside the HDF5 file.

    Returns:
        NumPy array of the data, or None if dataset not found.
    """
    with h5py.File(hdf5_file, "r") as f:
        if dataset_name not in f:
            return None
        return f[dataset_name][:]


def process_hdf5(file_path: str, dataset_name: str = "IMG_VIS") -> np.ndarray | None:
    """Process a single HDF5 file and return raw data."""
    return load_insat_data(file_path, dataset_name)


def extract_insat_region(
    insat_data: np.ndarray,
    lat: float,
    lon: float,
    bbox_size: tuple = (150, 150),
) -> np.ndarray:
    """
    Clip and normalise a region of interest from INSAT data.

    Args:
        insat_data: Array of shape (frames, height, width).
        lat: Latitude of the target location.
        lon: Longitude of the target location.
        bbox_size: Height and width (in pixels) of the clipping window.

    Returns:
        Normalised clipped region as float32 array.
    """
    height, width = insat_data.shape[1:]
    row = int(lat * height / 360)
    col = int(lon * width / 360)

    row_start = max(0, row - bbox_size[0] // 2)
    row_end   = min(height, row + bbox_size[0] // 2)
    col_start = max(0, col - bbox_size[1] // 2)
    col_end   = min(width, col + bbox_size[1] // 2)

    region = insat_data[:, row_start:row_end, col_start:col_end]

    if region.size == 0:
        raise ValueError(
            "Extracted region is empty. Check latitude/longitude values or bbox_size."
        )

    mn, mx = np.min(region), np.max(region)
    return (region - mn) / (mx - mn + 1e-6)


def batch_process_h5_to_npy(
    input_dir: str,
    output_dir: str,
    dataset_name: str = "IMG_VIS",
) -> None:
    """
    Convert all .h5 files in a directory to .npy arrays.

    Args:
        input_dir: Directory containing raw HDF5 files.
        output_dir: Directory to save .npy files.
        dataset_name: HDF5 dataset key to extract.
    """
    os.makedirs(output_dir, exist_ok=True)
    h5_files = [f for f in os.listdir(input_dir) if f.endswith(".h5")]

    for filename in h5_files:
        file_path = os.path.join(input_dir, filename)
        data = process_hdf5(file_path, dataset_name)
        if data is not None:
            out_path = os.path.join(output_dir, filename.replace(".h5", ".npy"))
            np.save(out_path, data)
            print(f"Saved: {out_path}")
        else:
            print(f"Skipped (dataset not found): {filename}")


def batch_clip_npy(
    input_dir: str,
    output_dir: str,
    lat: float,
    lon: float,
    bbox_size: tuple = (150, 150),
) -> None:
    """
    Clip and normalise all .npy files and save results.

    Args:
        input_dir: Directory of raw .npy files.
        output_dir: Directory to save clipped .npy files.
        lat: Target latitude.
        lon: Target longitude.
        bbox_size: Clipping window size.
    """
    os.makedirs(output_dir, exist_ok=True)
    npy_files = [f for f in os.listdir(input_dir) if f.endswith(".npy")]

    for npy_file in npy_files:
        npy_path = os.path.join(input_dir, npy_file)
        data = np.load(npy_path)
        clipped = extract_insat_region(data, lat, lon, bbox_size)
        num_frames, h, w = clipped.shape
        clipped = clipped.reshape((num_frames, h, w, 1))
        out_path = os.path.join(output_dir, npy_file)
        np.save(out_path, clipped)
        print(f"Saved clipped: {out_path}")
