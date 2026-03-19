"""
autoencoder.py
--------------
Conv2D Autoencoder for cloud mask generation via reconstruction error.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, Input


def build_autoencoder(input_shape: tuple, latent_dim: int = 64) -> Model:
    """
    Build a Conv2D Autoencoder for cloud pattern learning.

    Architecture:
        Encoder: Conv2D (64→32→16) + Flatten + Dense(latent_dim)
        Decoder: Dense + Reshape + Conv2DTranspose (32→64→1)

    Args:
        input_shape: Shape of a single input image, e.g. (H, W, 1).
        latent_dim: Size of the bottleneck latent space.

    Returns:
        Compiled Keras Model.
    """
    inputs = Input(shape=input_shape)

    # Encoder
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(inputs)
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = layers.Conv2D(16, (3, 3), activation="relu", padding="same")(x)
    encoded = layers.Flatten()(x)
    latent = layers.Dense(latent_dim, activation="relu")(encoded)

    # Decoder
    spatial_size = input_shape[0] * input_shape[1] * 16
    x = layers.Dense(spatial_size, activation="relu")(latent)
    x = layers.Reshape((input_shape[0], input_shape[1], 16))(x)
    x = layers.Conv2DTranspose(32, (3, 3), activation="relu", padding="same")(x)
    x = layers.Conv2DTranspose(64, (3, 3), activation="relu", padding="same")(x)
    decoded = layers.Conv2DTranspose(1, (3, 3), activation="sigmoid", padding="same")(x)

    autoencoder = Model(inputs, decoded, name="conv2d_autoencoder")
    autoencoder.compile(optimizer="adam", loss="mse")
    return autoencoder


def get_cloud_mask(
    original: np.ndarray,
    reconstructed: np.ndarray,
    threshold: float = 0.08,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a binary cloud mask from the reconstruction error.

    Pixels with high reconstruction error correspond to clouds
    (unusual patterns the autoencoder cannot reconstruct well).

    Args:
        original: Original image array.
        reconstructed: Autoencoder-reconstructed image array.
        threshold: Normalised error threshold above which a pixel is marked as cloud.

    Returns:
        (mask, norm_error): Binary cloud mask and normalised error map.
    """
    error_map = np.abs(original - reconstructed)
    norm_error = (error_map - error_map.min()) / (error_map.max() - error_map.min() + 1e-6)
    mask = (norm_error > threshold).astype(np.uint8)
    return mask.squeeze(), norm_error.squeeze()


def generate_and_save_masks(
    autoencoder: Model,
    X: np.ndarray,
    image_files: list,
    mask_output_dir: str,
    threshold: float = 0.08,
) -> None:
    """
    Run the autoencoder over all images, generate masks, and save them.

    Args:
        autoencoder: Trained autoencoder model.
        X: Image array of shape (N, H, W, 1).
        image_files: Corresponding filenames for each image.
        mask_output_dir: Directory to save cloud mask .npy files.
        threshold: Cloud detection threshold.
    """
    import os
    import matplotlib.pyplot as plt

    os.makedirs(mask_output_dir, exist_ok=True)

    for i, img in enumerate(X):
        recon = autoencoder.predict(img[np.newaxis], verbose=0)[0]
        mask, err_map = get_cloud_mask(img, recon, threshold)

        mask_fname = f"cloudmask_{image_files[i].replace('.npy', '')}.npy"
        mask_path = os.path.join(mask_output_dir, mask_fname)
        np.save(mask_path, mask)

        if i < 3:
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(img.squeeze(), cmap="gray"); axes[0].set_title("Original"); axes[0].axis("off")
            axes[1].imshow(mask, cmap="gray");           axes[1].set_title("Cloud Mask"); axes[1].axis("off")
            overlay = np.stack([img.squeeze()] * 3, axis=-1)
            overlay[mask == 1] = [1, 0, 0]
            axes[2].imshow(overlay);                     axes[2].set_title("Overlay"); axes[2].axis("off")
            plt.tight_layout(); plt.show()

        print(f"Saved mask: {mask_path}")
