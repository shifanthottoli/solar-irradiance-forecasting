"""
model.py
--------
ConvLSTM + Multi-Head Attention model for multi-step GHI forecasting.
"""

import tensorflow as tf
from tensorflow.keras import Model, Input, callbacks
from tensorflow.keras.layers import (
    ConvLSTM2D,
    LayerNormalization,
    MultiHeadAttention,
    Reshape,
    Flatten,
    Dense,
    Dropout,
)


def build_convlstm_attention_model(
    seq_len: int,
    num_features: int,
    pred_len: int,
    lstm_units: int = 64,
    num_heads: int = 4,
    key_dim: int = 16,
    dense_units: int = 128,
    dropout_rate: float = 0.3,
) -> Model:
    """
    Build the ConvLSTM + Multi-Head Attention forecasting model.

    Architecture:
        Input (seq_len, 1, 1, num_features)
            → ConvLSTM2D (return_sequences=True)
            → LayerNormalization
            → ConvLSTM2D (return_sequences=True)
            → LayerNormalization
            → Reshape → MultiHeadAttention
            → Flatten → Dense → Dropout
            → Dense (pred_len outputs)

    Args:
        seq_len: Number of input time steps.
        num_features: Number of features per time step.
        pred_len: Number of GHI values to predict.
        lstm_units: Number of ConvLSTM filters.
        num_heads: Number of attention heads.
        key_dim: Key dimension for each attention head.
        dense_units: Units in the intermediate Dense layer.
        dropout_rate: Dropout probability.

    Returns:
        Compiled Keras Model.
    """
    seq_input = Input(shape=(seq_len, 1, 1, num_features), name="sequence_input")

    x = ConvLSTM2D(lstm_units, (1, 1), activation="tanh", return_sequences=True)(seq_input)
    x = LayerNormalization()(x)
    x = ConvLSTM2D(lstm_units, (1, 1), activation="tanh", return_sequences=True)(x)
    x = LayerNormalization()(x)

    # Reshape to (batch, seq_len, lstm_units) for attention
    x = Reshape((seq_len, lstm_units))(x)
    x = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(x, x)

    x = Flatten()(x)
    x = Dense(dense_units, activation="relu")(x)
    x = Dropout(dropout_rate)(x)
    output = Dense(pred_len, name="ghi_forecast")(x)

    model = Model(seq_input, output, name="convlstm_attention_ghi")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="huber",
        metrics=["mae"],
    )
    return model


def get_training_callbacks(
    checkpoint_path: str = "best_model.h5",
    patience_early_stop: int = 15,
    patience_lr: int = 7,
    lr_factor: float = 0.3,
) -> list:
    """
    Return standard training callbacks.

    Args:
        checkpoint_path: Path to save the best model.
        patience_early_stop: Epochs to wait before early stopping.
        patience_lr: Epochs to wait before reducing learning rate.
        lr_factor: Factor to reduce LR by on plateau.

    Returns:
        List of Keras callbacks.
    """
    return [
        callbacks.EarlyStopping(patience=patience_early_stop, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(patience=patience_lr, factor=lr_factor, verbose=1),
        callbacks.ModelCheckpoint(checkpoint_path, save_best_only=True, monitor="val_loss"),
    ]
