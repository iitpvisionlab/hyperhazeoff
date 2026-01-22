from __future__ import annotations

from typing import Callable, Tuple

import tensorflow as tf
from tensorflow.keras import Model, layers


def make_lr_metric(
    optimizer: tf.keras.optimizers.Optimizer,
) -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
    """Return a metric function that reports the current learning rate."""

    def lr_metric(_: tf.Tensor, __: tf.Tensor) -> tf.Tensor:
        # Handles both constant LR and schedules.
        return tf.convert_to_tensor(optimizer.learning_rate)

    lr_metric.__name__ = "learning_rate"
    return lr_metric


def create_model(
    input_shape: Tuple[int, int, int],
    num_classes: int,
    optimizer: tf.keras.optimizers.Optimizer,
) -> Model:
    """Create and compile a simple CNN classifier."""
    if num_classes < 2:
        raise ValueError(f"num_classes must be >= 2, got {num_classes}")

    model = tf.keras.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(256, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(512, activation="relu"),
            layers.Dense(num_classes, activation="softmax"),
        ],
        name="cnn_classifier",
    )

    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            make_lr_metric(optimizer),
        ],
    )
    return model
