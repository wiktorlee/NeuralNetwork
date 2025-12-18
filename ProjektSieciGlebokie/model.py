from typing import Tuple

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_model(
    input_shape: Tuple[int, int, int] = (128, 128, 3),
    num_classes: int = 195,
    base_filters: int = 32,
) -> keras.Model:
    """
    Tworzy lekki model CNN do klasyfikacji 195 flag.
    """
    model = keras.Sequential(name="flag_classifier")

    model.add(keras.Input(shape=input_shape))

    filters = base_filters
    for block_idx in range(4):
        model.add(layers.Conv2D(filters, kernel_size=3, padding="same"))
        model.add(layers.BatchNormalization())
        model.add(layers.ReLU())
        model.add(layers.MaxPooling2D(pool_size=2))
        if block_idx >= 2:
            model.add(layers.Dropout(0.25))
        filters *= 2

    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(256, activation="relu"))
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(num_classes, activation="softmax"))

    return model


if __name__ == "__main__":
    model = build_model()
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    dummy_batch = tf.zeros((1, 128, 128, 3))
    logits = model(dummy_batch)
    print("Testowy forward pass:", logits.shape)





