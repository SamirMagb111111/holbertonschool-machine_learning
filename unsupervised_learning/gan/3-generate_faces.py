#!/usr/bin/env python3
"""Convolutional generator and discriminator for face generation."""
from tensorflow import keras


def convolutional_GenDiscr():
    """Build a convolutional generator and discriminator.

    The generator maps a latent vector of shape ``(16,)`` to a
    ``(16, 16, 1)`` image through a dense projection followed by three
    upsampling / convolution blocks. The discriminator maps a
    ``(16, 16, 1)`` image to a single scalar through four convolution /
    pooling blocks and a dense layer.

    Returns:
        (generator, discriminator): the two keras models.
    """

    def get_generator():
        """Return the convolutional generator model."""
        inputs = keras.Input(shape=(16,))
        x = keras.layers.Dense(2048, activation="tanh")(inputs)
        x = keras.layers.Reshape((2, 2, 512))(x)

        x = keras.layers.UpSampling2D((2, 2))(x)
        x = keras.layers.Conv2D(64, (3, 3), padding="same")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("tanh")(x)

        x = keras.layers.UpSampling2D((2, 2))(x)
        x = keras.layers.Conv2D(16, (3, 3), padding="same")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("tanh")(x)

        x = keras.layers.UpSampling2D((2, 2))(x)
        x = keras.layers.Conv2D(1, (3, 3), padding="same")(x)
        x = keras.layers.BatchNormalization()(x)
        outputs = keras.layers.Activation("tanh")(x)

        return keras.Model(inputs, outputs, name="generator")

    def get_discriminator():
        """Return the convolutional discriminator model."""
        inputs = keras.Input(shape=(16, 16, 1))

        x = keras.layers.Conv2D(32, (3, 3), padding="same")(inputs)
        x = keras.layers.MaxPooling2D((2, 2))(x)
        x = keras.layers.Activation("tanh")(x)

        x = keras.layers.Conv2D(64, (3, 3), padding="same")(x)
        x = keras.layers.MaxPooling2D((2, 2))(x)
        x = keras.layers.Activation("tanh")(x)

        x = keras.layers.Conv2D(128, (3, 3), padding="same")(x)
        x = keras.layers.MaxPooling2D((2, 2))(x)
        x = keras.layers.Activation("tanh")(x)

        x = keras.layers.Conv2D(256, (3, 3), padding="same")(x)
        x = keras.layers.MaxPooling2D((2, 2))(x)
        x = keras.layers.Activation("tanh")(x)

        x = keras.layers.Flatten()(x)
        outputs = keras.layers.Dense(1, activation="tanh")(x)

        return keras.Model(inputs, outputs, name="discriminator")

    return get_generator(), get_discriminator()
