#!/usr/bin/env python3
"""Convolutional autoencoder module."""

import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """Create a convolutional autoencoder."""
    encoder_input = keras.Input(shape=input_dims)
    x = encoder_input

    for fil in filters:
        x = keras.layers.Conv2D(
            filters=fil,
            kernel_size=(3, 3),
            padding='same',
            activation='relu'
        )(x)
        x = keras.layers.MaxPooling2D(
            pool_size=(2, 2),
            padding='same'
        )(x)

    encoder = keras.Model(
        inputs=encoder_input,
        outputs=x
    )

    decoder_input = keras.Input(shape=latent_dims)
    x = decoder_input

    reversed_filters = filters[::-1]

    for i, fil in enumerate(reversed_filters):
        if i == len(reversed_filters) - 1:
            padding = 'valid'
        else:
            padding = 'same'

        x = keras.layers.Conv2D(
            filters=fil,
            kernel_size=(3, 3),
            padding=padding,
            activation='relu'
        )(x)

        x = keras.layers.UpSampling2D(
            size=(2, 2)
        )(x)

    decoder_output = keras.layers.Conv2D(
        filters=input_dims[-1],
        kernel_size=(3, 3),
        padding='same',
        activation='sigmoid'
    )(x)

    decoder = keras.Model(
        inputs=decoder_input,
        outputs=decoder_output
    )

    auto_output = decoder(encoder(encoder_input))

    auto = keras.Model(
        inputs=encoder_input,
        outputs=auto_output
    )

    auto.compile(
        optimizer='adam',
        loss='binary_crossentropy'
    )

    return encoder, decoder, auto
