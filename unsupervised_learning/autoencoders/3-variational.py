#!/usr/bin/env python3
"""Variational autoencoder module."""

import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """Create a variational autoencoder."""
    encoder_input = keras.Input(shape=(input_dims,))
    x = encoder_input

    for nodes in hidden_layers:
        x = keras.layers.Dense(
            nodes,
            activation='relu'
        )(x)

    mean = keras.layers.Dense(
        latent_dims,
        activation=None
    )(x)

    log_variance = keras.layers.Dense(
        latent_dims,
        activation=None
    )(x)

    def sampling(args):
        """Sample a point from the latent distribution."""
        mu, log_var = args

        epsilon = keras.backend.random_normal(
            shape=keras.backend.shape(mu)
        )

        return (
            mu +
            keras.backend.exp(log_var / 2) * epsilon
        )

    latent = keras.layers.Lambda(
        sampling
    )([mean, log_variance])

    encoder = keras.Model(
        inputs=encoder_input,
        outputs=[latent, mean, log_variance]
    )

    decoder_input = keras.Input(shape=(latent_dims,))
    x = decoder_input

    for nodes in reversed(hidden_layers):
        x = keras.layers.Dense(
            nodes,
            activation='relu'
        )(x)

    decoder_output = keras.layers.Dense(
        input_dims,
        activation='sigmoid'
    )(x)

    decoder = keras.Model(
        inputs=decoder_input,
        outputs=decoder_output
    )

    auto_output = decoder(latent)

    auto = keras.Model(
        inputs=encoder_input,
        outputs=auto_output
    )

    def vae_loss(y_true, y_pred):
        """Calculate reconstruction and KL divergence loss."""
        reconstruction_loss = keras.losses.binary_crossentropy(
            y_true,
            y_pred
        )
        reconstruction_loss *= input_dims

        kl_loss = -0.5 * keras.backend.sum(
            1 +
            log_variance -
            keras.backend.square(mean) -
            keras.backend.exp(log_variance),
            axis=-1
        )

        return keras.backend.mean(
            reconstruction_loss + kl_loss
        )

    auto.compile(
        optimizer='adam',
        loss=vae_loss
    )

    return encoder, decoder, auto
