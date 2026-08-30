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
        """Sample from the latent distribution."""
        mu, log_var = args
        epsilon = keras.backend.random_normal(
            shape=keras.backend.shape(mu)
        )
        return mu + keras.backend.exp(0.5 * log_var) * epsilon

    latent = keras.layers.Lambda(
        sampling
    )([mean, log_variance])

    encoder = keras.Model(
        encoder_input,
        [latent, mean, log_variance]
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
        decoder_input,
        decoder_output
    )

    latent_sample, mean_sample, log_var_sample = encoder(
        encoder_input
    )

    auto_output = decoder(latent_sample)

    auto = keras.Model(
        encoder_input,
        auto_output
    )

    reconstruction_loss = keras.losses.binary_crossentropy(
        encoder_input,
        auto_output
    )
    reconstruction_loss *= input_dims

    kl_loss = -0.5 * keras.backend.sum(
        1 +
        log_var_sample -
        keras.backend.square(mean_sample) -
        keras.backend.exp(log_var_sample),
        axis=-1
    )

    vae_loss = keras.backend.mean(
        reconstruction_loss + kl_loss
    )

    auto.add_loss(vae_loss)

    auto.compile(
        optimizer=keras.optimizers.Adam()
    )

    return encoder, decoder, auto
