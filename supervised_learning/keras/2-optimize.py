#!/usr/bin/env python3
"""Set up Adam optimization for a Keras model."""
import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """Compile the model with Adam and categorical crossentropy.

    Args:
        network: the Keras model to optimize.
        alpha: the learning rate.
        beta1: the first Adam moment weight.
        beta2: the second Adam moment weight.

    Returns:
        None
    """
    network.compile(
        optimizer=K.optimizers.Adam(learning_rate=alpha, beta_1=beta1,
                                    beta_2=beta2),
        loss="categorical_crossentropy", metrics=["accuracy"])
