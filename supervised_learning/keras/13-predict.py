#!/usr/bin/env python3
"""Make predictions with a Keras model."""
import tensorflow.keras as K


def predict(network, data, verbose=False):
    """Predict with the model.

    Args:
        network: the model to make the prediction with.
        data: the input data to predict on.
        verbose: whether to print the prediction output.

    Returns:
        The prediction probabilities for ``data``.
    """
    return network.predict(data, verbose=verbose)
