#!/usr/bin/env python3
"""Save and load an entire Keras model."""
import tensorflow.keras as K


def save_model(network, filename):
    """Save the whole model to ``filename``.

    Args:
        network: the model to save.
        filename: path the model should be saved to.

    Returns:
        None
    """
    network.save(filename)


def load_model(filename):
    """Load a whole model from ``filename``.

    Args:
        filename: path of the model to load.

    Returns:
        The loaded model.
    """
    return K.models.load_model(filename)
