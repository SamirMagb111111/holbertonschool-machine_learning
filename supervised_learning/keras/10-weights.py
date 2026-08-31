#!/usr/bin/env python3
"""Save and load a Keras model's weights."""
import tensorflow.keras as K


def save_weights(network, filename, save_format="keras"):
    """Save the model weights to ``filename``.

    Args:
        network: the model whose weights are saved.
        filename: path the weights should be saved to.
        save_format: kept for API compatibility; Keras infers the format
            from ``filename``.

    Returns:
        None
    """
    network.save_weights(filename)


def load_weights(network, filename):
    """Load weights from ``filename`` into ``network``.

    Args:
        network: the model to load the weights into.
        filename: path of the weights to load.

    Returns:
        None
    """
    network.load_weights(filename)
