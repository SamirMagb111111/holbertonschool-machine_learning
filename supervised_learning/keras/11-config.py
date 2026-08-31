#!/usr/bin/env python3
"""Save and load a Keras model's configuration in JSON."""
import tensorflow.keras as K


def save_config(network, filename):
    """Write the model's JSON configuration to ``filename``.

    Args:
        network: the model whose configuration is saved.
        filename: path the JSON configuration should be saved to.

    Returns:
        None
    """
    with open(filename, "w") as config:
        config.write(network.to_json())


def load_config(filename):
    """Rebuild a model from the JSON configuration in ``filename``.

    Args:
        filename: path of the JSON configuration to load.

    Returns:
        The model built from the configuration.
    """
    with open(filename, "r") as config:
        return K.models.model_from_json(config.read())
