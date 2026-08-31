#!/usr/bin/env python3
"""Test a Keras model."""
import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """Evaluate the model on a dataset.

    Args:
        network: the model to test.
        data: the input data to test with.
        labels: the one-hot labels of ``data``.
        verbose: whether to print the evaluation output.

    Returns:
        The loss and accuracy of the model as ``[loss, accuracy]``.
    """
    return network.evaluate(data, labels, verbose=verbose)
