#!/usr/bin/env python3
"""One-hot encode a label vector."""
import tensorflow.keras as K


def one_hot(labels, classes=None):
    """Convert a label vector to a one-hot matrix.

    Args:
        labels: the vector of integer labels.
        classes: the number of classes; inferred by Keras when None.

    Returns:
        The one-hot matrix with the classes on the last axis.
    """
    return K.utils.to_categorical(labels, classes)
