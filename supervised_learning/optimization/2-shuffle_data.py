#!/usr/bin/env python3
"""Module for shuffling data."""

import numpy as np


def shuffle_data(X, Y):
    """Shuffle X and Y using the same permutation.

    Args:
        X (numpy.ndarray): First data matrix of shape (m, nx).
        Y (numpy.ndarray): Second data matrix of shape (m, ny).

    Returns:
        tuple: The shuffled X and Y matrices.
    """
    permutation = np.random.permutation(X.shape[0])

    return X[permutation], Y[permutation]
