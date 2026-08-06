#!/usr/bin/env python3
"""Module for normalizing a data matrix."""

import numpy as np


def normalize(X, m, s):
    """Normalize a matrix using mean and standard deviation.

    Args:
        X (numpy.ndarray): Data matrix of shape (d, nx).
        m (numpy.ndarray): Mean of each feature.
        s (numpy.ndarray): Standard deviation of each feature.

    Returns:
        numpy.ndarray: The normalized data matrix.
    """
    return (X - m) / s
