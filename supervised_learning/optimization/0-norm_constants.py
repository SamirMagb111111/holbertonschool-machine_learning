#!/usr/bin/env python3
"""Module for calculating normalization constants."""

import numpy as np


def normalization_constants(X):
    """Calculate the mean and standard deviation of each feature.

    Args:
        X (numpy.ndarray): Data matrix of shape (m, nx).

    Returns:
        tuple: Mean and standard deviation of each feature.
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)

    return mean, std
