#!/usr/bin/env python3
"""Module for batch normalization."""

import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """Normalize an unactivated output using batch normalization.

    Args:
        Z (numpy.ndarray): Unactivated output of shape (m, n).
        gamma (numpy.ndarray): Scale parameters of shape (1, n).
        beta (numpy.ndarray): Offset parameters of shape (1, n).
        epsilon (float): Small value to avoid division by zero.

    Returns:
        numpy.ndarray: Batch-normalized Z matrix.
    """
    mean = np.mean(Z, axis=0)
    variance = np.var(Z, axis=0)

    Z_norm = (Z - mean) / np.sqrt(variance + epsilon)

    return gamma * Z_norm + beta
