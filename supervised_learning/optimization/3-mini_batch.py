#!/usr/bin/env python3
"""Module for creating mini-batches."""

import numpy as np

shuffle_data = __import__('2-shuffle_data').shuffle_data


def create_mini_batches(X, Y, batch_size):
    """Create mini-batches from input data and labels.

    Args:
        X (numpy.ndarray): Input data of shape (m, nx).
        Y (numpy.ndarray): Labels of shape (m, ny).
        batch_size (int): Number of data points per batch.

    Returns:
        list: A list of tuples containing X and Y mini-batches.
    """
    X_shuffled, Y_shuffled = shuffle_data(X, Y)
    mini_batches = []

    m = X.shape[0]

    for start in range(0, m, batch_size):
        end = start + batch_size
        X_batch = X_shuffled[start:end]
        Y_batch = Y_shuffled[start:end]

        mini_batches.append((X_batch, Y_batch))

    return mini_batches
