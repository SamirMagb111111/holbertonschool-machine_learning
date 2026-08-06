#!/usr/bin/env python3
"""Module for Adam optimization."""

import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon,
                          var, grad, v, s, t):
    """Update a variable using the Adam optimization algorithm.

    Args:
        alpha (float): Learning rate.
        beta1 (float): Weight for the first moment.
        beta2 (float): Weight for the second moment.
        epsilon (float): Small value to avoid division by zero.
        var (numpy.ndarray): Variable to update.
        grad (numpy.ndarray): Gradient of the variable.
        v: Previous first moment.
        s: Previous second moment.
        t (int): Time step for bias correction.

    Returns:
        tuple: Updated variable, first moment, and second moment.
    """
    v = beta1 * v + (1 - beta1) * grad
    s = beta2 * s + (1 - beta2) * (grad ** 2)

    v_corrected = v / (1 - beta1 ** t)
    s_corrected = s / (1 - beta2 ** t)

    var = var - alpha * v_corrected / (
        np.sqrt(s_corrected) + epsilon
    )

    return var, v, s
