#!/usr/bin/env python3
"""Module for RMSProp optimization."""

import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """Update a variable using the RMSProp optimization algorithm.

    Args:
        alpha (float): Learning rate.
        beta2 (float): RMSProp weight.
        epsilon (float): Small value to avoid division by zero.
        var (numpy.ndarray): Variable to update.
        grad (numpy.ndarray): Gradient of the variable.
        s: Previous second moment.

    Returns:
        tuple: Updated variable and new second moment.
    """
    s = beta2 * s + (1 - beta2) * (grad ** 2)
    var = var - alpha * grad / (np.sqrt(s) + epsilon)

    return var, s
