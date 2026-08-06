#!/usr/bin/env python3
"""Module for calculating L2 regularized cost."""

import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """Calculate the cost of a network with L2 regularization.

    Args:
        cost: Cost without regularization.
        lambtha (float): Regularization parameter.
        weights (dict): Dictionary containing network weights.
        L (int): Number of layers.
        m (int): Number of data points.

    Returns:
        Cost including L2 regularization.
    """
    l2_cost = 0

    for layer in range(1, L + 1):
        l2_cost += np.linalg.norm(weights['W{}'.format(layer)]) ** 2

    return cost + (lambtha / (2 * m)) * l2_cost
