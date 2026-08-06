#!/usr/bin/env python3
"""Module for gradient descent with momentum."""


def update_variables_momentum(alpha, beta1, var, grad, v):
    """Update a variable using gradient descent with momentum.

    Args:
        alpha (float): Learning rate.
        beta1 (float): Momentum weight.
        var: Variable to update.
        grad: Gradient of the variable.
        v: Previous first moment.

    Returns:
        tuple: Updated variable and new moment.
    """
    v = beta1 * v + (1 - beta1) * grad
    var = var - alpha * v

    return var, v
