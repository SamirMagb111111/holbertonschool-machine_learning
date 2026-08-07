#!/usr/bin/env python3
"""Module for forward propagation with dropout."""

import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """Perform forward propagation using dropout.

    Args:
        X (numpy.ndarray): Input data of shape (nx, m).
        weights (dict): Dictionary containing weights and biases.
        L (int): Number of layers in the network.
        keep_prob (float): Probability of keeping a node active.

    Returns:
        dict: Layer activations and dropout masks.
    """
    cache = {'A0': X}

    for layer in range(1, L + 1):
        W = weights['W{}'.format(layer)]
        b = weights['b{}'.format(layer)]
        A_prev = cache['A{}'.format(layer - 1)]

        Z = np.matmul(W, A_prev) + b

        if layer == L:
            exp_Z = np.exp(Z)
            A = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
        else:
            A = np.tanh(Z)

            D = np.random.binomial(
                1,
                keep_prob,
                size=A.shape
            )

            A = (A * D) / keep_prob

            cache['D{}'.format(layer)] = D

        cache['A{}'.format(layer)] = A

    return cache
