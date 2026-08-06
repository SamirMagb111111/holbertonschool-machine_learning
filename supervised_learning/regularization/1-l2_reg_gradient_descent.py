#!/usr/bin/env python3
"""Module for gradient descent with L2 regularization."""

import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """Update neural network weights using L2 regularization.

    Args:
        Y (numpy.ndarray): One-hot labels of shape (classes, m).
        weights (dict): Dictionary containing weights and biases.
        cache (dict): Dictionary containing layer activations.
        alpha (float): Learning rate.
        lambtha (float): L2 regularization parameter.
        L (int): Number of layers.
    """
    m = Y.shape[1]
    dZ = cache['A{}'.format(L)] - Y

    for layer in range(L, 0, -1):
        A_prev = cache['A{}'.format(layer - 1)]
        W = weights['W{}'.format(layer)]

        dW = np.matmul(dZ, A_prev.T) / m
        dW += (lambtha / m) * W
        db = np.sum(dZ, axis=1, keepdims=True) / m

        if layer > 1:
            dA_prev = np.matmul(W.T, dZ)
            dZ_prev = dA_prev * (1 - A_prev ** 2)

        weights['W{}'.format(layer)] -= alpha * dW
        weights['b{}'.format(layer)] -= alpha * db

        if layer > 1:
            dZ = dZ_prev
