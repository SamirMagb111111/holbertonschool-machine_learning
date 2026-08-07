#!/usr/bin/env python3
"""Module for gradient descent with dropout regularization."""

import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """Update neural network weights using dropout gradient descent.

    Args:
        Y (numpy.ndarray): One-hot labels of shape (classes, m).
        weights (dict): Dictionary containing weights and biases.
        cache (dict): Activations and dropout masks from forward propagation.
        alpha (float): Learning rate.
        keep_prob (float): Probability that a node is kept.
        L (int): Number of layers in the network.
    """
    m = Y.shape[1]
    dZ = cache['A{}'.format(L)] - Y

    for layer in range(L, 0, -1):
        A_prev = cache['A{}'.format(layer - 1)]
        W = weights['W{}'.format(layer)]

        dW = np.matmul(dZ, A_prev.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m

        if layer > 1:
            dA_prev = np.matmul(W.T, dZ)

            D_prev = cache['D{}'.format(layer - 1)]
            dA_prev = (dA_prev * D_prev) / keep_prob

            dZ_prev = dA_prev * (1 - A_prev ** 2)

        weights['W{}'.format(layer)] -= alpha * dW
        weights['b{}'.format(layer)] -= alpha * db

        if layer > 1:
            dZ = dZ_prev
