#!/usr/bin/env python3
"""Forward propagation for a deep RNN."""
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """Perform forward propagation for a deep RNN.

    Args:
        rnn_cells: list of RNNCell instances of length l, one per layer.
        X: numpy.ndarray of shape (t, m, i), the data to be used.
        h_0: numpy.ndarray of shape (l, m, h), the initial hidden state.

    Returns:
        H: numpy.ndarray containing all of the hidden states.
        Y: numpy.ndarray containing all of the outputs.
    """
    t, m, i = X.shape
    layers = len(rnn_cells)
    h = h_0.shape[2]
    o = rnn_cells[-1].Wy.shape[1]

    H = np.zeros((t + 1, layers, m, h))
    Y = np.zeros((t, m, o))
    H[0] = h_0

    for time in range(t):
        x = X[time]
        for layer in range(layers):
            h_next, y = rnn_cells[layer].forward(H[time, layer], x)
            H[time + 1, layer] = h_next
            x = h_next
        Y[time] = y

    return H, Y
