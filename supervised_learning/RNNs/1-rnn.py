#!/usr/bin/env python3
"""Forward propagation for a simple RNN."""
import numpy as np


def rnn(rnn_cell, X, h_0):
    """Perform forward propagation for a simple RNN.

    Args:
        rnn_cell: an instance of RNNCell used for the forward propagation.
        X: numpy.ndarray of shape (t, m, i), the data to be used.
        h_0: numpy.ndarray of shape (m, h), the initial hidden state.

    Returns:
        H: numpy.ndarray containing all of the hidden states.
        Y: numpy.ndarray containing all of the outputs.
    """
    t, m, i = X.shape
    h = h_0.shape[1]
    o = rnn_cell.Wy.shape[1]

    H = np.zeros((t + 1, m, h))
    Y = np.zeros((t, m, o))
    H[0] = h_0

    for time in range(t):
        h_next, y = rnn_cell.forward(H[time], X[time])
        H[time + 1] = h_next
        Y[time] = y

    return H, Y
