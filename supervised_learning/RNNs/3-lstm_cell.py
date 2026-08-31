#!/usr/bin/env python3
"""Long Short-Term Memory cell implemented with NumPy."""
import numpy as np


class LSTMCell:
    """Represents an LSTM unit."""

    def __init__(self, i, h, o):
        """Initialize the cell.

        Args:
            i: dimensionality of the data.
            h: dimensionality of the hidden state.
            o: dimensionality of the outputs.
        """
        self.Wf = np.random.randn(i + h, h)
        self.Wu = np.random.randn(i + h, h)
        self.Wc = np.random.randn(i + h, h)
        self.Wo = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)
        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, c_prev, x_t):
        """Perform forward propagation for one time step.

        Args:
            h_prev: numpy.ndarray of shape (m, h), previous hidden state.
            c_prev: numpy.ndarray of shape (m, h), previous cell state.
            x_t: numpy.ndarray of shape (m, i), data input for the cell.

        Returns:
            h_next: the next hidden state.
            c_next: the next cell state.
            y: the output of the cell (softmax).
        """
        concat = np.concatenate((h_prev, x_t), axis=1)
        f = 1 / (1 + np.exp(-(np.matmul(concat, self.Wf) + self.bf)))
        u = 1 / (1 + np.exp(-(np.matmul(concat, self.Wu) + self.bu)))
        c_candidate = np.tanh(np.matmul(concat, self.Wc) + self.bc)
        c_next = f * c_prev + u * c_candidate
        o_gate = 1 / (1 + np.exp(-(np.matmul(concat, self.Wo) + self.bo)))
        h_next = o_gate * np.tanh(c_next)

        y_linear = np.matmul(h_next, self.Wy) + self.by
        exp = np.exp(y_linear - np.max(y_linear, axis=1, keepdims=True))
        y = exp / np.sum(exp, axis=1, keepdims=True)
        return h_next, c_next, y
