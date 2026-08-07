#!/usr/bin/env python3
"""Module for convolutional forward propagation."""

import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same",
                 stride=(1, 1)):
    """Perform forward propagation over a convolutional layer.

    Args:
        A_prev (numpy.ndarray): Previous layer output of shape
            (m, h_prev, w_prev, c_prev).
        W (numpy.ndarray): Kernels of shape
            (kh, kw, c_prev, c_new).
        b (numpy.ndarray): Biases of shape (1, 1, 1, c_new).
        activation (function): Activation function.
        padding (str): Either 'same' or 'valid'.
        stride (tuple): Strides (sh, sw).

    Returns:
        numpy.ndarray: Output of the convolutional layer.
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride

    if padding == "same":
        ph = int(np.ceil(
            (((h_prev - 1) * sh) + kh - h_prev) / 2
        ))
        pw = int(np.ceil(
            (((w_prev - 1) * sw) + kw - w_prev) / 2
        ))
    else:
        ph = 0
        pw = 0

    h_new = ((h_prev + (2 * ph) - kh) // sh) + 1
    w_new = ((w_prev + (2 * pw) - kw) // sw) + 1

    A_prev_pad = np.pad(
        A_prev,
        ((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode="constant"
    )

    Z = np.zeros((m, h_new, w_new, c_new))

    for i in range(h_new):
        for j in range(w_new):
            region = A_prev_pad[
                :,
                i * sh:i * sh + kh,
                j * sw:j * sw + kw,
                :
            ]

            for k in range(c_new):
                Z[:, i, j, k] = (
                    np.sum(
                        region * W[:, :, :, k],
                        axis=(1, 2, 3)
                    ) + b[0, 0, 0, k]
                )

    return activation(Z)
