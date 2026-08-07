#!/usr/bin/env python3
"""Module for convolutional backward propagation."""

import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """Perform backward propagation over a convolutional layer.

    Args:
        dZ (numpy.ndarray): Gradient of the convolution output with shape
            (m, h_new, w_new, c_new).
        A_prev (numpy.ndarray): Previous layer output with shape
            (m, h_prev, w_prev, c_prev).
        W (numpy.ndarray): Kernels with shape
            (kh, kw, c_prev, c_new).
        b (numpy.ndarray): Biases with shape (1, 1, 1, c_new).
        padding (str): Either 'same' or 'valid'.
        stride (tuple): Strides (sh, sw).

    Returns:
        tuple: Gradients dA_prev, dW, and db.
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    _, h_new, w_new, _ = dZ.shape
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

    A_prev_pad = np.pad(
        A_prev,
        ((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode="constant"
    )

    dA_prev_pad = np.zeros_like(A_prev_pad, dtype=float)
    dW = np.zeros_like(W, dtype=float)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    for i in range(h_new):
        for j in range(w_new):
            vert_start = i * sh
            vert_end = vert_start + kh
            horiz_start = j * sw
            horiz_end = horiz_start + kw

            region = A_prev_pad[
                :,
                vert_start:vert_end,
                horiz_start:horiz_end,
                :
            ]

            for k in range(c_new):
                dz = dZ[:, i, j, k].reshape(m, 1, 1, 1)

                dA_prev_pad[
                    :,
                    vert_start:vert_end,
                    horiz_start:horiz_end,
                    :
                ] += W[:, :, :, k] * dz

                dW[:, :, :, k] += np.sum(
                    region * dz,
                    axis=0
                )

    dA_prev = dA_prev_pad[
        :,
        ph:ph + h_prev,
        pw:pw + w_prev,
        :
    ]

    return dA_prev, dW, db
