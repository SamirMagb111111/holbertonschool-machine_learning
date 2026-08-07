#!/usr/bin/env python3
"""Module for pooling backward propagation."""

import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """Perform backward propagation over a pooling layer.

    Args:
        dA (numpy.ndarray): Gradient with respect to the pooling output.
        A_prev (numpy.ndarray): Output of the previous layer.
        kernel_shape (tuple): Pooling kernel shape (kh, kw).
        stride (tuple): Strides (sh, sw).
        mode (str): Either 'max' or 'avg'.

    Returns:
        numpy.ndarray: Gradient with respect to the previous layer.
    """
    m, h_prev, w_prev, c = A_prev.shape
    _, h_new, w_new, _ = dA.shape
    kh, kw = kernel_shape
    sh, sw = stride

    dA_prev = np.zeros_like(A_prev, dtype=float)

    for i in range(h_new):
        for j in range(w_new):
            vert_start = i * sh
            vert_end = vert_start + kh
            horiz_start = j * sw
            horiz_end = horiz_start + kw

            if mode == 'max':
                region = A_prev[
                    :,
                    vert_start:vert_end,
                    horiz_start:horiz_end,
                    :
                ]

                maximum = np.max(
                    region,
                    axis=(1, 2),
                    keepdims=True
                )

                mask = region == maximum

                dA_prev[
                    :,
                    vert_start:vert_end,
                    horiz_start:horiz_end,
                    :
                ] += mask * dA[:, i:i + 1, j:j + 1, :]

            else:
                gradient = dA[:, i:i + 1, j:j + 1, :] / (kh * kw)

                dA_prev[
                    :,
                    vert_start:vert_end,
                    horiz_start:horiz_end,
                    :
                ] += gradient

    return dA_prev
