#!/usr/bin/env python3
"""Module for pooling forward propagation."""

import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """Perform forward propagation over a pooling layer.

    Args:
        A_prev (numpy.ndarray): Previous layer output of shape
            (m, h_prev, w_prev, c_prev).
        kernel_shape (tuple): Pooling kernel shape (kh, kw).
        stride (tuple): Strides (sh, sw).
        mode (str): Either 'max' or 'avg'.

    Returns:
        numpy.ndarray: Output of the pooling layer.
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    h_new = ((h_prev - kh) // sh) + 1
    w_new = ((w_prev - kw) // sw) + 1

    A = np.zeros((m, h_new, w_new, c_prev))

    for i in range(h_new):
        for j in range(w_new):
            region = A_prev[
                :,
                i * sh:i * sh + kh,
                j * sw:j * sw + kw,
                :
            ]

            if mode == 'max':
                A[:, i, j, :] = np.max(region, axis=(1, 2))
            else:
                A[:, i, j, :] = np.mean(region, axis=(1, 2))

    return A
