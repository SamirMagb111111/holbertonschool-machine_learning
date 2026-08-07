#!/usr/bin/env python3
"""Module for convolution using multiple kernels."""

import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """Perform convolution on images using multiple kernels.

    Args:
        images (numpy.ndarray): Images of shape (m, h, w, c).
        kernels (numpy.ndarray): Kernels of shape (kh, kw, c, nc).
        padding: 'same', 'valid', or tuple (ph, pw).
        stride (tuple): Stride values (sh, sw).

    Returns:
        numpy.ndarray: Convolved images.
    """
    m, h, w, c = images.shape
    kh, kw, kc, nc = kernels.shape
    sh, sw = stride

    if padding == 'same':
        ph = int(np.ceil(
            (((h - 1) * sh) + kh - h) / 2
        ))
        pw = int(np.ceil(
            (((w - 1) * sw) + kw - w) / 2
        ))
        output_h = h
        output_w = w
    elif padding == 'valid':
        ph = 0
        pw = 0
        output_h = (h - kh) // sh + 1
        output_w = (w - kw) // sw + 1
    else:
        ph, pw = padding
        output_h = (h + 2 * ph - kh) // sh + 1
        output_w = (w + 2 * pw - kw) // sw + 1

    padded = np.pad(
        images,
        ((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode='constant'
    )

    convolved = np.zeros((m, output_h, output_w, nc))

    for i in range(output_h):
        for j in range(output_w):
            region = padded[
                :,
                i * sh:i * sh + kh,
                j * sw:j * sw + kw,
                :
            ]

            for k in range(nc):
                convolved[:, i, j, k] = np.sum(
                    region * kernels[:, :, :, k],
                    axis=(1, 2, 3)
                )

    return convolved
