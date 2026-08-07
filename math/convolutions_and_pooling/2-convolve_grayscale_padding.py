#!/usr/bin/env python3
"""Module for convolution with custom padding."""

import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """Perform convolution on grayscale images with custom padding.

    Args:
        images (numpy.ndarray): Images of shape (m, h, w).
        kernel (numpy.ndarray): Kernel of shape (kh, kw).
        padding (tuple): Padding values (ph, pw).

    Returns:
        numpy.ndarray: Convolved images.
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = padding

    padded = np.pad(
        images,
        ((0, 0), (ph, ph), (pw, pw)),
        mode='constant'
    )

    output_h = h + (2 * ph) - kh + 1
    output_w = w + (2 * pw) - kw + 1

    convolved = np.zeros((m, output_h, output_w))

    for i in range(output_h):
        for j in range(output_w):
            region = padded[:, i:i + kh, j:j + kw]
            convolved[:, i, j] = np.sum(
                region * kernel,
                axis=(1, 2)
            )

    return convolved
