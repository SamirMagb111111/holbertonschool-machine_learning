#!/usr/bin/env python3
"""Module for performing valid convolution on grayscale images."""

import numpy as np


def convolve_grayscale_valid(images, kernel):
    """Perform a valid convolution on grayscale images.

    Args:
        images (numpy.ndarray): Images of shape (m, h, w).
        kernel (numpy.ndarray): Kernel of shape (kh, kw).

    Returns:
        numpy.ndarray: Convolved images.
    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    output_h = h - kh + 1
    output_w = w - kw + 1

    convolved = np.zeros((m, output_h, output_w))

    for i in range(output_h):
        for j in range(output_w):
            region = images[:, i:i + kh, j:j + kw]
            convolved[:, i, j] = np.sum(
                region * kernel,
                axis=(1, 2)
            )

    return convolved
