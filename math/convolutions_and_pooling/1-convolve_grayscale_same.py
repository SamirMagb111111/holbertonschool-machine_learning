#!/usr/bin/env python3
"""Module for performing same convolution on grayscale images."""

import numpy as np


def convolve_grayscale_same(images, kernel):
    """Perform a same convolution on grayscale images.

    Args:
        images (numpy.ndarray): Images of shape (m, h, w).
        kernel (numpy.ndarray): Kernel of shape (kh, kw).

    Returns:
        numpy.ndarray: Convolved images with the same height and width.
    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    pad_h = kh - 1
    pad_w = kw - 1

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    padded = np.pad(
        images,
        (
            (0, 0),
            (pad_top, pad_bottom),
            (pad_left, pad_right)
        ),
        mode='constant'
    )

    convolved = np.zeros((m, h, w))

    for i in range(h):
        for j in range(w):
            region = padded[:, i:i + kh, j:j + kw]
            convolved[:, i, j] = np.sum(
                region * kernel,
                axis=(1, 2)
            )

    return convolved
