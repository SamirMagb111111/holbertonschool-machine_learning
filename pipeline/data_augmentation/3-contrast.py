#!/usr/bin/env python3
"""Module for randomly adjusting image contrast."""

import tensorflow as tf


def change_contrast(image, lower, upper):
    """Randomly adjust the contrast of an image.

    Args:
        image (tf.Tensor): A 3D tensor containing an image.
        lower (float): Lower bound of the contrast factor.
        upper (float): Upper bound of the contrast factor.

    Returns:
        tf.Tensor: The contrast-adjusted image.
    """
    return tf.image.random_contrast(image, lower=lower, upper=upper)
