#!/usr/bin/env python3
"""Module for randomly adjusting image brightness."""

import tensorflow as tf


def change_brightness(image, max_delta):
    """Randomly change the brightness of an image.

    Args:
        image (tf.Tensor): A 3D tensor containing an image.
        max_delta (float): Maximum brightness adjustment.

    Returns:
        tf.Tensor: The brightness-adjusted image.
    """
    return tf.image.random_brightness(image, max_delta=max_delta)
