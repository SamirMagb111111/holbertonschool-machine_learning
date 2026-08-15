#!/usr/bin/env python3
"""Module for adjusting image hue."""

import tensorflow as tf


def change_hue(image, delta):
    """Change the hue of an image.

    Args:
        image (tf.Tensor): A 3D tensor containing an image.
        delta (float): Amount to adjust the hue.

    Returns:
        tf.Tensor: The hue-adjusted image.
    """
    return tf.image.adjust_hue(image, delta)
