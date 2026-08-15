#!/usr/bin/env python3
"""Module for horizontally flipping an image."""

import tensorflow as tf


def flip_image(image):
    """Flip an image horizontally.

    Args:
        image (tf.Tensor): A 3D tensor containing an image.

    Returns:
        tf.Tensor: The horizontally flipped image.
    """
    return tf.image.flip_left_right(image)
