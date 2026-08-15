#!/usr/bin/env python3
"""Module for randomly cropping an image."""

import tensorflow as tf


def crop_image(image, size):
    """Perform a random crop of an image.

    Args:
        image (tf.Tensor): A 3D tensor containing an image.
        size (tuple): Size of the crop.

    Returns:
        tf.Tensor: The randomly cropped image.
    """
    return tf.image.random_crop(image, size=size)
