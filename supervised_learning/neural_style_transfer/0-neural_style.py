#!/usr/bin/env python3
"""Neural Style Transfer module."""

import numpy as np
import tensorflow as tf


class NST:
    """Class that performs tasks for Neural Style Transfer."""

    style_layers = [
        'block1_conv1',
        'block2_conv1',
        'block3_conv1',
        'block4_conv1',
        'block5_conv1'
    ]

    content_layer = 'block5_conv2'

    def __init__(
            self, style_image, content_image,
            alpha=1e4, beta=1):
        """Initialize the Neural Style Transfer class."""
        if (
            not isinstance(style_image, np.ndarray)
            or style_image.ndim != 3
            or style_image.shape[2] != 3
        ):
            raise TypeError(
                "style_image must be a numpy.ndarray "
                "with shape (h, w, 3)"
            )

        if (
            not isinstance(content_image, np.ndarray)
            or content_image.ndim != 3
            or content_image.shape[2] != 3
        ):
            raise TypeError(
                "content_image must be a numpy.ndarray "
                "with shape (h, w, 3)"
            )

        if (
            not isinstance(
                alpha,
                (int, float, np.integer, np.floating)
            )
            or alpha < 0
        ):
            raise TypeError(
                "alpha must be a non-negative number"
            )

        if (
            not isinstance(
                beta,
                (int, float, np.integer, np.floating)
            )
            or beta < 0
        ):
            raise TypeError(
                "beta must be a non-negative number"
            )

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta

    @staticmethod
    def scale_image(image):
        """Scale an image for Neural Style Transfer."""
        if (
            not isinstance(image, np.ndarray)
            or image.ndim != 3
            or image.shape[2] != 3
        ):
            raise TypeError(
                "image must be a numpy.ndarray "
                "with shape (h, w, 3)"
            )

        height = image.shape[0]
        width = image.shape[1]

        scale = 512 / max(height, width)

        new_height = int(round(height * scale))
        new_width = int(round(width * scale))

        image = tf.convert_to_tensor(
            image,
            dtype=tf.float32
        )

        image = tf.image.resize(
            image,
            (new_height, new_width),
            method='bicubic'
        )

        image = image / 255.0

        image = tf.clip_by_value(
            image,
            0.0,
            1.0
        )

        image = tf.expand_dims(
            image,
            axis=0
        )

        return image
