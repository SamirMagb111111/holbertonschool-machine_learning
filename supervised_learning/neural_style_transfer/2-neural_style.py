#!/usr/bin/env python3
"""Neural Style Transfer module."""

import numpy as np
import tensorflow as tf


class NST:
    """Class that performs Neural Style Transfer tasks."""

    style_layers = [
        'block1_conv1',
        'block2_conv1',
        'block3_conv1',
        'block4_conv1',
        'block5_conv1'
    ]
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """Initialize the NST class."""
        if (not isinstance(style_image, np.ndarray) or
                style_image.ndim != 3 or style_image.shape[2] != 3):
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)"
            )

        if (not isinstance(content_image, np.ndarray) or
                content_image.ndim != 3 or content_image.shape[2] != 3):
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)"
            )

        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")

        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError("beta must be a non-negative number")

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta
        self.load_model()

    @staticmethod
    def scale_image(image):
        """Scale an image so its largest side is 512 pixels."""
        if (not isinstance(image, np.ndarray) or
                image.ndim != 3 or image.shape[2] != 3):
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)"
            )

        image = tf.image.resize(
            image,
            size=[512, 512],
            preserve_aspect_ratio=True,
            method='bicubic'
        )

        image = tf.expand_dims(image, axis=0)
        image = tf.clip_by_value(image, 0, 255) / 255

        return image

    def load_model(self):
        """Create the VGG19 model used to calculate cost."""
        vgg = tf.keras.applications.VGG19(
            include_top=False,
            weights='imagenet'
        )

        vgg.trainable = False

        outputs = []
        x = vgg.input

        for layer in vgg.layers[1:]:
            if isinstance(layer, tf.keras.layers.MaxPooling2D):
                x = tf.keras.layers.AveragePooling2D(
                    pool_size=layer.pool_size,
                    strides=layer.strides,
                    padding=layer.padding,
                    name=layer.name
                )(x)
            else:
                x = layer(x)

            if layer.name in self.style_layers:
                outputs.append(x)

            if layer.name == self.content_layer:
                outputs.append(x)
                break

        self.model = tf.keras.Model(
            inputs=vgg.input,
            outputs=outputs
        )

    @staticmethod
    def gram_matrix(input_layer):
        """Calculate the Gram matrix of a layer output."""
        if (not isinstance(input_layer, (tf.Tensor, tf.Variable)) or
                len(input_layer.shape) != 4):
            raise TypeError(
                "input_layer must be a tensor of rank 4"
            )

        gram = tf.linalg.einsum(
            'bijc,bijd->bcd',
            input_layer,
            input_layer
        )

        height = tf.shape(input_layer)[1]
        width = tf.shape(input_layer)[2]

        gram = gram / tf.cast(
            height * width,
            tf.float32
        )

        return gram
