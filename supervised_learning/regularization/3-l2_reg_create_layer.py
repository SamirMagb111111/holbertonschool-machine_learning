#!/usr/bin/env python3
"""Module for creating a layer with L2 regularization."""

import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """Create a neural network layer with L2 regularization.

    Args:
        prev: Output tensor of the previous layer.
        n (int): Number of nodes in the new layer.
        activation: Activation function for the new layer.
        lambtha (float): L2 regularization parameter.

    Returns:
        Tensor: Output of the new layer.
    """
    initializer = tf.keras.initializers.VarianceScaling(
        scale=2.0,
        mode='fan_avg'
    )

    layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=initializer,
        kernel_regularizer=tf.keras.regularizers.L2(lambtha)
    )

    return layer(prev)
