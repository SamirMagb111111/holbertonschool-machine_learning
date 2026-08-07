#!/usr/bin/env python3
"""Module for creating a neural network layer with dropout."""

import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob, training=True):
    """Create a neural network layer using dropout.

    Args:
        prev: Output tensor of the previous layer.
        n (int): Number of nodes in the new layer.
        activation: Activation function for the new layer.
        keep_prob (float): Probability that a node is kept.
        training (bool): Whether the model is in training mode.

    Returns:
        Tensor: Output of the layer after dropout.
    """
    initializer = tf.keras.initializers.VarianceScaling(
        scale=2.0,
        mode='fan_avg'
    )

    dense = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=initializer
    )

    output = dense(prev)

    dropout = tf.keras.layers.Dropout(
        rate=1 - keep_prob
    )

    return dropout(output, training=training)
