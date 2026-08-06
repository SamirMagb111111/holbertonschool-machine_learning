#!/usr/bin/env python3
"""Module for creating a batch normalized neural network layer."""

import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """Create a batch normalized layer.

    Args:
        prev: Activated output of the previous layer.
        n (int): Number of nodes in the new layer.
        activation: Activation function for the layer.

    Returns:
        Tensor: Activated output of the batch normalized layer.
    """
    initializer = tf.keras.initializers.VarianceScaling(mode='fan_avg')

    dense = tf.keras.layers.Dense(
        n,
        kernel_initializer=initializer
    )

    Z = dense(prev)

    gamma = tf.Variable(tf.ones(n), trainable=True)
    beta = tf.Variable(tf.zeros(n), trainable=True)

    mean, variance = tf.nn.moments(Z, axes=[0])

    Z_norm = tf.nn.batch_normalization(
        Z,
        mean,
        variance,
        beta,
        gamma,
        1e-7
    )

    return activation(Z_norm)
