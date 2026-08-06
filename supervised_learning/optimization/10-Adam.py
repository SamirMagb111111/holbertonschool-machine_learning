#!/usr/bin/env python3
"""Module for creating an Adam optimizer."""

import tensorflow as tf


def create_Adam_op(alpha, beta1, beta2, epsilon):
    """Create an Adam optimizer.

    Args:
        alpha (float): Learning rate.
        beta1 (float): Weight for the first moment.
        beta2 (float): Weight for the second moment.
        epsilon (float): Small value to avoid division by zero.

    Returns:
        tensorflow.keras.optimizers.Optimizer: Adam optimizer.
    """
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=alpha,
        beta_1=beta1,
        beta_2=beta2,
        epsilon=epsilon
    )

    return optimizer
