#!/usr/bin/env python3
"""Module for creating an RMSProp optimizer."""

import tensorflow as tf


def create_RMSProp_op(alpha, beta2, epsilon):
    """Create an RMSProp optimizer.

    Args:
        alpha (float): Learning rate.
        beta2 (float): RMSProp discounting factor.
        epsilon (float): Small value to avoid division by zero.

    Returns:
        tensorflow.keras.optimizers.Optimizer: RMSProp optimizer.
    """
    optimizer = tf.keras.optimizers.RMSprop(
        learning_rate=alpha,
        rho=beta2,
        epsilon=epsilon
    )

    return optimizer
