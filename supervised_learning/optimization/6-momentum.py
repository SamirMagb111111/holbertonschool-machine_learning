#!/usr/bin/env python3
"""Module for creating a momentum optimizer."""

import tensorflow as tf


def create_momentum_op(alpha, beta1):
    """Create a gradient descent optimizer with momentum.

    Args:
        alpha (float): Learning rate.
        beta1 (float): Momentum weight.

    Returns:
        tensorflow.keras.optimizers.Optimizer: Momentum optimizer.
    """
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=alpha,
        momentum=beta1
    )

    return optimizer
