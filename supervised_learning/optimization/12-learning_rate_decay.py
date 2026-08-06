#!/usr/bin/env python3
"""Module for TensorFlow inverse time learning rate decay."""

import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, decay_step):
    """Create a stepwise inverse time learning rate decay schedule.

    Args:
        alpha (float): Original learning rate.
        decay_rate (float): Rate at which the learning rate decays.
        decay_step (int): Number of steps before further decay.

    Returns:
        tf.keras.optimizers.schedules.InverseTimeDecay: Decay schedule.
    """
    return tf.keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=alpha,
        decay_steps=decay_step,
        decay_rate=decay_rate,
        staircase=True
    )
