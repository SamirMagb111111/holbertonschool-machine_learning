#!/usr/bin/env python3
"""Module for inverse time learning rate decay."""


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """Update the learning rate using stepwise inverse time decay.

    Args:
        alpha (float): Original learning rate.
        decay_rate (float): Decay rate.
        global_step (int): Number of gradient descent steps elapsed.
        decay_step (int): Steps before the learning rate decays further.

    Returns:
        float: Updated learning rate.
    """
    return alpha / (
        1 + decay_rate * (global_step // decay_step)
    )
