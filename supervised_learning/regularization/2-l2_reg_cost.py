#!/usr/bin/env python3
"""Module for calculating L2 regularized cost in Keras."""

import tensorflow as tf


def l2_reg_cost(cost, model):
    """Calculate total cost with L2 regularization for each layer.

    Args:
        cost: Cost of the network without regularization.
        model: Keras model containing L2 regularized layers.

    Returns:
        Tensor containing the regularized cost for each layer.
    """
    return cost + tf.stack(model.losses)
