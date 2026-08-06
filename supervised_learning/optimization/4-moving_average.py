#!/usr/bin/env python3
"""Module for calculating a weighted moving average."""


def moving_average(data, beta):
    """Calculate the bias-corrected weighted moving average.

    Args:
        data (list): Data points.
        beta (float): Weight used for the moving average.

    Returns:
        list: Bias-corrected moving averages.
    """
    averages = []
    v = 0

    for t, value in enumerate(data, start=1):
        v = beta * v + (1 - beta) * value
        corrected_v = v / (1 - beta ** t)
        averages.append(corrected_v)

    return averages
