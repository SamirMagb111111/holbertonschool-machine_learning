#!/usr/bin/env python3
"""Module for determining early stopping."""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """Determine whether gradient descent should stop early.

    Args:
        cost (float): Current validation cost.
        opt_cost (float): Lowest recorded validation cost.
        threshold (float): Minimum required improvement.
        patience (int): Number of tolerated unsuccessful checks.
        count (int): Current number of unsuccessful checks.

    Returns:
        tuple: Boolean indicating whether to stop and updated count.
    """
    if opt_cost - cost > threshold:
        count = 0
    else:
        count += 1

    return count >= patience, count
