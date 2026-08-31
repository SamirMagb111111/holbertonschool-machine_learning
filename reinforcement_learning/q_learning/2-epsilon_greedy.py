#!/usr/bin/env python3
"""Epsilon-greedy action selection."""
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """Select the next action using the epsilon-greedy policy.

    Args:
        Q: the Q-table (numpy.ndarray).
        state: the current state index.
        epsilon: the probability of exploring.

    Returns:
        The index of the chosen action.
    """
    if np.random.uniform(0, 1) < epsilon:
        return np.random.randint(0, Q.shape[1])
    return np.argmax(Q[state])
