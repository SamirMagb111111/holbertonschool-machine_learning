#!/usr/bin/env python3
"""Initialize the Q-table for a FrozenLake environment."""
import numpy as np


def q_init(env):
    """Initialize the Q-table with zeros.

    Args:
        env: the FrozenLake environment.

    Returns:
        numpy.ndarray of shape ``(n_states, n_actions)`` filled with zeros.
    """
    return np.zeros((env.observation_space.n, env.action_space.n))
