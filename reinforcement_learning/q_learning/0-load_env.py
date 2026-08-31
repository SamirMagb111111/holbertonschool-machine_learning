#!/usr/bin/env python3
"""Load the FrozenLake environment from Gymnasium."""
import gymnasium as gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """Load the pre-made FrozenLakeEnv environment.

    Args:
        desc: None or a list of lists describing a custom map.
        map_name: None or a string with a pre-made map name (e.g. "4x4").
        is_slippery: whether the ice is slippery (stochastic transitions).

    Returns:
        The FrozenLake environment.
    """
    return gym.make("FrozenLake-v1", desc=desc, map_name=map_name,
                    is_slippery=is_slippery, render_mode="ansi")
