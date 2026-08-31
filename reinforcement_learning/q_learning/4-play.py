#!/usr/bin/env python3
"""Have a trained agent play an episode of FrozenLake."""
import numpy as np


def play(env, Q, max_steps=100):
    """Play one episode, always exploiting the Q-table.

    Args:
        env: the FrozenLake environment.
        Q: the trained Q-table (numpy.ndarray).
        max_steps: maximum number of steps in the episode.

    Returns:
        total_rewards: the total reward for the episode.
        rendered_outputs: list with the board rendered at each step.
    """
    state, _ = env.reset()
    total_rewards = 0
    rendered_outputs = [env.render()]

    for _ in range(max_steps):
        action = np.argmax(Q[state])
        next_state, reward, terminated, truncated, _ = env.step(action)
        rendered_outputs.append(env.render())
        total_rewards += reward
        state = next_state

        if terminated or truncated:
            break

    return total_rewards, rendered_outputs
