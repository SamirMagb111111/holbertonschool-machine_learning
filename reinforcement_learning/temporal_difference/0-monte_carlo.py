#!/usr/bin/env python3
"""Monte Carlo state-value estimation."""
import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100,
                alpha=0.1, gamma=0.99):
    """Estimate the value function with the Monte Carlo algorithm.

    Args:
        env: the environment instance.
        V: numpy.ndarray of shape ``(s,)`` with the value estimate.
        policy: function that takes a state and returns an action.
        episodes: number of episodes to train over.
        max_steps: maximum number of steps per episode.
        alpha: the learning rate.
        gamma: the discount rate.

    Returns:
        V: the updated value estimate.
    """
    for episode in range(episodes):
        state, _ = env.reset()
        episode_data = []

        for _ in range(max_steps):
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode_data.append([state, action, reward, next_state])
            if terminated or truncated:
                break
            state = next_state

        episode_data = np.array(episode_data, dtype=int)
        G = 0
        for step in reversed(range(len(episode_data))):
            state, action, reward, next_state = episode_data[step]
            G = gamma * G + reward
            if state not in episode_data[:episode, 0]:
                V[state] += alpha * (G - V[state])

    return V
