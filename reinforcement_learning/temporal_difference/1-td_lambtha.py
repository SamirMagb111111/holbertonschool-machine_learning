#!/usr/bin/env python3
"""TD(lambda) state-value estimation with eligibility traces."""
import numpy as np


def td_lambtha(env, V, policy, lambtha, episodes=5000,
               max_steps=100, alpha=0.1, gamma=0.99):
    """Estimate the value function with the TD(lambda) algorithm.

    Args:
        env: the environment instance.
        V: numpy.ndarray of shape ``(s,)`` with the value estimate.
        policy: function that takes a state and returns an action.
        lambtha: the eligibility trace decay factor.
        episodes: number of episodes to train over.
        max_steps: maximum number of steps per episode.
        alpha: the learning rate.
        gamma: the discount rate.

    Returns:
        V: the updated value estimate.
    """
    for _ in range(episodes):
        state, _ = env.reset()
        eligibility = np.zeros(V.shape[0])

        for _ in range(max_steps):
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)

            delta = reward + gamma * V[next_state] - V[state]
            eligibility[state] += 1
            V += alpha * delta * eligibility
            eligibility *= gamma * lambtha

            state = next_state
            if terminated or truncated:
                break

    return V
