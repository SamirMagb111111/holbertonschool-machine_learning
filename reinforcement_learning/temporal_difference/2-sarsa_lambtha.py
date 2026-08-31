#!/usr/bin/env python3
"""SARSA(lambda) action-value estimation with eligibility traces."""
import numpy as np


def _epsilon_greedy(Q, state, epsilon):
    """Return an epsilon-greedy action for ``state``.

    Args:
        Q: the action-value estimate.
        state: the current state index.
        epsilon: the exploration threshold.

    Returns:
        The chosen action index.
    """
    if np.random.uniform() > epsilon:
        return np.argmax(Q[state])
    return np.random.randint(0, Q.shape[1])


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100,
                  alpha=0.1, gamma=0.99, epsilon=1,
                  min_epsilon=0.1, epsilon_decay=0.05):
    """Estimate the action-value function with SARSA(lambda).

    Args:
        env: the environment instance.
        Q: numpy.ndarray of shape ``(s, a)`` with the action-value estimate.
        lambtha: the eligibility trace decay factor.
        episodes: number of episodes to train over.
        max_steps: maximum number of steps per episode.
        alpha: the learning rate.
        gamma: the discount rate.
        epsilon: the initial exploration threshold.
        min_epsilon: the minimum value epsilon decays to.
        epsilon_decay: the exponential decay rate for epsilon.

    Returns:
        Q: the updated action-value estimate.
    """
    initial_epsilon = epsilon

    for episode in range(episodes):
        state, _ = env.reset()
        eligibility = np.zeros_like(Q)
        action = _epsilon_greedy(Q, state, epsilon)

        for _ in range(max_steps):
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_action = _epsilon_greedy(Q, next_state, epsilon)

            delta = (reward + gamma * Q[next_state, next_action]
                     - Q[state, action])
            eligibility[state, action] += 1
            Q += alpha * delta * eligibility
            eligibility *= gamma * lambtha

            state = next_state
            action = next_action
            if terminated or truncated:
                break

        epsilon = min_epsilon + (initial_epsilon - min_epsilon) * np.exp(
            -epsilon_decay * episode)

    return Q
