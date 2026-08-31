#!/usr/bin/env python3
"""Train an agent with Q-learning on FrozenLake."""
import numpy as np

epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy


def train(env, Q, episodes=5000, max_steps=100, alpha=0.1,
          gamma=0.99, epsilon=1, min_epsilon=0.1,
          epsilon_decay=0.05):
    """Perform Q-learning.

    When the agent falls in a hole the reward is updated to -1.

    Args:
        env: the FrozenLake environment.
        Q: the Q-table (numpy.ndarray).
        episodes: total number of training episodes.
        max_steps: maximum number of steps per episode.
        alpha: the learning rate.
        gamma: the discount rate.
        epsilon: the initial threshold for epsilon-greedy.
        min_epsilon: the minimum value that epsilon should decay to.
        epsilon_decay: the decay rate for epsilon between episodes.

    Returns:
        Q: the updated Q-table.
        total_rewards: list with the reward obtained on each episode.
    """
    initial_epsilon = epsilon
    total_rewards = []

    for episode in range(episodes):
        state, _ = env.reset()
        episode_reward = 0

        for _ in range(max_steps):
            action = epsilon_greedy(Q, state, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)

            if terminated and reward == 0:
                reward = -1

            Q[state, action] += alpha * (
                reward + gamma * np.max(Q[next_state]) - Q[state, action])

            episode_reward += reward
            state = next_state

            if terminated or truncated:
                break

        epsilon = min_epsilon + (initial_epsilon - min_epsilon) * np.exp(
            -epsilon_decay * episode)
        total_rewards.append(episode_reward)

    return Q, total_rewards
