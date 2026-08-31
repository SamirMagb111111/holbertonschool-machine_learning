#!/usr/bin/env python3
"""Train a softmax policy on CartPole with Monte-Carlo policy gradient."""
import numpy as np
policy_gradient = __import__('policy_gradient').policy_gradient


def train(env, nb_episodes, alpha=0.000045, gamma=0.98,
          show_result=False):
    """Train the policy with the Monte-Carlo policy gradient algorithm.

    Args:
        env: the CartPole environment.
        nb_episodes: number of training episodes.
        alpha: the learning rate.
        gamma: the discount factor.
        show_result: render every 1000th episode when True.

    Returns:
        scores: list with the total reward obtained in every episode.
    """
    weight = np.random.rand(env.observation_space.shape[0],
                            env.action_space.n)
    scores = []

    for episode in range(nb_episodes):
        state, _ = env.reset()
        rewards = []
        gradients = []
        done = False

        while not done:
            if show_result and episode % 1000 == 0:
                env.render()
            action, gradient = policy_gradient(state, weight)
            state, reward, terminated, truncated, _ = env.step(action)
            rewards.append(reward)
            gradients.append(gradient)
            done = terminated or truncated

        G = 0
        for reward, gradient in zip(reversed(rewards), reversed(gradients)):
            G = reward + gamma * G
            weight += alpha * G * gradient

        score = sum(rewards)
        scores.append(score)
        print("Episode: {} Score: {}".format(episode, score))

    return scores
