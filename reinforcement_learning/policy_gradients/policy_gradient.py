#!/usr/bin/env python3
"""Monte-Carlo policy gradient for a softmax linear policy."""
import numpy as np


def policy(matrix, weight):
    """Compute the softmax policy over actions.

    Args:
        matrix: state/input matrix of shape ``(m, n)``.
        weight: policy weight matrix of shape ``(n, a)``.

    Returns:
        numpy.ndarray of shape ``(m, a)`` with the action probabilities.
    """
    scores = matrix @ weight
    scores -= np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(scores)
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)


def policy_gradient(state, weight):
    """Sample an action and compute the gradient of ``log pi(a|s)``.

    Args:
        state: current observation (1-D array of length ``n``).
        weight: policy weight matrix of shape ``(n, a)``.

    Returns:
        action: the sampled action index.
        gradient: numpy.ndarray of shape ``(n, a)``.
    """
    state = state.reshape(1, -1)
    probs = policy(state, weight)
    action = np.random.choice(probs.shape[1], p=probs[0])
    one_hot = np.zeros_like(probs)
    one_hot[0, action] = 1
    gradient = state.T @ (one_hot - probs)
    return action, gradient
