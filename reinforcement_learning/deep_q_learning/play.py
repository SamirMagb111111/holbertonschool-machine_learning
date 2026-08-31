#!/usr/bin/env python3
"""Watch the trained DQN agent play Atari Breakout greedily."""
import numpy as np
import gymnasium as gym
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Permute, Conv2D, Flatten, Dense)
from tensorflow.keras.optimizers.legacy import Adam
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import GreedyQPolicy
from rl.core import Processor

ENV_NAME = "ALE/Breakout-v5"
INPUT_SHAPE = (84, 84)
WINDOW_LENGTH = 4


class KerasRLWrapper(gym.Wrapper):
    """Adapt a Gymnasium env to the old Gym API that keras-rl2 expects."""

    def reset(self, **kwargs):
        """Reset the env and return only the observation."""
        observation, _ = self.env.reset(**kwargs)
        return observation

    def step(self, action):
        """Step the env, merging ``terminated``/``truncated`` to ``done``."""
        observation, reward, terminated, truncated, info = self.env.step(
            action)
        return observation, reward, terminated or truncated, info

    def render(self, *args, **kwargs):
        """Render using the mode chosen when the env was created."""
        return self.env.render()


class AtariProcessor(Processor):
    """Down-sample Atari frames and normalize replay batches."""

    def process_observation(self, observation):
        """Convert an RGB frame to an 84x84 uint8 grayscale image."""
        img = Image.fromarray(observation).resize(INPUT_SHAPE).convert("L")
        return np.array(img).astype("uint8")

    def process_state_batch(self, batch):
        """Scale a stacked-frame batch to floats in ``[0, 1]``."""
        return batch.astype("float32") / 255.0

    def process_reward(self, reward):
        """Clip rewards to ``[-1, 1]``."""
        return np.clip(reward, -1.0, 1.0)


def build_model(nb_actions):
    """Build the convolutional DQN network (identical to ``train.py``).

    Args:
        nb_actions: number of possible actions (Q-values to output).

    Returns:
        An uncompiled ``keras.Sequential`` model.
    """
    model = Sequential([
        Permute((2, 3, 1), input_shape=(WINDOW_LENGTH,) + INPUT_SHAPE),
        Conv2D(32, (8, 8), strides=(4, 4), activation="relu"),
        Conv2D(64, (4, 4), strides=(2, 2), activation="relu"),
        Conv2D(64, (3, 3), strides=(1, 1), activation="relu"),
        Flatten(),
        Dense(512, activation="relu"),
        Dense(nb_actions, activation="linear"),
    ])
    return model


def build_agent(model, nb_actions):
    """Build a :class:`DQNAgent` with a greedy policy for inference."""
    memory = SequentialMemory(limit=1000000, window_length=WINDOW_LENGTH)
    return DQNAgent(
        model=model, nb_actions=nb_actions, memory=memory,
        processor=AtariProcessor(), policy=GreedyQPolicy(),
        nb_steps_warmup=50000, gamma=0.99, target_model_update=10000)


def main():
    """Load ``policy.h5`` and let the greedy agent play Breakout."""
    env = KerasRLWrapper(gym.make(ENV_NAME, render_mode="human"))
    nb_actions = env.action_space.n

    model = build_model(nb_actions)
    dqn = build_agent(model, nb_actions)
    dqn.compile(Adam(learning_rate=0.00025), metrics=["mae"])

    dqn.load_weights("policy.h5")
    dqn.test(env, nb_episodes=5, visualize=True)
    env.close()


if __name__ == "__main__":
    main()
