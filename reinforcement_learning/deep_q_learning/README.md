# Deep Q-learning — Atari Breakout

Train a Deep Q-Network with **keras-rl2** to play Atari Breakout, then watch it
play.

## Concepts

- **Reinforcement Learning** — an agent learns a behaviour by interacting with
  an environment and maximizing cumulative reward.
- **Q-learning** — learn `Q(s, a)`, the expected return of taking action `a`
  in state `s`, and act greedily with respect to it.
- **Deep Q-Network (DQN)** — approximate `Q(s, a)` with a convolutional
  network that reads raw pixels, because a table cannot cover every Atari
  screen.
- **Policy network** — the network being trained; it produces the Q-values
  used to pick actions.
- **Target network** — a periodically-frozen copy of the policy network used
  to compute the TD target. Two networks are used so the target does not move
  every step, which otherwise makes training unstable / divergent
  (`target_model_update=10000`).
- **Experience replay / replay memory** — transitions are stored in a
  `SequentialMemory` and sampled in random minibatches, breaking the
  correlation between consecutive frames and reusing data efficiently.
- **Epsilon-greedy exploration** — during training the agent takes a random
  action with probability `eps` (`EpsGreedyQPolicy`), otherwise the greedy
  one; at play time it is purely greedy (`GreedyQPolicy`).
- **Frame preprocessing** — RGB frames → 84x84 grayscale `uint8`; four frames
  are stacked (`window_length=4`) so the network can perceive motion.
- **Atari Breakout** — `ALE/Breakout-v5` from Gymnasium / ALE.
- **Gymnasium compatibility wrapper** — keras-rl2 targets the old Gym API, so
  `KerasRLWrapper` makes `reset()` return only the observation and `step()`
  return `(obs, reward, done, info)` with `done = terminated or truncated`.

## Files

| File | Description |
| --- | --- |
| `train.py` | Build the env + DQN, train with `EpsGreedyQPolicy`, save `policy.h5`. |
| `play.py` | Rebuild the same net, load `policy.h5`, play with `GreedyQPolicy` and `render_mode="human"`. |

## Usage

```
python train.py        # long; writes policy.h5
python play.py          # loads policy.h5 and renders an episode
```

`policy.h5` is a generated artifact and is not committed.
