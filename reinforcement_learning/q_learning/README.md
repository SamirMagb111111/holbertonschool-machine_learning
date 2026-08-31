# Q-learning

Train a tabular Q-learning agent to solve the FrozenLake environment.

## Concepts

- **Reinforcement Learning** — an **agent** learns by acting in an
  **environment** and adjusting its behaviour from the **reward** signal it
  receives, without labelled examples.
- **State** — the environment's situation the agent observes (here, the tile
  index on the lake).
- **Action** — a choice the agent makes (left, down, right, up).
- **Reward** — scalar feedback after an action (goal `+1`; in this project a
  hole is set to `-1`; every other step `0`).
- **Markov Decision Process (MDP)** — the formal model: states, actions,
  transition probabilities and rewards, where the next state depends only on
  the current state and action.
- **Policy** — the agent's rule for choosing actions given a state.
- **Value function** — expected cumulative future reward from a state (or
  state-action pair) under a policy.
- **Q-table** — a `(states x actions)` array holding the estimated value of
  taking each action in each state.
- **Epsilon-greedy** — with probability `epsilon` take a random action
  (**exploration**), otherwise take the best known action
  (**exploitation**); `epsilon` decays over training so the agent explores
  early and exploits later.
- **Q-learning** — off-policy update:
  `Q(s,a) += alpha * (r + gamma * max_a' Q(s',a') - Q(s,a))`.
- **Learning rate (`alpha`)** — how much each update moves the estimate.
- **Discount factor (`gamma`)** — how much future rewards count versus
  immediate ones.
- **FrozenLake** — a grid world (`S` start, `F` frozen, `H` hole, `G` goal);
  slippery mode makes movement stochastic.

## Files

| File | Description |
| --- | --- |
| `0-load_env.py` | `load_frozen_lake` — load FrozenLake (`render_mode="ansi"`). |
| `1-q_init.py` | `q_init` — zero-initialize the Q-table. |
| `2-epsilon_greedy.py` | `epsilon_greedy` — epsilon-greedy action selection. |
| `3-q_learning.py` | `train` — the Q-learning training loop. |
| `4-play.py` | `play` — run a greedy episode and return the renders. |
