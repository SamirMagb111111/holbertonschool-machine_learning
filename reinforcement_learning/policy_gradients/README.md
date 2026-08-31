# Policy Gradients

Train a CartPole agent with the **Monte-Carlo policy gradient**
(REINFORCE) algorithm, using only NumPy.

## Concepts

- **Reinforcement Learning** — an agent learns a behaviour from reward while
  interacting with an environment.
- **Policy** — the rule `pi(a | s)` that maps a state to a distribution over
  actions.
- **Policy-based learning** — optimize the policy parameters directly,
  instead of learning a value function and acting greedily on it.
- **Policy gradient** — follow the gradient of expected return with respect
  to the policy parameters: `grad J = E[ grad log pi(a|s) * G ]`.
- **Softmax policy** — a linear score `s = state @ weight` passed through a
  numerically stable softmax; `grad log pi = stateᵀ (one_hot(a) - probs)`.
- **Monte-Carlo policy gradient** — run a whole episode, then update the
  weights using the actual (sampled) discounted returns — no bootstrapping,
  no critic, no baseline.
- **Discounted return** — `G_t = R_t + γ R_{t+1} + γ² R_{t+2} + ...`,
  computed backwards as `G = reward + γ·G`.
- **Learning rate `alpha`** — step size of `weight += alpha · G · gradient`.
- **Discount factor `gamma`** — how much future rewards count.
- **CartPole** — `CartPole-v1`: balance a pole; reward `+1` per step,
  episode truncates at 500 steps.
- **Exploration** — comes for free from sampling actions from the stochastic
  softmax policy (`np.random.choice`), not from epsilon-greedy.

## Files

| File | Description |
| --- | --- |
| `policy_gradient.py` | `policy(matrix, weight)` and `policy_gradient(state, weight)`. |
| `train.py` | `train(env, nb_episodes, alpha=0.000045, gamma=0.98, show_result=False)`. |
