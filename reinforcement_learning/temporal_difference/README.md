# Temporal Difference

Value-estimation algorithms for reinforcement learning, applied to
`FrozenLake8x8-v1`.

## Concepts

- **Reinforcement Learning** — an agent learns a behaviour from reward while
  interacting with an environment.
- **Monte Carlo** — estimate a state's value by averaging the *full* return
  observed after visiting it; the update happens only once the episode ends.
- **Temporal Difference (TD)** — update an estimate from the immediate reward
  plus the current estimate of the next state, without waiting for the
  episode to finish.
- **Bootstrapping** — using an existing estimate (`V[next_state]`) inside the
  update target; Monte Carlo does not bootstrap, TD does.
- **n-step TD** — a middle ground: bootstrap after `n` real rewards.
- **TD(lambda)** — averages every n-step return with weights `(1-λ)λ^(n-1)`,
  implemented online with **eligibility traces**.
- **Lambda (`λ`)** — the trace decay: `λ=0` is one-step TD, `λ=1` is close to
  Monte Carlo.
- **Eligibility trace** — a per-state (or per-state-action) memory of how
  recently and how often it was visited; every TD error is applied to all
  states in proportion to their trace (`e[s] += 1`, then `e *= γλ`).
- **SARSA** — on-policy TD control on `Q(s, a)`; the target uses the action
  actually taken next: `r + γ·Q(s', a')`.
- **SARSA(lambda)** — SARSA with eligibility traces over state-action pairs.
- **On-policy vs off-policy** — SARSA learns the value of the policy it
  follows (on-policy); Q-learning learns the greedy policy while following
  another (off-policy).
- **Epsilon-greedy** — explore a random action with probability `ε`,
  otherwise exploit `argmax Q`; `ε` decays over training.
- **Alpha (`α`)** — learning rate. **Gamma (`γ`)** — discount factor for
  future rewards.

## Files

| File | Description |
| --- | --- |
| `0-monte_carlo.py` | `monte_carlo(env, V, policy, ...)`. |
| `1-td_lambtha.py` | `td_lambtha(env, V, policy, lambtha, ...)`. |
| `2-sarsa_lambtha.py` | `sarsa_lambtha(env, Q, lambtha, ...)`. |
