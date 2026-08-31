# RNNs

NumPy-only implementations of recurrent neural network cells and their
forward propagation.

## Concepts

- **Recurrent Neural Network (RNN)** — a network that processes a sequence
  one step at a time, reusing the same weights at every step and carrying
  information forward through a hidden state.
- **Hidden state** — the vector passed from one time step to the next; it is
  the network's memory of everything seen so far in the sequence.
- **GRU** — Gated Recurrent Unit: adds an update gate and a reset gate so the
  cell can learn how much past information to keep or discard, easing the
  vanishing-gradient problem.
- **LSTM** — Long Short-Term Memory: keeps a separate cell state regulated by
  forget, update and output gates, letting gradients flow over long
  sequences.
- **Deep RNN** — several recurrent layers stacked so the output hidden state
  of one layer is the input of the next at the same time step.
- **Vanishing gradients** — gradients shrink exponentially as they are
  back-propagated through many time steps, so early steps barely learn.
  Gated cells (GRU, LSTM) mitigate this.
- **Exploding gradients** — gradients grow exponentially through time,
  destabilizing training; usually handled with gradient clipping.

## Files

| File | Description |
| --- | --- |
| `0-rnn_cell.py` | `RNNCell` — a single vanilla RNN cell. |
| `1-rnn.py` | `rnn` — forward propagation for a simple RNN. |
| `2-gru_cell.py` | `GRUCell` — a gated recurrent unit cell. |
| `3-lstm_cell.py` | `LSTMCell` — a long short-term memory cell. |
| `4-deep_rnn.py` | `deep_rnn` — forward propagation for a deep (stacked) RNN. |
