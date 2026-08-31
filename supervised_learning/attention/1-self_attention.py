#!/usr/bin/env python3
"""Additive (Bahdanau) self attention for the RNN decoder."""
import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """Compute an attention context vector from decoder and encoder states."""

    def __init__(self, units):
        """Initialize the attention layer.

        Args:
            units: number of hidden units in the alignment layers.
        """
        super().__init__()
        self.W = tf.keras.layers.Dense(units)
        self.U = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, s_prev, hidden_states):
        """Compute the context vector and attention weights.

        Args:
            s_prev: previous decoder hidden state, ``(batch, units)``.
            hidden_states: encoder outputs,
                ``(batch, input_seq_len, units)``.

        Returns:
            context: ``(batch, units)``.
            weights: ``(batch, input_seq_len, 1)``.
        """
        s_prev = tf.expand_dims(s_prev, 1)
        score = self.V(tf.nn.tanh(self.W(s_prev) + self.U(hidden_states)))
        weights = tf.nn.softmax(score, axis=1)
        context = tf.reduce_sum(weights * hidden_states, axis=1)
        return context, weights
