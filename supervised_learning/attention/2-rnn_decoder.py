#!/usr/bin/env python3
"""RNN decoder with additive attention for machine translation."""
import tensorflow as tf

SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """Decode one token at a time using an attention context vector."""

    def __init__(self, vocab, embedding, units, batch):
        """Initialize the decoder.

        Args:
            vocab: size of the target vocabulary.
            embedding: dimensionality of the embedding vectors.
            units: number of hidden units in the GRU cell.
            batch: batch size.
        """
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab,
                                                   output_dim=embedding)
        self.gru = tf.keras.layers.GRU(
            units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer="glorot_uniform")
        self.F = tf.keras.layers.Dense(vocab)
        self.attention = SelfAttention(units)

    def call(self, x, s_prev, hidden_states):
        """Decode a single step.

        Args:
            x: previous target token, ``(batch, 1)``.
            s_prev: previous decoder hidden state, ``(batch, units)``.
            hidden_states: encoder outputs,
                ``(batch, input_seq_len, units)``.

        Returns:
            y: vocabulary logits, ``(batch, vocab)``.
            s: new decoder hidden state, ``(batch, units)``.
        """
        context, _ = self.attention(s_prev, hidden_states)
        x = self.embedding(x)
        context = tf.expand_dims(context, 1)
        x = tf.concat([context, x], axis=-1)
        output, s = self.gru(x, initial_state=s_prev)
        output = tf.squeeze(output, axis=1)
        y = self.F(output)
        return y, s
