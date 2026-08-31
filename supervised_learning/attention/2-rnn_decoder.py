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
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(
            units,
            recurrent_initializer="glorot_uniform",
            return_sequences=True,
            return_state=True)
        self.F = tf.keras.layers.Dense(vocab)

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
        attention = SelfAttention(s_prev.shape[1])
        context, weights = attention(s_prev, hidden_states)

        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context, 1), x], axis=-1)

        outputs, s = self.gru(x)
        outputs = tf.reshape(outputs, (-1, outputs.shape[2]))
        y = self.F(outputs)
        return y, s
