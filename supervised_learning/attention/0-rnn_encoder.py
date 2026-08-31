#!/usr/bin/env python3
"""RNN encoder for machine translation."""
import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """Encode a sequence of tokens with an embedding layer and a GRU."""

    def __init__(self, vocab, embedding, units, batch):
        """Initialize the encoder.

        Args:
            vocab: size of the input vocabulary.
            embedding: dimensionality of the embedding vectors.
            units: number of hidden units in the GRU cell.
            batch: batch size.
        """
        super().__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab,
                                                   output_dim=embedding)
        self.gru = tf.keras.layers.GRU(
            units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer="glorot_uniform")

    def initialize_hidden_state(self):
        """Return a tensor of zeros of shape ``(batch, units)``."""
        return tf.zeros((self.batch, self.units))

    def call(self, x, initial):
        """Run the encoder over a batch of token sequences.

        Args:
            x: tensor of shape ``(batch, input_seq_len)`` of token indices.
            initial: initial GRU hidden state, ``(batch, units)``.

        Returns:
            outputs: ``(batch, input_seq_len, units)``.
            hidden: ``(batch, units)``.
        """
        x = self.embedding(x)
        outputs, hidden = self.gru(x, initial_state=initial)
        return outputs, hidden
