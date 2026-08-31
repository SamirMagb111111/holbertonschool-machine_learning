#!/usr/bin/env python3
"""Transformer encoder block."""
import tensorflow as tf

MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class EncoderBlock(tf.keras.layers.Layer):
    """A single block of the Transformer encoder."""

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """Initialize the block.

        Args:
            dm: model dimensionality.
            h: number of attention heads.
            hidden: number of units in the feed-forward hidden layer.
            drop_rate: dropout rate.
        """
        super().__init__()
        self.mha = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation="relu")
        self.dense_output = tf.keras.layers.Dense(dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask=None):
        """Run the encoder block.

        Args:
            x: input tensor, ``(batch, input_seq_len, dm)``.
            training: whether the model is in training mode.
            mask: optional padding mask.

        Returns:
            Tensor of shape ``(batch, input_seq_len, dm)``.
        """
        attention, _ = self.mha(x, x, x, mask)
        attention = self.dropout1(attention, training=training)
        out1 = self.layernorm1(x + attention)

        ffn = self.dense_hidden(out1)
        ffn = self.dense_output(ffn)
        ffn = self.dropout2(ffn, training=training)
        return self.layernorm2(out1 + ffn)
