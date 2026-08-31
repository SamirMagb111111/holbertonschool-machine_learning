#!/usr/bin/env python3
"""Transformer decoder block."""
import tensorflow as tf

MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class DecoderBlock(tf.keras.layers.Layer):
    """A single block of the Transformer decoder."""

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """Initialize the block.

        Args:
            dm: model dimensionality.
            h: number of attention heads.
            hidden: number of units in the feed-forward hidden layer.
            drop_rate: dropout rate.
        """
        super().__init__()
        self.mha1 = MultiHeadAttention(dm, h)
        self.mha2 = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation="relu")
        self.dense_output = tf.keras.layers.Dense(dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)
        self.dropout3 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask,
             padding_mask):
        """Run the decoder block.

        Args:
            x: input tensor, ``(batch, target_seq_len, dm)``.
            encoder_output: ``(batch, input_seq_len, dm)``.
            training: whether the model is in training mode.
            look_ahead_mask: mask for the first attention block.
            padding_mask: mask for the second attention block.

        Returns:
            Tensor of shape ``(batch, target_seq_len, dm)``.
        """
        attention1, _ = self.mha1(x, x, x, look_ahead_mask)
        attention1 = self.dropout1(attention1, training=training)
        out1 = self.layernorm1(x + attention1)

        attention2, _ = self.mha2(out1, encoder_output, encoder_output,
                                  padding_mask)
        attention2 = self.dropout2(attention2, training=training)
        out2 = self.layernorm2(out1 + attention2)

        ffn = self.dense_hidden(out2)
        ffn = self.dense_output(ffn)
        ffn = self.dropout3(ffn, training=training)
        return self.layernorm3(out2 + ffn)
