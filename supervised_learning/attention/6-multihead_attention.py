#!/usr/bin/env python3
"""Multi-head attention."""
import tensorflow as tf

sdp_attention = __import__('5-sdp_attention').sdp_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    """Perform multi-head attention."""

    def __init__(self, dm, h):
        """Initialize the layer.

        Args:
            dm: model dimensionality (must be divisible by ``h``).
            h: number of attention heads.
        """
        super().__init__()
        self.h = h
        self.dm = dm
        self.depth = dm // h
        self.Wq = tf.keras.layers.Dense(dm)
        self.Wk = tf.keras.layers.Dense(dm)
        self.Wv = tf.keras.layers.Dense(dm)
        self.linear = tf.keras.layers.Dense(dm)

    def split_heads(self, x, batch):
        """Split the last dimension of ``x`` into ``(h, depth)``.

        Args:
            x: tensor of shape ``(batch, seq_len, dm)``.
            batch: dynamic batch size.

        Returns:
            Tensor of shape ``(batch, h, seq_len, depth)``.
        """
        x = tf.reshape(x, (batch, -1, self.h, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, Q, K, V, mask):
        """Apply multi-head attention.

        Args:
            Q: query tensor, ``(batch, seq_len_q, dk)``.
            K: key tensor, ``(batch, seq_len_v, dk)``.
            V: value tensor, ``(batch, seq_len_v, dv)``.
            mask: optional attention mask.

        Returns:
            output: ``(batch, seq_len_q, dm)``.
            weights: ``(batch, h, seq_len_q, seq_len_v)``.
        """
        batch = tf.shape(Q)[0]

        Q = self.split_heads(self.Wq(Q), batch)
        K = self.split_heads(self.Wk(K), batch)
        V = self.split_heads(self.Wv(V), batch)

        scaled_attention, weights = sdp_attention(Q, K, V, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch, -1, self.dm))

        output = self.linear(concat_attention)
        return output, weights
