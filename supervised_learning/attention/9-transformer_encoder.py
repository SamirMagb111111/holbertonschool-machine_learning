#!/usr/bin/env python3
"""Transformer encoder."""
import tensorflow as tf

positional_encoding = __import__('4-positional_encoding').positional_encoding
EncoderBlock = __import__('7-transformer_encoder_block').EncoderBlock


class Encoder(tf.keras.layers.Layer):
    """The full Transformer encoder: embedding + N encoder blocks."""

    def __init__(self, N, dm, h, hidden, input_vocab, max_seq_len,
                 drop_rate=0.1):
        """Initialize the encoder.

        Args:
            N: number of encoder blocks.
            dm: model dimensionality.
            h: number of attention heads.
            hidden: number of units in the feed-forward hidden layer.
            input_vocab: size of the input vocabulary.
            max_seq_len: maximum input sequence length.
            drop_rate: dropout rate.
        """
        super().__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(input_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = [EncoderBlock(dm, h, hidden, drop_rate)
                       for _ in range(N)]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask):
        """Encode a batch of token sequences.

        Args:
            x: token indices, ``(batch, input_seq_len)``.
            training: whether the model is in training mode.
            mask: padding mask applied inside every block.

        Returns:
            Tensor of shape ``(batch, input_seq_len, dm)``.
        """
        seq_len = tf.shape(x)[1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x += tf.cast(self.positional_encoding[:seq_len], x.dtype)
        x = self.dropout(x, training=training)

        for block in self.blocks:
            x = block(x, training, mask)
        return x
