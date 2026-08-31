#!/usr/bin/env python3
"""Transformer decoder."""
import tensorflow as tf

positional_encoding = __import__('4-positional_encoding').positional_encoding
DecoderBlock = __import__('8-transformer_decoder_block').DecoderBlock


class Decoder(tf.keras.layers.Layer):
    """The full Transformer decoder: embedding + N decoder blocks."""

    def __init__(self, N, dm, h, hidden, target_vocab, max_seq_len,
                 drop_rate=0.1):
        """Initialize the decoder.

        Args:
            N: number of decoder blocks.
            dm: model dimensionality.
            h: number of attention heads.
            hidden: number of units in the feed-forward hidden layer.
            target_vocab: size of the target vocabulary.
            max_seq_len: maximum target sequence length.
            drop_rate: dropout rate.
        """
        super().__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(target_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = [DecoderBlock(dm, h, hidden, drop_rate)
                       for _ in range(N)]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask,
             padding_mask):
        """Decode a batch of target token sequences.

        Args:
            x: target token indices, ``(batch, target_seq_len)``.
            encoder_output: ``(batch, input_seq_len, dm)``.
            training: whether the model is in training mode.
            look_ahead_mask: mask for the masked self-attention block.
            padding_mask: mask for the encoder-decoder attention block.

        Returns:
            Tensor of shape ``(batch, target_seq_len, dm)``.
        """
        seq_len = tf.shape(x)[1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x += tf.cast(self.positional_encoding[:seq_len], x.dtype)
        x = self.dropout(x, training=training)

        for block in self.blocks:
            x = block(x, encoder_output, training, look_ahead_mask,
                      padding_mask)
        return x
