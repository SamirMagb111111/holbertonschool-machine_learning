#!/usr/bin/env python3
"""Create the masks needed to train a Transformer."""
import tensorflow as tf


def create_masks(inputs, target):
    """Create the encoder, combined and decoder masks.

    Args:
        inputs: tf.Tensor of shape ``(batch_size, seq_len_in)``.
        target: tf.Tensor of shape ``(batch_size, seq_len_out)``.

    Returns:
        encoder_mask: ``(batch_size, 1, 1, seq_len_in)`` padding mask.
        combined_mask: ``(batch_size, 1, seq_len_out, seq_len_out)`` mask
            hiding both future positions and target padding.
        decoder_mask: ``(batch_size, 1, 1, seq_len_in)`` padding mask for the
            encoder-decoder attention block.
    """
    encoder_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    encoder_mask = encoder_mask[:, tf.newaxis, tf.newaxis, :]

    decoder_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    decoder_mask = decoder_mask[:, tf.newaxis, tf.newaxis, :]

    seq_len_out = tf.shape(target)[1]
    look_ahead_mask = 1 - tf.linalg.band_part(
        tf.ones((seq_len_out, seq_len_out)), -1, 0)
    target_padding_mask = tf.cast(tf.math.equal(target, 0), tf.float32)
    target_padding_mask = target_padding_mask[:, tf.newaxis, tf.newaxis, :]
    combined_mask = tf.maximum(look_ahead_mask, target_padding_mask)

    return encoder_mask, combined_mask, decoder_mask
