#!/usr/bin/env python3
"""Convert a gensim Word2Vec model to a Keras Embedding layer."""
import tensorflow as tf


def gensim_to_keras(model):
    """Build a trainable Keras Embedding layer from a Word2Vec model.

    Args:
        model: a trained gensim Word2Vec model.

    Returns:
        A trainable ``tf.keras.layers.Embedding`` layer initialized with the
        Word2Vec weights.
    """
    weights = model.wv.vectors
    return tf.keras.layers.Embedding(
        input_dim=weights.shape[0],
        output_dim=weights.shape[1],
        weights=[weights],
        trainable=True)
