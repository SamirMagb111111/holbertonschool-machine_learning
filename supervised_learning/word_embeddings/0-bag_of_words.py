#!/usr/bin/env python3
"""Bag of words embedding matrix."""
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


def bag_of_words(sentences, vocab=None):
    """Create a bag of words embedding matrix.

    Args:
        sentences: list of sentences to analyze.
        vocab: list of vocabulary words to use; if None the vocabulary is
            built from ``sentences``.

    Returns:
        embeddings: numpy.ndarray of shape (s, f) with the word counts.
        features: numpy.ndarray with the features used for the embeddings.
    """
    vectorizer = CountVectorizer(vocabulary=vocab)
    embeddings = vectorizer.fit_transform(sentences).toarray()
    features = vectorizer.get_feature_names_out()
    return embeddings, features
