#!/usr/bin/env python3
"""TF-IDF embedding matrix."""
from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(sentences, vocab=None):
    """Create a TF-IDF embedding matrix.

    Args:
        sentences: list of sentences to analyze.
        vocab: list of vocabulary words to use; if None the vocabulary is
            built from ``sentences``.

    Returns:
        embeddings: numpy.ndarray of shape (s, f) with the TF-IDF weights.
        features: numpy.ndarray with the features used for the embeddings.
    """
    vectorizer = TfidfVectorizer(vocabulary=vocab)
    embeddings = vectorizer.fit_transform(sentences).toarray()
    features = vectorizer.get_feature_names_out()
    return embeddings, features
