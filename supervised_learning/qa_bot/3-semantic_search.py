#!/usr/bin/env python3
"""Semantic search over a corpus with the Universal Sentence Encoder."""
import os

import numpy as np
import tensorflow_hub as hub


embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")


def semantic_search(corpus_path, sentence):
    """Return the full text of the corpus document closest to ``sentence``.

    Args:
        corpus_path: path to the directory holding the reference documents.
        sentence: the query sentence to match against the corpus.

    Returns:
        String with the full text of the semantically most similar document.
    """
    documents = []
    for name in sorted(os.listdir(corpus_path)):
        path = os.path.join(corpus_path, name)
        if not os.path.isfile(path) or not name.endswith(".md"):
            continue
        with open(path, "r", encoding="utf-8") as handle:
            documents.append(handle.read())

    embeddings = embed([sentence] + documents)
    query_embedding = embeddings[0]
    document_embeddings = embeddings[1:]
    similarities = np.inner(query_embedding, document_embeddings)
    return documents[int(np.argmax(similarities))]
