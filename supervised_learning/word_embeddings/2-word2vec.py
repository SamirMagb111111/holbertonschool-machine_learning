#!/usr/bin/env python3
"""Train a gensim Word2Vec model."""
import gensim


def word2vec_model(sentences, vector_size=100, min_count=5,
                   window=5, negative=5, cbow=True, epochs=5,
                   seed=0, workers=1):
    """Build and train a gensim Word2Vec model.

    Args:
        sentences: list of tokenized sentences to train on.
        vector_size: dimensionality of the embedding layer.
        min_count: minimum number of occurrences of a word for use.
        window: maximum distance between the current and predicted word.
        negative: size of negative sampling.
        cbow: True for CBOW, False for Skip-gram.
        epochs: number of training epochs.
        seed: seed for the random number generator.
        workers: number of worker threads to train the model.

    Returns:
        The trained gensim Word2Vec model.
    """
    model = gensim.models.word2vec.Word2Vec(
        seed=seed,
        sentences=sentences,
        vector_size=vector_size,
        min_count=min_count,
        window=window,
        negative=negative,
        sg=int(not cbow),
        epochs=epochs,
        workers=workers)
    model.build_vocab(sentences)
    model.train(sentences, total_examples=model.corpus_count,
                epochs=model.epochs)
    return model
