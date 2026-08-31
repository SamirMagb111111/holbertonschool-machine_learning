#!/usr/bin/env python3
"""Train a gensim FastText model."""
import gensim


def fasttext_model(sentences, vector_size=100, min_count=5,
                   negative=5, window=5, cbow=True, epochs=5,
                   seed=0, workers=1):
    """Build and train a gensim FastText model.

    Args:
        sentences: list of tokenized sentences to train on.
        vector_size: dimensionality of the embedding layer.
        min_count: minimum number of occurrences of a word for use.
        negative: size of negative sampling.
        window: maximum distance between the current and predicted word.
        cbow: True for CBOW, False for Skip-gram.
        epochs: number of training epochs.
        seed: seed for the random number generator.
        workers: number of worker threads to train the model.

    Returns:
        The trained gensim FastText model.
    """
    model = gensim.models.fasttext.FastText(
        sentences=sentences,
        vector_size=vector_size,
        min_count=min_count,
        window=window,
        negative=negative,
        sg=int(not cbow),
        epochs=epochs,
        seed=seed,
        workers=workers)
    model.build_vocab(corpus_iterable=sentences)
    model.train(corpus_iterable=sentences, total_examples=len(sentences),
                epochs=epochs)
    return model
