#!/usr/bin/env python3
"""N-gram BLEU score."""
import math
from collections import Counter


def _ngrams(tokens, n):
    """Return the list of n-gram tuples of size ``n`` for ``tokens``."""
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def ngram_bleu(references, sentence, n):
    """Calculate the BLEU score using only ``n``-gram precision.

    Args:
        references: list of reference translations, each a list of words.
        sentence: list of words containing the candidate sentence.
        n: size of the n-gram to use for evaluation.

    Returns:
        The n-gram BLEU score as a float (0 when no n-gram is possible).
    """
    if n <= 0 or len(sentence) < n:
        return 0

    candidate = Counter(_ngrams(sentence, n))
    max_ref = {}
    for reference in references:
        for gram, count in Counter(_ngrams(reference, n)).items():
            if count > max_ref.get(gram, 0):
                max_ref[gram] = count

    clipped = sum(min(count, max_ref.get(gram, 0))
                  for gram, count in candidate.items())
    precision = clipped / (len(sentence) - n + 1)

    c = len(sentence)
    r = min((len(reference) for reference in references),
            key=lambda length: (abs(length - c), length))
    brevity_penalty = 1 if c > r else math.exp(1 - r / c)

    return brevity_penalty * precision
