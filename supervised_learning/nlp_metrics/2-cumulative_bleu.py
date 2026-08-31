#!/usr/bin/env python3
"""Cumulative n-gram BLEU score."""
import math
from collections import Counter


def _ngrams(tokens, n):
    """Return the list of n-gram tuples of size ``n`` for ``tokens``."""
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def _modified_precision(references, sentence, n):
    """Return the clipped ``n``-gram precision of ``sentence``."""
    total = len(sentence) - n + 1
    if total <= 0:
        return 0

    candidate = Counter(_ngrams(sentence, n))
    max_ref = {}
    for reference in references:
        for gram, count in Counter(_ngrams(reference, n)).items():
            if count > max_ref.get(gram, 0):
                max_ref[gram] = count

    clipped = sum(min(count, max_ref.get(gram, 0))
                  for gram, count in candidate.items())
    return clipped / total


def cumulative_bleu(references, sentence, n):
    """Calculate the cumulative BLEU score up to ``n``-grams.

    All n-gram orders from 1 to ``n`` are weighted evenly.

    Args:
        references: list of reference translations, each a list of words.
        sentence: list of words containing the candidate sentence.
        n: largest n-gram order to use for evaluation.

    Returns:
        The cumulative BLEU score as a float (0 when any precision is 0).
    """
    if not sentence:
        return 0

    precisions = []
    for order in range(1, n + 1):
        precision = _modified_precision(references, sentence, order)
        if precision == 0:
            return 0
        precisions.append(precision)

    geometric_mean = math.exp(
        sum(math.log(p) for p in precisions) / n)

    c = len(sentence)
    r = min((len(reference) for reference in references),
            key=lambda length: (abs(length - c), length))
    brevity_penalty = 1 if c > r else math.exp(1 - r / c)

    return brevity_penalty * geometric_mean
