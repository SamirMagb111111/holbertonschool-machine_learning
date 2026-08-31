#!/usr/bin/env python3
"""Unigram BLEU score."""
import math
from collections import Counter


def uni_bleu(references, sentence):
    """Calculate the unigram BLEU score for a sentence.

    Args:
        references: list of reference translations, each a list of words.
        sentence: list of words containing the candidate sentence.

    Returns:
        The unigram BLEU score as a float.
    """
    if not sentence:
        return 0

    candidate = Counter(sentence)
    max_ref = {}
    for reference in references:
        for word, count in Counter(reference).items():
            if count > max_ref.get(word, 0):
                max_ref[word] = count

    clipped = sum(min(count, max_ref.get(word, 0))
                  for word, count in candidate.items())
    precision = clipped / len(sentence)

    c = len(sentence)
    r = min((len(reference) for reference in references),
            key=lambda length: (abs(length - c), length))
    brevity_penalty = 1 if c > r else math.exp(1 - r / c)

    return brevity_penalty * precision
