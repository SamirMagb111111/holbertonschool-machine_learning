#!/usr/bin/env python3
"""Positional encoding for a Transformer."""
import numpy as np


def positional_encoding(max_seq_len, dm):
    """Compute the positional encoding matrix.

    Args:
        max_seq_len: maximum sequence length.
        dm: model depth.

    Returns:
        numpy.ndarray of shape ``(max_seq_len, dm)`` with the positional
        encoding vectors.
    """
    positions = np.arange(max_seq_len)[:, np.newaxis]
    indices = np.arange(dm)[np.newaxis, :]
    angle_rates = 1 / np.power(10000, (2 * (indices // 2) / dm))
    angles = positions * angle_rates

    pe = np.zeros((max_seq_len, dm))
    pe[:, 0::2] = np.sin(angles[:, 0::2])
    pe[:, 1::2] = np.cos(angles[:, 1::2])
    return pe
