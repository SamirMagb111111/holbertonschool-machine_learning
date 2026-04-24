#!/usr/bin/env python3
"""Module that concatenates matrices using numpy"""

import numpy as np


def np_cat(mat1, mat2, axis=0):
    """Returns concatenated matrix along given axis"""
    return np.concatenate((mat1, mat2), axis=axis)
