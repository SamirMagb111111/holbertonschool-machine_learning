#!/usr/bin/env python3
"""Module that adds two 2D matrices element-wise"""


def add_matrices2D(mat1, mat2):
    """Returns a new 2D matrix with element-wise addition"""
    if len(mat1) != len(mat2):
        return None
    for i in range(len(mat1)):
        if len(mat1[i]) != len(mat2[i]):
            return None
    return [[mat1[i][j] + mat2[i][j]
             for j in range(len(mat1[i]))]
            for i in range(len(mat1))]
