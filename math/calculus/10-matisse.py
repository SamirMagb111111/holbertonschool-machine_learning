#!/usr/bin/env python3
"""Module for polynomial derivative"""


def poly_derivative(poly):
    """Calculates the derivative of a polynomial"""
    if (not isinstance(poly, list) or len(poly) == 0):
        return None

    if any(not isinstance(x, (int, float)) for x in poly):
        return None

    if len(poly) == 1:
        return [0]

    result = []

    for i in range(1, len(poly)):
        result.append(i * poly[i])

    return result
