#!/usr/bin/env python3
"""Module for polynomial integration"""


def poly_integral(poly, C=0):
    """Calculates integral of polynomial"""
    if not isinstance(poly, list) or len(poly) == 0:
        return None
    if not isinstance(C, int):
        return None

    result = [C]
    for i, coef in enumerate(poly):
        if coef is None:
            return None
        result.append(coef / (i + 1))

    # remove trailing zeros
    while len(result) > 1 and result[-1] == 0:
        result.pop()

    # convert floats like 5.0 → 5
    for i in range(len(result)):
        if result[i] == int(result[i]):
            result[i] = int(result[i])

    return result
