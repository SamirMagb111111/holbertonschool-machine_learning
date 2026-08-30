#!/usr/bin/env python3
"""Projection block module."""

from tensorflow import keras as K


def projection_block(A_prev, filters, s=2):
    """Build a projection block."""
    F11, F3, F12 = filters

    X = K.layers.Conv2D(
        F11,
        (1, 1),
        strides=(s, s),
        padding='same',
        kernel_initializer=K.initializers.he_normal(seed=0)
    )(A_prev)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    X = K.layers.Conv2D(
        F3,
        (3, 3),
        padding='same',
        kernel_initializer=K.initializers.he_normal(seed=0)
    )(X)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    X = K.layers.Conv2D(
        F12,
        (1, 1),
        padding='same',
        kernel_initializer=K.initializers.he_normal(seed=0)
    )(X)
    X = K.layers.BatchNormalization(axis=3)(X)

    shortcut = K.layers.Conv2D(
        F12,
        (1, 1),
        strides=(s, s),
        padding='same',
        kernel_initializer=K.initializers.he_normal(seed=0)
    )(A_prev)
    shortcut = K.layers.BatchNormalization(axis=3)(shortcut)

    X = K.layers.Add()([X, shortcut])
    X = K.layers.Activation('relu')(X)

    return X
