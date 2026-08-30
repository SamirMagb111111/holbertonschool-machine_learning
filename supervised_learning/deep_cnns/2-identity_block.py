#!/usr/bin/env python3
"""Identity block module."""

from tensorflow import keras as K


def identity_block(A_prev, filters):
    """Build an identity block."""
    F11, F3, F12 = filters
    initializer = K.initializers.he_normal(seed=0)

    X = K.layers.Conv2D(
        filters=F11,
        kernel_size=(1, 1),
        padding='same',
        kernel_initializer=initializer
    )(A_prev)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    X = K.layers.Conv2D(
        filters=F3,
        kernel_size=(3, 3),
        padding='same',
        kernel_initializer=initializer
    )(X)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    X = K.layers.Conv2D(
        filters=F12,
        kernel_size=(1, 1),
        padding='same',
        kernel_initializer=initializer
    )(X)
    X = K.layers.BatchNormalization(axis=3)(X)

    X = K.layers.Add()([X, A_prev])
    X = K.layers.Activation('relu')(X)

    return X
