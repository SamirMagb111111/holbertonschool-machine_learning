#!/usr/bin/env python3
"""Build a neural network with the Keras Functional API."""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """Build a Keras model with the Functional API.

    Args:
        nx: number of input features.
        layers: list with the number of nodes in each layer.
        activations: list with the activation function of each layer.
        lambtha: L2 regularization parameter.
        keep_prob: probability that a node is kept for dropout.

    Returns:
        The Keras model.
    """
    inputs = K.Input(shape=(nx,))
    x = inputs
    for i in range(len(layers)):
        x = K.layers.Dense(
            layers[i], activation=activations[i],
            kernel_regularizer=K.regularizers.l2(lambtha))(x)
        if i != len(layers) - 1:
            x = K.layers.Dropout(1 - keep_prob)(x)
    return K.Model(inputs=inputs, outputs=x)
