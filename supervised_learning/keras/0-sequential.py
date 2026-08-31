#!/usr/bin/env python3
"""Build a neural network with the Keras Sequential API."""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """Build a Keras Sequential model.

    Args:
        nx: number of input features.
        layers: list with the number of nodes in each layer.
        activations: list with the activation function of each layer.
        lambtha: L2 regularization parameter.
        keep_prob: probability that a node is kept for dropout.

    Returns:
        The Keras model.
    """
    model = K.Sequential()
    for i in range(len(layers)):
        kwargs = {"activation": activations[i],
                  "kernel_regularizer": K.regularizers.l2(lambtha)}
        if i == 0:
            kwargs["input_shape"] = (nx,)
        model.add(K.layers.Dense(layers[i], **kwargs))
        if i != len(layers) - 1:
            model.add(K.layers.Dropout(1 - keep_prob))
    return model
