# Deep Convolutional Neural Networks

This project covers advanced convolutional neural network architectures
using TensorFlow and Keras.

## Learning Objectives

- What is a skip connection
- What is a bottleneck layer
- What is the Inception Network
- What is ResNet
- What is ResNeXt
- What is DenseNet
- How to reproduce neural network architectures from research papers

## Requirements

- Python 3.9
- TensorFlow 2.15
- NumPy 1.25.2
- Pycodestyle 2.11.1
- Only `from tensorflow import keras as K` may be imported unless stated otherwise

## Tasks

### Identity Block

Implements a ResNet identity block using:

- 1x1 convolution
- 3x3 convolution
- 1x1 convolution
- Batch normalization
- ReLU activation
- Skip connection
- He normal initialization with seed 0
