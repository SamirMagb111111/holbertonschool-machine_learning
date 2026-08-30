#!/usr/bin/env python3
"""Gaussian Process module."""

import numpy as np


class GaussianProcess:
    """Represents a noiseless 1D Gaussian process."""

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """Initialize the Gaussian Process."""
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        """Calculate the RBF covariance kernel matrix."""
        sqdist = (X1 - X2.T) ** 2

        return (self.sigma_f ** 2) * np.exp(
            -sqdist / (2 * self.l ** 2)
        )

    def predict(self, X_s):
        """Predict the mean and variance for sample points."""
        K_s = self.kernel(self.X, X_s)
        K_ss = self.kernel(X_s, X_s)

        K_inv = np.linalg.inv(self.K)

        mu = K_s.T.dot(K_inv).dot(self.Y)

        sigma = K_ss - K_s.T.dot(K_inv).dot(K_s)

        return mu.reshape(-1), np.diag(sigma)
