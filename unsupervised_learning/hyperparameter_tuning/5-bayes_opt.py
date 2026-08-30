#!/usr/bin/env python3
"""Bayesian Optimization module."""

import numpy as np
from scipy.stats import norm

GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """Performs Bayesian optimization on a Gaussian process."""

    def __init__(
            self, f, X_init, Y_init, bounds, ac_samples,
            l=1, sigma_f=1, xsi=0.01, minimize=True):
        """Initialize Bayesian Optimization."""
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)

        self.X_s = np.linspace(
            bounds[0],
            bounds[1],
            ac_samples
        ).reshape(-1, 1)

        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """Calculate the next best sample using Expected Improvement."""
        mu, sigma = self.gp.predict(self.X_s)
        sigma = sigma.reshape(-1)

        if self.minimize:
            optimum = np.min(self.gp.Y)
            improvement = optimum - mu - self.xsi
        else:
            optimum = np.max(self.gp.Y)
            improvement = mu - optimum - self.xsi

        Z = np.zeros_like(improvement)
        mask = sigma > 0
        Z[mask] = improvement[mask] / sigma[mask]

        EI = np.zeros_like(improvement)
        EI[mask] = (
            improvement[mask] * norm.cdf(Z[mask]) +
            sigma[mask] * norm.pdf(Z[mask])
        )

        X_next = self.X_s[np.argmax(EI)]

        return X_next, EI

    def optimize(self, iterations=100):
        """Optimize the black-box function."""
        for _ in range(iterations):
            X_next, _ = self.acquisition()

            if np.any(np.isclose(self.gp.X, X_next)):
                break

            Y_next = self.f(X_next).reshape(-1, 1)
            self.gp.update(X_next.reshape(-1, 1), Y_next)

        if self.minimize:
            index = np.argmin(self.gp.Y)
        else:
            index = np.argmax(self.gp.Y)

        X_opt = self.gp.X[index]
        Y_opt = self.gp.Y[index]

        return X_opt, Y_opt
