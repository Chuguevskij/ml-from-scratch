import pandas as pd
import numpy as np
import random

from metrics.classification import *
from metrics.classification import METRIC_MAP


class MyLogReg:
    def __init__(self, n_iter=50, learning_rate=0.1, metric=None,
                 eps=1e-15, weights=None):
        self._n_iter = n_iter
        self._learning_rate = learning_rate
        self._metric = metric
        self.eps = eps
        self._weights = weights
        self._X = None
        self.score = None

    def add_intercept(self, X):
        """Add intercept to feature array."""
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate([intercept, X], axis=1)

    def sigmoid(self, z):
        """Get logits."""
        return 1 / (1 + np.exp(-z))

    def gradient(self, X, y, y_hat):
        """Get a vector with partirial derivatives."""
        return (y_hat - y) @ X / X.shape[0]

    def cost_function(self, y, y_hat):
        """Get loss."""
        return -np.mean(
            y * np.log(y_hat + self.eps) +
            (1 - y) * np.log(1 - y_hat + self.eps)
        )

    def fit(self, X, y, verbose=False):
        """Train the model"""
        self._X = self.add_intercept(X)
        self._weights = np.ones(self._X.shape[1])

        for i in range(1, self._n_iter+1):
            z = self._X @ self._weights
            y_hat = self.sigmoid(z)
            loss = self.cost_function(y, y_hat)
            dw = self.gradient(self._X, y, y_hat)
            self._weights -= self._learning_rate * dw

            # get metric
            if self._metric:
                metric_func = METRIC_MAP.get(self._metric)
                if metric_func:
                    self.score = metric_func(y, y_hat)
                else:
                    available = list(METRIC_MAP.keys())
                    raise ValueError(
                        f"Unknown metric: {self._metric}. "
                        f"Available metrics: {available}"
                    )
                verbose_score = f' | {self._metric}: {self.score}'
            else:
                verbose_score = ''

            # display information
            if verbose:
                # for start iter
                if i == 1:
                    print(f'start | loss: {loss}' + verbose_score)
                # for every i iter
                if i % verbose == 0:
                    print(f'{i} | loss: {loss}' + verbose_score)

    def predict_proba(self, X):
        """Return probabilities of class."""
        X = self.add_intercept(X)
        return self.sigmoid(X @ self._weights)

    def predict(self, X, threshold=0.5):
        """Return classes."""
        return np.where(self.predict_proba(X) >= threshold, 1, 0)

    def get_best_score(self):
        """Get metric for trained model."""
        return self.score

    def get_coef(self):
        """Get trained weights except the bias."""
        return self._weights[1:]

    def __str__(self):
        return (
            f"{__class__.__name__} class: "
            f"n_iter={self._n_iter}, "
            f"learning_rate={self._learning_rate}"
        )
