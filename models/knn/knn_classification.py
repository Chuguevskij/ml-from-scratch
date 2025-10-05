import numpy as np
import pandas as pd

from metrics.distance import *


class MyKNNClf():
    def __init__(self, k=3, metric='euclidean', weight='uniform'):
        self.k = k
        self.train_size = None
        self.X = None
        self.y = None
        self.classes_ = None
        self.weight = weight

        # Map string to actual function
        metric_map = {
            'euclidean': euclidean,
            'manhattan': manhattan,
            'cosine': cosine,
            'chebyshev': chebyshev
        }

        if metric not in metric_map:
            raise ValueError(
                f"Unknown metric '{metric}'. Available: {list(metric_map.keys())}"
            )
        self.metric = metric_map[metric]

    def fit(self, X, y):
        self.X = np.array(X)
        self.y = np.array(y)
        self.classes_ = np.unique(self.y)

    def _compute_distances(self, x):
        return np.array([self.metric(x, x_train) for x_train in self.X])

    def _get_k_neighbors(self, distances):
        nearest_idx = np.argpartition(distances, self.k)[:self.k]
        return self.y[nearest_idx], distances[nearest_idx]

    # Voting functions
    def _vote_uniform(self, labels, **kwargs):
        unique, counts = np.unique(labels, return_counts=True)
        candidates = unique[counts == counts.max()]
        return candidates.max()

    def _vote_distance(self, labels=None, distances=None, **kwargs):
        if labels is None or distances is None:
            raise ValueError("Both 'labels' and 'distances' must be provided for distance weighting.")
        weights = 1 / (distances + 1e-8)
        unique_labels = np.unique(labels)
        scores = np.array([weights[labels == cls].sum() for cls in unique_labels])
        candidates = unique_labels[scores == scores.max()]
        return candidates.max()

    def _vote_rank(self, labels=None, ranks=None, **kwargs):
        if labels is None or ranks is None:
            raise ValueError("Both 'labels' and 'ranks' must be provided for rank weighting.")
        weights = 1 / ranks
        unique_labels = np.unique(labels)
        scores = np.array([weights[labels == cls].sum() for cls in unique_labels])
        candidates = unique_labels[scores == scores.max()]
        return candidates.max()

    def _majority_vote(self, labels, distances=None, ranks=None):
        func = getattr(self, f"_vote_{self.weight}")
        return func(labels=labels, distances=distances, ranks=ranks)

    # Weight functions for probabilities
    def _weight_uniform(self, labels, distances=None, ranks=None, **kwargs):
        return np.ones_like(labels)

    def _weight_distance(self, labels, distances=None, ranks=None, **kwargs):
        if distances is None:
            raise ValueError("Distances must be provided for distance weighting.")
        return 1 / (distances + 1e-8)

    def _weight_rank(self, labels, distances=None, ranks=None, **kwargs):
        if ranks is None:
            raise ValueError("Ranks must be provided for rank weighting.")
        return 1 / ranks

    # Predictions
    def predict(self, X_pred):
        X_pred = np.array(X_pred)
        predictions = []

        for x in X_pred:
            distances = self._compute_distances(x)
            nearest_labels, nearest_distances = self._get_k_neighbors(distances)

            if self.weight == 'rank':
                sorted_idx = np.argsort(nearest_distances)
                ranks = np.arange(1, self.k + 1)[np.argsort(sorted_idx)]
            else:
                ranks = None

            pred = self._majority_vote(
                labels=nearest_labels,
                distances=nearest_distances if self.weight == 'distance' else None,
                ranks=ranks
            )
            predictions.append(pred)

        return np.array(predictions)

    def predict_proba(self, X_pred):
        X_pred = np.array(X_pred)
        proba_list = []

        for x in X_pred:
            distances = self._compute_distances(x)
            nearest_labels, nearest_distances = self._get_k_neighbors(distances)

            if self.weight == 'rank':
                sorted_idx = np.argsort(nearest_distances)
                ranks = np.arange(1, self.k + 1)[np.argsort(sorted_idx)]
            else:
                ranks = None

            weight_func = getattr(self, f"_weight_{self.weight}")
            weights = weight_func(labels=nearest_labels,
                                  distances=nearest_distances,
                                  ranks=ranks)

            total_weight = np.sum(weights)
            weight_class1 = np.sum(weights[nearest_labels == 1])
            proba = weight_class1 / total_weight
            proba_list.append(proba)

        return np.array(proba_list)

    def __str__(self):
        return f"MyKNNClf(k={self.k}, weight='{self.weight}')"
