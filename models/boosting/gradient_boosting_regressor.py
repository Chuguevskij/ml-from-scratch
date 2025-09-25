import numpy as np
import pandas as pd

from models.decision_tree import MyTreeReg, DecisionNode
from metrics import regression_metrics as metrics


class MyBoostReg:
    def __init__(
        self,
        n_estimators=6,
        learning_rate=0.1,
        max_depth=5,
        min_samples_split=2,
        max_leafs=20,
        bins=16,
        loss = 'MSE',
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate

        # Tree parameters
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
        self.bins = bins

        # Learning
        self.loss = loss

        # Boosting utility parameters
        self.pred_0 = None
        self.trees = []

    def fit(self, X, y):
        """Train gradient boosting regressor."""
        # Check y is np
        y = np.array(y)

        # 1st baseline prediction
        self.pred_0 = np.mean(y)

        # 1st baseline target
        residuals = y - self.pred_0

        for _ in range(self.n_estimators):
            # Current target

            # Train a single tree
            single_tree = MyTreeReg(
                self.max_depth,
                self.min_samples_split,
                self.max_leafs,
                self.bins,
            )
            single_tree.fit(X, residuals)
            self.trees.append(single_tree)

            # Current prediction
            cur_pred = single_tree.predict(X)
            residuals = residuals - self.learning_rate * cur_pred

    def predict(self, X):
        pred = self.pred_0 + self.learning_rate * sum(tree.predict(X) for tree in self.trees)
        return pred

    def __str__(self):
        params = [
            "n_estimators",
            "learning_rate",
            "max_depth",
            "min_samples_split",
            "max_leafs",
            "bins",
        ]
        attrs = ", ".join(f"{p}={getattr(self, p)}" for p in params)
        return f"{self.__class__.__name__} class: {attrs}"
