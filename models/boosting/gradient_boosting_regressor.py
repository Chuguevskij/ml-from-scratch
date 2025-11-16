import numpy as np
import pandas as pd

from models.decision_tree import MyTreeReg, DecisionNode
from metrics.regression import *


class MyBoostReg:
    def __init__(
        self,
        n_estimators=6,
        learning_rate=0.1,
        max_features=0.5,
        max_samples=0.5,
        random_state=42,
        max_depth=5,
        min_samples_split=2,
        max_leafs=20,
        bins=16,
        loss='MSE',
        metric=None
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_features = max_features
        self.max_samples = max_samples
        self.random_state = random_state

        # Tree parameters
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
        self.bins = bins

        # Learning
        self.loss = loss
        self.metric = metric
        self.best_score = None

        # Boosting utility parameters
        self.pred_0 = None
        self.trees = []

        # Metrics and losses
        loss_map = {
            "MSE": neg_grad_mse,
            "MAE": neg_grad_mae
        }
        loss_funcs = {
            "MSE": mse,
            "MAE": mae,
            "RMSE": rmse,
            "MAPE": mape,
            "R2": r2
        }
        leaf_functions = {
            "MSE": np.mean,
            "MAE": np.median,
            "Quantile": lambda x: np.quantile(x, 0.9),
        }

        # Store the chosen functions
        if self.metric:
            self.metric_function = loss_funcs[self.metric]
        self.neg_grad_func = loss_map[self.loss]
        self.loss_function = loss_funcs[self.loss]
        self.leaf_function = leaf_functions[self.loss]

    def _leaves_update(self, tree, X, y, F_pred_last):
        """
        Update each leaf value to minimize the loss, relative to current ensemble prediction.
        X: full feature array
        y: target
        F_pred_last: current predictions before this tree
        """
        y = np.asarray(y).flatten()
        F_pred_last = np.asarray(F_pred_last).flatten()

        def update_node(node, indices):
            if node is None:
                return

            # Leaf check: no children
            if node.left_branch is None and node.right_branch is None:
                residuals = y[indices] - F_pred_last[indices]
                node.value = residuals[0] if residuals.size == 1 else self.leaf_function(residuals)
                return

            # Internal node split
            feature_vals = X[indices, node.feature_i]
            left_indices = indices[feature_vals <= node.threshold]
            right_indices = indices[feature_vals > node.threshold]

            if left_indices.size > 0:
                update_node(node.left_branch, left_indices)
            if right_indices.size > 0:
                update_node(node.right_branch, right_indices)

        update_node(tree.root, np.arange(X.shape[0]))

        update_node(tree.root, np.arange(X.shape[0]))

    def fit(self, X, y, verbose=None):
        """Train gradient boosting regressor."""
        # Check y is np
        y = np.array(y)
        X = np.asarray(X)

        # 1st baseline prediction
        self.pred_0 = self.leaf_function(y)
        F_pred = np.ones(len(y), dtype=float) * self.pred_0

        # 1st baseline target
        residuals = self.neg_grad_func(y, self.pred_0)

        for n in range(self.n_estimators):
            # Current target

            # Train a single tree
            single_tree = MyTreeReg(
                self.max_depth,
                self.min_samples_split,
                self.max_leafs,
                self.bins,
            )
            single_tree.fit(X, residuals)

            # Recompute leaves using true residuals
            self._leaves_update(single_tree, X, y, F_pred)

            # Update ensemble prediction
            F_pred += self.learning_rate * single_tree.predict(X)

            # Recompute residuals for next iteration
            residuals = self.neg_grad_func(y, F_pred)

            # Store fitted tree
            self.trees.append(single_tree)

            # Print step learning stats
            if verbose:
                if n % verbose == 0:
                    if self.metric:
                        print(f"{n}. Loss[{self.loss}]: {self.loss_function(y, F_pred)} | {self.metric}: {self.metric_function(y, F_pred)}")
                    else:
                        print(f"{n}. Loss[{self.loss}]: {self.loss_function(y, F_pred)}")

            # Evaluate the last model
            if self.metric:
                self.best_score = self.metric_function(y, F_pred)
            else:
                self.best_score = self.loss_function(y, F_pred)

    def predict(self, X):
        F_pred = self.pred_0
        for tree in self.trees:
            F_pred += self.learning_rate * tree.predict(X)
        return F_pred

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
