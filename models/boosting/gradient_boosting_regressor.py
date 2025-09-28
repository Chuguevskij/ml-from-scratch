import numpy as np
import pandas as pd

from models.decision_tree import MyTreeReg, DecisionNode
from metrics.regression import *


class MyBoostReg:
    def __init__(
        self,
        n_estimators=6,
        learning_rate=0.1,
        max_depth=5,
        min_samples_split=2,
        max_leafs=20,
        bins=16,
        loss='MSE',
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

        # Store the chosen functions
        self.neg_grad_func = loss_map[self.loss]
        self.loss_function = loss_funcs[self.loss]

    def fit(self, X, y, verbose=None):
        """Train gradient boosting regressor."""
        # Check y is np
        y = np.array(y)

        # 1st baseline prediction
        self.pred_0 = np.mean(y)
        F_pred = self.pred_0

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
            # Step 3: update leaf values to minimize original loss
            for leaf in single_tree.get_leaves():  # you need a method to access leaves
                indices = leaf.sample_indices  # indices of samples in this leaf
                target_residual = y[indices] - F_pred[indices]
                
                if self.loss == "MSE":
                    leaf.value = np.mean(target_residual)
                elif self.loss == "MAE":
                    leaf.value = np.median(target_residual)
            self.trees.append(single_tree)

            # Current prediction
            F_pred += self.learning_rate * single_tree.predict(X)
            residuals = self.neg_grad_func(y, F_pred)

            if verbose:
                if n % verbose == 0:
                    print(f"{n}. Loss[{self.loss}]: {self.loss_function(y, F_pred)}")

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
