import random

import numpy as np
import pandas as pd

from models.decision_tree import MyTreeReg, DecisionNode


class MyForestReg:
    def __init__(
        self,
        n_estimators=10,
        max_features=0.5,
        max_samples=0.5,
        random_state=42,
        max_depth=5,
        min_samples_split=2,
        max_leafs=20,
        bins=16
    ):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_samples = max_samples
        self.random_state = random_state
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
        self.bins = bins

        self.leafs_cnt = 0
        self.trees = []
        self.fi = {}

    def fit(self, X, y):
        """Train random forest regressor."""
        random.seed(self.random_state)
        init_rows_cnt, init_cols_cnt = X.shape
        init_cols = X.columns.tolist()
        cols_smpl_cnt = round(self.max_features * init_cols_cnt)
        rows_smpl_cnt = round(self.max_samples * init_rows_cnt)
        self.fi = {col: 0.0 for col in init_cols}

        # Train a forest
        for _ in range(self.n_estimators):
            cols_idx = random.sample(init_cols, cols_smpl_cnt)
            rows_idx = random.sample(range(init_rows_cnt), rows_smpl_cnt)
            bt_y = y.iloc[rows_idx]
            bt_X = X.loc[rows_idx, cols_idx]

            # Train a single tree
            single_tree = MyTreeReg(
                self.max_depth,
                self.min_samples_split,
                self.max_leafs,
                self.bins,
                init_rows_cnt
            )
            single_tree.fit(bt_X, bt_y)
            self.trees.append((single_tree, cols_idx))

            # Sum feature importance (no weighting here!)
            for f in self.fi.keys():
                self.fi[f] += single_tree.fi.get(f, 0.0)

        for tree, _ in self.trees:
            self.leafs_cnt += tree.leafs_cnt

    def predict(self, X):
        preds = []
        # Collect predictions from each tree
        for single_tree, cols_idx in self.trees:
            single_preds = single_tree.predict(X[cols_idx])
            preds.append(single_preds)
        preds = np.array(preds)
        return preds.mean(axis=0).tolist()

    def __str__(self):
        params = [
            "n_estimators",
            "max_features",
            "max_samples",
            "random_state",
            "max_depth",
            "min_samples_split",
            "max_leafs",
            "bins",
        ]
        attrs = ", ".join(f"{p}={getattr(self, p)}" for p in params)
        return f"MyForestReg({attrs})"