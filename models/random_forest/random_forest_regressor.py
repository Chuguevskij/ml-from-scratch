import random

import numpy as np
import pandas as pd

from models.decision_tree import MyTreeReg, DecisionNode
from metrics.regression import *


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
        bins=16,
        oob_score=None
    ):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_samples = max_samples
        self.random_state = random_state
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
        self.bins = bins
        self.oob_score = oob_score

        self.leafs_cnt = 0
        self.trees = []
        self.fi = {}
        self.oob_preds_ = None
        self.oob_counts_ = None
        self.oob_score_ = None

    def fit(self, X, y):
        """Train random forest regressor."""
        random.seed(self.random_state)
        init_rows_cnt, init_cols_cnt = X.shape
        init_cols = X.columns.tolist()
        cols_smpl_cnt = round(self.max_features * init_cols_cnt)
        rows_smpl_cnt = round(self.max_samples * init_rows_cnt)
        all_indices = set([i for i in range(init_rows_cnt)])
        self.fi = {col: 0.0 for col in init_cols}
        self.oob_preds_ = np.zeros(len(y))
        self.oob_counts_ = np.zeros(len(y))

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

            # Collect OOB preds
            if self.oob_score:
                oob_indices = list(all_indices - set(rows_idx))
                self.oob_preds_[oob_indices] += single_tree.predict(
                    X.loc[oob_indices, cols_idx]
                )
                self.oob_counts_[oob_indices] += 1

            # Sum feature importance
            for f in self.fi.keys():
                self.fi[f] += single_tree.fi.get(f, 0.0)

        # Evaluate on OOB preds
        if self.oob_score:
            mask = self.oob_counts_ > 0
            self.oob_score_ = None
            if np.any(mask):
                oob_mean_preds = self.oob_preds_[mask] / self.oob_counts_[mask]
                metric_func = getattr(metrics, self.oob_score)
                self.oob_score_ = metric_func(y[mask], oob_mean_preds)

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
