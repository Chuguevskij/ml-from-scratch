import random

import numpy as np
import pandas as pd

from models.decision_tree import MyTreeClf, DecisionNode


class MyForestClf:
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
        criterion='entropy',
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
        self.criterion = criterion
        self.oob_score = oob_score

        self.leafs_cnt = 0
        self.trees = []
        self.fi = {}
        self.oob_preds_ = None
        self.oob_counts_ = None
        self.oob_score_ = None

    def __str__(self):
        params = [
            "n_estimators",
            "max_features",
            "max_samples",
            "max_depth",
            "min_samples_split",
            "max_leafs",
            "bins",
            "criterion",
            "random_state"
        ]
        attrs = ", ".join(f"{p}={getattr(self, p)}" for p in params)
        return f"{self.__class__.__name__} class: {attrs}"
