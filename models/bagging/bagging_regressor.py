import random

import numpy as np
import pandas as pd


class MyBaggingReg:
    def __init__(
        self,
        estimator=None,
        n_estimators=10,
        max_samples=0.5,
        random_state=42,
    ):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state

    def __str__(self):
        params = [
            "estimator",
            "n_estimators",
            "max_samples",
            "random_state",
        ]
        attrs = ", ".join(f"{p}={getattr(self, p)}" for p in params)
        return f"MyForestReg({attrs})"
