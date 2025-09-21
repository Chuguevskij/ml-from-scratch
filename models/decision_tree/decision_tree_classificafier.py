import numpy as np
import pandas as pd
from tree_node import DecisionNode


class MyTreeClf():
    """
    A simple implementation of a Decision Tree Classifier.
    """

    def __init__(
        self,
        max_depth=5,
        min_samples_split=2,
        max_leafs=20,
        bins=None
    ):
        """
        Initializes a Decision Tree.

        Args:
            max_depth (int): Maximum depth of the decision tree. Defaults to 5.
            min_samples_split (int): Minimum number of samples required
            to split. Defaults to 2.
            max_leafs (int): Maximum number of leaves in the tree. Defaults
            to 20.
        """
        self.root = None
        self.leafs_cnt = 1
        self.leaves_sum = None

        # Building restriction parameters
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs

        # Binarization
        self.bins = bins
        self.threshold_lists = None

    @staticmethod
    def _shannon_entropy(y, eps=1e-12):
        """Calculate shannon_entropy for a single sample."""
        p_0 = len(y[y == 0]) / len(y)
        p_1 = len(y[y == 1]) / len(y)
        return -(
            p_0 * np.log2(p_0+eps) +
            p_1 * np.log2(p_1+eps)
        )

    @staticmethod
    def _information_gain(y, y_left, y_right):
        """Calculate information gain for a single split."""
        n = len(y)
        share_left = len(y_left) / n
        share_right = len(y_right) / n
        return (
            MyTreeClf._shannon_entropy(y) -
            share_left * MyTreeClf._shannon_entropy(y_left) -
            share_right * MyTreeClf._shannon_entropy(y_right)
        )

    def _prepare_thresholds(self, X):
        """Prepare thresholds based on bins or unique values (aligned with regression)."""
        n_samples, n_features = X.shape
        binarized_thresholds = {}

        for feature_i in range(n_features):
            unique_splits = np.unique(X[:, feature_i])

            if self.bins is None or len(unique_splits) <= self.bins:
                # Use midpoints between sorted unique values
                if len(unique_splits) > 1:
                    binarized_thresholds[feature_i] = [
                        (unique_splits[i] + unique_splits[i+1]) / 2
                        for i in range(len(unique_splits)-1)
                    ]
                else:
                    binarized_thresholds[feature_i] = unique_splits
            else:
                # Use histogram bin edges (excluding min and max)
                _, bins = np.histogram(X[:, feature_i], bins=self.bins)
                binarized_thresholds[feature_i] = bins[1:-1]

        return binarized_thresholds

    def _get_leaf_value(self, y, th=0.5):
        """
        Calculates the most occurring value in the given list of y values.

        Args:
            y (list): The list of y values.

        Returns:
            The most occurring value in the list.
        """
        # Most occurring value
        # values, counts = np.unique(y, return_counts=True)
        # most_occuring_value = values[np.argmax(counts)]

        # Probability of class 1
        prob_class_1 = np.mean(y == 1)

        return prob_class_1

    def _get_best_split(self, X, y):
        """Find the best feature and threshold to split."""
        n_samples, n_features = X.shape
        best_gain = -1
        best_split = None

        for feature_i in range(n_features):
            feature_values = X[:, feature_i]
            # Choose what thresholds to use
            if self.threshold_lists:
                splits = self.threshold_lists[feature_i]
                if len(splits) <= 1:
                    continue
            else:
                thresholds = np.unique(feature_values)
                thresholds = np.sort(thresholds)
                if len(thresholds) <= 1:
                    continue
                splits = [
                    (thresholds[i] + thresholds[i + 1]) / 2
                    for i in range(len(thresholds) - 1)
                ]

            for threshold in splits:
                left_mask = feature_values <= threshold
                right_mask = feature_values > threshold

                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                y_left = y[left_mask]
                y_right = y[right_mask]

                gain = self._information_gain(y, y_left, y_right)

                if gain > best_gain:
                    best_gain = gain
                    best_split = {
                        'feature_i': feature_i,
                        'threshold': threshold,
                        'gain': gain,
                        'left_mask': left_mask,
                        'right_mask': right_mask
                    }

        return best_split

    def _build_tree(self, X, y, depth=0):
        n_labels = len(np.unique(y))
        n_samples = len(y)

        # Standard stopping criteria
        if (
            n_labels <= 1 or
            depth >= self.max_depth or
            n_samples < self.min_samples_split or
            self.leafs_cnt >= self.max_leafs
        ):
            return DecisionNode(value=self._get_leaf_value(y))

        split = self._get_best_split(X, y)
        if split is None or split['gain'] <= 0 or split['feature_i'] is None:
            return DecisionNode(value=self._get_leaf_value(y))

        # Real split -> only now increment leaf counter
        self.leafs_cnt += 1

        feature = split['feature_i']
        threshold = split['threshold']
        left = self._build_tree(X[split['left_mask']], y[split['left_mask']], depth+1)
        right = self._build_tree(X[split['right_mask']], y[split['right_mask']], depth+1)

        return DecisionNode(
            feature_i=feature,
            threshold=threshold,
            gain=split['gain'],
            left_branch=left,
            right_branch=right
        )

    def fit(self, X, y):
        """Fit the decision tree classifier to the data."""
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
            X = X.values
        else:
            self.feature_names = None
        y = np.array(y)
        self.leafs_cnt = 1
        self.threshold_lists = None
        if self.bins:
            self.threshold_lists = self._prepare_thresholds(X)
        self.root = self._build_tree(X, y)

    def predict_proba(self, X_pred):
        "Predict probability for the 1st class."
        if self.root is None:
            print('Tree is not build')
            return []

        X_pred = np.array(X_pred)

        def _predict_proba_sample(node, x):
            if node.value is not None:
                return node.value
            if x[node.feature_i] <= node.threshold:
                return _predict_proba_sample(node.left_branch, x)
            else:
                return _predict_proba_sample(node.right_branch, x)

        return np.array([_predict_proba_sample(self.root, x) for x in X_pred])

    def predict(self, X_pred):
        "Predict class."
        probs = self.predict_proba(X_pred)
        return (probs > 0.5).astype(int)

    def calculate_leaves(self):
        """
        Calculates sum of leaves from bft().
        """
        levels = self.bft()
        self.leaves_sum = sum(
            num for sublist in levels
            for num in sublist
            if not isinstance(num, tuple)
        )

    def print_stat(self):
        self.calculate_leaves()
        print(f"leaves number: {self.leafs_cnt}, leaves sum: {self.leaves_sum}")

    def bft(self):
        "Breadth-first traversal."
        if self.root is None:
            print('Tree is not build')
            return []

        queue = []
        res = []

        level = 0
        queue.append(self.root)

        while queue:
            res.append([])
            for _ in range(len(queue)):
                node = queue.pop(0)

                if node.value is None:
                    node_data = (node.feature_i, node.threshold)
                else:
                    node_data = (node.value)
                res[level].append(node_data)

                if node.left_branch is not None:
                    queue.append(node.left_branch)

                if node.right_branch is not None:
                    queue.append(node.right_branch)
            level += 1

        return res

    def print_tree(self):
        levels = self.bft()
        for level_idx, nodes in enumerate(levels):
            indent = "  " * level_idx
            for node in nodes:
                if isinstance(node, tuple):
                    feature_i, threshold = node
                    if self.feature_names and feature_i is not None:
                        feature_label = self.feature_names[feature_i]
                    else:
                        feature_label = f"feature {feature_i}"
                    print(f"{indent}Level {level_idx}: ({feature_label} <= {threshold})")
                else:
                    print(f"{indent}Level {level_idx}: Predict: {node}")

    def __str__(self):
        return (
            f"{__class__.__name__} class: max_depth={self.max_depth}, "
            f"min_samples_split={self.min_samples_split}, "
            f"max_leafs={self.max_leafs}"
            )
