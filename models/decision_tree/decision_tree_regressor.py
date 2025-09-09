import numpy as np
import pandas as pd
from tree_node import DecisionNode


class MyTreeReg():
    """
    Decision Tree Regressor.
    """

    def __init__(
        self,
        max_depth=5,
        min_samples_split=2,
        max_leafs=20,
        bins=None
    ):
        self.root = None
        self.leafs_cnt = 1
        self.leaves_sum = None

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs

        self.bins = bins
        self.threshold_lists = None
        self.feature_names = None
        self.fi = {}

    @staticmethod
    def _variance(y):
        n = len(y)
        y_mean = np.mean(y)
        return np.sum((y - y_mean) ** 2) / n

    @staticmethod
    def _variance_reduction(y, y_left, y_right):
        n = len(y)
        share_left = len(y_left) / n
        share_right = len(y_right) / n
        return (
            MyTreeReg._variance(y)
            - share_left * MyTreeReg._variance(y_left)
            - share_right * MyTreeReg._variance(y_right))

    def _prepare_thresholds(self, X):
        """Prepare thresholds based on bins or unique values."""
        n_samples, n_features = X.shape
        binarized_thresholds = {}

        for feature_i in range(n_features):
            unique_splits = np.unique(X[:, feature_i])
            if self.bins is None or len(unique_splits) <= self.bins:
                binarized_thresholds[feature_i] = [
                    (unique_splits[i] + unique_splits[i+1]) / 2
                    for i in range(len(unique_splits)-1)
                ]
            else:
                _, bins = np.histogram(X[:, feature_i], bins=self.bins)
                binarized_thresholds[feature_i] = bins[1:-1]
        return binarized_thresholds

    def _get_best_split(self, X, y):
        n_samples, n_features = X.shape
        best_gain = -1
        best_split = None

        for feature_i in range(n_features):
            feature_values = X[:, feature_i]

            if self.threshold_lists:
                splits = self.threshold_lists[feature_i]
                if len(splits) <= 0:
                    continue
            else:
                thresholds = np.unique(feature_values)
                thresholds = np.sort(thresholds)
                if len(thresholds) <= 1:
                    continue
                splits = [(thresholds[i] + thresholds[i + 1]) / 2 for i in range(len(thresholds)-1)]

            for threshold in splits:
                left_mask = feature_values <= threshold
                right_mask = feature_values > threshold
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue
                y_left, y_right = y[left_mask], y[right_mask]
                gain = self._variance_reduction(y, y_left, y_right)
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
            return DecisionNode(value=np.mean(y))

        split = self._get_best_split(X, y)
        if split is None or split['gain'] <= 0 or split['feature_i'] is None:
            return DecisionNode(value=np.mean(y))

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
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
            X = X.values
        else:
            self.feature_names = None
        y = np.array(y)
        self.leafs_cnt = 1
        if self.bins:
            self.threshold_lists = self._prepare_thresholds(X)
        self.root = self._build_tree(X, y)
        self.fi = self._feature_importance(X)

    def predict(self, X_pred):
        if self.root is None:
            return []

        X_pred = np.array(X_pred)

        def predict_sample(node, x):
            if node.value is not None:
                return node.value
            if x[node.feature_i] <= node.threshold:
                return predict_sample(node.left_branch, x)
            else:
                return predict_sample(node.right_branch, x)

        return np.array([predict_sample(self.root, x) for x in X_pred])

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

    def _feature_importance(self, X):
        n_total, f_total = X.shape
        fi = {f: 0 for f in range(f_total)}

        def traverse(node, X_subset):
            if node.value is not None:
                return

            # weighted contribution of this split
            n_samples_node = len(X_subset)
            fi[node.feature_i] += node.gain * n_samples_node / n_total

            # split X and y according to this node's threshold
            left_mask = X_subset[:, node.feature_i] <= node.threshold
            right_mask = X_subset[:, node.feature_i] > node.threshold

            if node.left_branch:
                traverse(node.left_branch, X_subset[left_mask])
            if node.right_branch:
                traverse(node.right_branch, X_subset[right_mask])
        traverse(self.root, X)

        if self.feature_names:
            fi_names = {self.feature_names[i]: v for i, v in fi.items()}
            return fi_names
        return fi

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

    def print_tree(self):
        """
        Print of tree levels from bft().
        """
        levels = self.bft()
        for level_idx, nodes in enumerate(levels):
            indent = "  " * level_idx
            for node in nodes:
                if isinstance(node, tuple):
                    feature_i, threshold = node
                    print(
                        f"{indent}Level {level_idx}: "
                        f"(feature {feature_i} <= {threshold})"
                    )
                else:
                    print(f"{indent}Level {level_idx}: Predict: {node}")

    def __str__(self):
        return (
            f"{__class__.__name__} class: max_depth={self.max_depth}, "
            f"min_samples_split={self.min_samples_split}, "
            f"max_leafs={self.max_leafs}"
            )
