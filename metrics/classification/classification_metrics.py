import numpy as np


def get_class_stats(y, y_hat, threshhold=0.5):
    """Get confusion matrix."""
    y_hat = np.where(y_hat >= threshhold, 1, 0)
    tp = np.sum((y == 1) & (y_hat == 1))
    tn = np.sum((y == 0) & (y_hat == 0))
    fp = np.sum((y == 0) & (y_hat == 1))
    fn = np.sum((y == 1) & (y_hat == 0))
    return tp, tn, fp, fn


def accuracy(y, y_hat):
    """Get accuracy."""
    tp, tn, fp, fn = get_class_stats(y, y_hat)
    return (tp + tn) / (tp + tn + fp + fn)


def precision(y, y_hat):
    """
    Compute share of realy true positive.
    """
    tp, tn, fp, fn = get_class_stats(y, y_hat)
    return tp / (tp + fp)


def recall(y, y_hat):
    """
    Shows ability to distinguish
    positive class among all ones.
    """
    tp, tn, fp, fn = get_class_stats(y, y_hat)
    return tp / (tp + fn)


def f1(y, y_hat, beta=1):
    """Combination of recall and precision."""
    recall_val = recall(y, y_hat)
    precision_val = precision(y, y_hat)
    return (
        (1 + beta**2) * (precision_val * recall_val)
        / (beta * precision_val + recall_val)
        )


def roc_auc(y, y_hat):
    """
    Compute area under the receiver operating characteristics (AUC).

    Formula: AUC = (1/(P*N)) * sum over all pairs (i,j) of
    I[y_i < y_j] * I'[a_i < a_j]
    where I[y_i < y_j] = 1 if y_i < y_j else 0
    and I'[a_i < a_j] = 1 if a_i < a_j, 0.5 if a_i == a_j, 0 if a_i > a_j
    """
    y = np.asarray(y)
    y_hat = np.asarray(y_hat)

    # Count positive and negative examples
    P = np.sum(y == 1)
    N = np.sum(y == 0)

    if P == 0 or N == 0:
        return 0.0

    # Round predictions to handle floating point precision
    y_hat_rounded = np.round(y_hat, 10)

    # Get indices of negative and positive classes
    neg_indices = np.where(y == 0)[0]
    pos_indices = np.where(y == 1)[0]

    total_sum = 0.0

    # For each negative class, compare with all positive classes
    for neg_idx in neg_indices:
        neg_score = y_hat_rounded[neg_idx]

        # Compare with all positive classes
        pos_scores = y_hat_rounded[pos_indices]

        # Count pairs where neg_score < pos_score (contribution = 1)
        greater_count = np.sum(neg_score < pos_scores)

        # Count pairs where neg_score == pos_score (contribution = 0.5)
        equal_count = np.sum(neg_score == pos_scores)

        # Add to total
        total_sum += greater_count + 0.5 * equal_count

    # Final AUC calculation: total_sum / (P * N)
    auc = total_sum / (P * N)

    return auc


# Mapping of metric names to metric functions
METRIC_MAP = {
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1': f1,
    'roc_auc': roc_auc,
}
