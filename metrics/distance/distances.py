import numpy as np


def euclidean(v1, v2):
    """
    Calculate Euclidean distance between two vectors of n-features.
    """
    # Check inputs to be numpy arrays
    v1 = np.array(v1)
    v2 = np.array(v2)

    # Check if shapes match
    if v1.shape != v2.shape:
        raise ValueError("Input vectors must have the same length.")

    # Compute Euclidean distance
    return np.sqrt(np.sum((v1 - v2) ** 2))


def manhattan(v1, v2):
    """
    Calculate Manhattan distance between two vectors of n-features.
    """
    # Check inputs to be numpy arrays
    v1 = np.array(v1)
    v2 = np.array(v2)

    # Check if shapes match
    if v1.shape != v2.shape:
        raise ValueError("Input vectors must have the same length.")

    # Compute Manhattan distance
    return np.sum(np.abs(v1 - v2))


def cosine(v1, v2):
    """
    Calculate cosine distance between two vectors of n-features.
    """
    # Check inputs to be numpy arrays
    v1 = np.array(v1)
    v2 = np.array(v2)

    # Check if shapes match
    if v1.shape != v2.shape:
        raise ValueError("Input vectors must have the same length.")

    numerator = np.sum(v1 * v2)
    denominator = (
        np.sqrt(np.sum(v1**2)) * np.sqrt(np.sum(v2**2))
    )

    return 1 - (numerator / denominator)


def chebyshev(v1, v2):
    """
    Calculate Chebyshev distance between two vectors of n-features.
    """
    # Check inputs to be numpy arrays
    v1 = np.array(v1)
    v2 = np.array(v2)

    # Check if shapes match
    if v1.shape != v2.shape:
        raise ValueError("Input vectors must have the same length.")

    # Compute Chebyshev distance
    return np.max(np.abs(v1 - v2))
