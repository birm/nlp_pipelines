from typing import Tuple
import numpy as np
from nlp_pipelines.dataset.dataset import Dataset
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    precision_score,
    recall_score,
    f1_score,
)

def evaluate(
    dataset: Dataset
) -> Tuple[float, float, float, float, float, float]:
    """
    Compute clustering evaluation metrics given a dataset object with
    `truths` and `results` attributes.

    Args:
        dataset: An object with
            - `truths`: Sequence of true labels.
            - `results`: Sequence of predicted cluster labels.
            Both sequences must have the same length.

    Returns:
        Tuple containing:
            - Adjusted Rand Index (ARI)
            - Normalized Mutual Information (NMI)
            - Precision (macro-average)
            - Recall (macro-average)
            - F1 score (macro-average)
            - Exact match ratio (accuracy)
    """
    truths = dataset.truths
    results = dataset.results

    if truths is not None and results is not None and len(truths) != len(results):
        raise ValueError("Length of truths and results must be the same.")

    truths_np = np.array(truths)
    results_np = np.array(results)

    ari = adjusted_rand_score(truths_np, results_np)
    nmi = normalized_mutual_info_score(truths_np, results_np)
    precision = precision_score(truths_np, results_np, average="macro", zero_division=0)
    recall = recall_score(truths_np, results_np, average="macro", zero_division=0)
    f1 = f1_score(truths_np, results_np, average="macro", zero_division=0)
    exact_match = np.mean(truths_np == results_np)

    return ari, nmi, precision, recall, f1, exact_match
