from typing import Dict
import numpy as np
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    v_measure_score,
    fowlkes_mallows_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)
from nlp_pipelines.dataset import Dataset


def evaluate(dataset: Dataset) -> Dict[str, float]:
    """
    Evaluate a dataset's results against its truths using the appropriate metrics for:
    - labels (multi-label, per-instance),
    - clusters (permutation-invariant integers),
    - classification (flat, meaningful categories).

    Returns a dictionary of metric names to values.
    """
    truths = dataset.truths
    results = dataset.results

    if truths is None or results is None:
        raise ValueError("Both `truths` and `results` must be provided.")
    if len(truths) != len(results):
        raise ValueError("Length of `truths` and `results` must match.")

    def is_listlike(x): return isinstance(x, (list, set, tuple))
    is_truth_list = is_listlike(truths[0])
    is_result_list = is_listlike(results[0])

    if is_truth_list != is_result_list:
        raise ValueError("Mismatch: one of `truths` or `results` is list-like, the other is not.")

    metrics = {}

    # multi-label (keywords)
    if is_truth_list:
        precision_total = 0.0
        recall_total = 0.0
        f1_total = 0.0
        exact_match_total = 0.0
        jaccard_total = 0.0
        n = len(truths)

        for t, r in zip(truths, results):
            t_set, r_set = set(t), set(r)
            intersection = t_set & r_set
            union = t_set | r_set

            tp = len(intersection)
            fp = len(r_set - t_set)
            fn = len(t_set - r_set)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            jaccard = tp / len(union) if union else 1.0
            exact = 1.0 if t_set == r_set else 0.0

            precision_total += precision
            recall_total += recall
            f1_total += f1
            jaccard_total += jaccard
            exact_match_total += exact

        metrics["precision_macro"] = precision_total / n
        metrics["recall_macro"] = recall_total / n
        metrics["f1_macro"] = f1_total / n
        metrics["jaccard"] = jaccard_total / n
        metrics["exact_match_ratio"] = exact_match_total / n

    else:
        # Now assume flat labels; check if it's clustering or classification
        truths_np = np.array(truths)
        results_np = np.array(results)

        is_numeric = np.issubdtype(type(truths_np[0]), np.integer)

        if is_numeric:
            # Clustering (no label identity) # may need work??
            metrics["adjusted_rand_index"] = adjusted_rand_score(truths_np, results_np)
            metrics["normalized_mutual_info"] = normalized_mutual_info_score(truths_np, results_np)
            metrics["v_measure"] = v_measure_score(truths_np, results_np)
            metrics["fowlkes_mallows"] = fowlkes_mallows_score(truths_np, results_np)
        else:
            # Classification (labels have identity)
            metrics["accuracy"] = accuracy_score(truths_np, results_np)
            metrics["precision_macro"] = precision_score(truths_np, results_np, average="macro", zero_division=0)
            metrics["recall_macro"] = recall_score(truths_np, results_np, average="macro", zero_division=0)
            metrics["f1_macro"] = f1_score(truths_np, results_np, average="macro", zero_division=0)

    return metrics
