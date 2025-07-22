import random
from typing import List, Optional, Tuple
import numpy as np

class Dataset:

    texts: List[str]
    original_texts: List[str]
    truths: Optional[List[str]]
    vectors: Optional[List[List[float]] | np.ndarray]
    results: Optional[List[str]]

    def __init__(self, texts: List[str], truths: Optional[List[str]] = None):
        """
        Initialize the Dataset object.

        Args:
            texts: List of document strings.
            truths: (Optional) List of truths corresponding to the texts.

        Raises:
            AssertionError: If truths are provided, but their length doesn't match the length of texts, or if the dataset is empty.
        """
        if truths:
            assert truths is None or len(texts) == len(truths), "Texts and truths must have the same length."
        assert len(texts) > 0, "Refusing to make a text dataset with no text."

        self.texts = texts
        self.original_texts = texts.copy()
        self.truths = truths
        self.vectors = None
        self.results = None

    def get(self, labeled: bool = True, count: int = -1, ratio: float = 1.0, seed: Optional[int] = None) -> 'Dataset':
        """
        Get a random subset of the dataset.

        Args:
            labeled: (Optional) If True, return only labeled data.
            count: (Optional) Max number of documents to return.
            ratio: (Optional) Proportion of the dataset to return (0 to 1).
            seed: (Optional) seed for reproducibility.

        Returns:
            A tuple of (texts, truths), where truths will be None if unlabeled data is requested.
        
        Raises:
            ValueError: If ratio is not between 0 and 1.
        """
        if ratio < 0 or ratio > 1:
            raise ValueError("Ratio must be between 0 and 1.")

        indices = list(range(len(self.texts)))
        if labeled and self.truths is not None:
            indices = [i for i in indices if self.truths[i] is not None]

        # if count not set, get all
        if count == -1:
            count = len(indices)

        subset_size = min(len(indices), int(len(indices) * ratio), count)
        random.seed(seed) # if seed is None, it uses system time to seed
        selected_indices = random.sample(indices, subset_size)

        texts = [self.texts[i] for i in selected_indices]
        truths = [self.truths[i] for i in selected_indices] if labeled and self.truths is not None else None

        ds = Dataset(texts, truths)

        return ds

    def split(self, labeled: bool = True, splitLabeled: bool = True, count: int = 100, ratio: float = 1.0, seed: Optional[int] = None) -> Tuple['Dataset', 'Dataset']:
        """
        Split the dataset into two disjoint subsets.

        Args:
            labeled: (Optional) If True, return only labeled data for the first split.
            splitLabeled: (Optional) If True, return from only labeled data for the second split, otherwise from all remaiing documents.
            count: (Optional) Max number of documents to return.
            ratio: (Optional) Proportion of the dataset to return (0 to 1).
            seed: (Optional) seed for reproducibility.

        Returns:
            A tuple of (texts, truths), where truths will be None if unlabeled data is requested.
        
        Raises:
            ValueError: If ratio is not between 0 and 1.
        """
        if ratio < 0 or ratio > 1:
            raise ValueError("Ratio must be between 0 and 1.")

        indices = list(range(len(self.texts)))
        split_indices = indices.copy()
        if labeled and self.truths is not None:
            indices = [i for i in indices if self.truths[i] is not None]
        # if this is set, pull from labeled docs for the split, otherwise from ALL docs.
        if splitLabeled:
            split_indices = indices.copy()
        
        # if count not set, get all
        if count == -1:
            count = len(indices)

        subset_size = min(len(indices), int(len(indices) * ratio), count)

        random.seed(seed)
        selected_indices = set(random.sample(indices, subset_size))
        remaining_indices = [i for i in split_indices if i not in selected_indices]

        subset_texts = [self.texts[i] for i in selected_indices]
        subset_truths = [self.truths[i] for i in selected_indices] if labeled and self.truths is not None else None

        remaining_texts = [self.texts[i] for i in remaining_indices]
        remaining_truths = [self.truths[i] for i in remaining_indices] if labeled and self.truths is not None else None

        ds_1 = Dataset(subset_texts, subset_truths)
        ds_2 = Dataset(remaining_texts, remaining_truths)

        return ds_1, ds_2
