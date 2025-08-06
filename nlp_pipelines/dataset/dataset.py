import random
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
import json

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

    @classmethod
    def from_csv(cls, filename: str, text_field: str, truth_field: Optional[str] = None) -> 'Dataset':
        """
        Read dataset from a CSV file.

        Args:
            filename (str): Path to the CSV file.
            text_field (str): The column name containing the text.
            truth_field (str, optional): The column name containing the truth labels (optional).

        Returns:
            Dataset: The loaded Dataset object.
        """
        df = pd.read_csv(filename)
        texts = df[text_field].tolist()

        truths = None
        if truth_field and truth_field in df.columns:
            truths = df[truth_field].tolist()

        return cls(texts, truths)

    @classmethod
    def from_parquet(cls, filename: str, text_field: str, truth_field: Optional[str] = None) -> 'Dataset':
        """
        Read dataset from a Parquet file.

        Args:
            filename (str): Path to the Parquet file.
            text_field (str): The column name containing the text.
            truth_field (str, optional): The column name containing the truth labels (optional).

        Returns:
            Dataset: The loaded Dataset object.
        """
        df = pd.read_parquet(filename)
        texts = df[text_field].tolist()

        truths = None
        if truth_field and truth_field in df.columns:
            truths = df[truth_field].tolist()

        return cls(texts, truths)

    @classmethod
    def from_json(cls, filename: str, text_field: str, truth_field: Optional[str] = None) -> 'Dataset':
        """
        Read dataset from a JSON file.

        Args:
            filename (str): Path to the JSON file.
            text_field (str): The key containing the text.
            truth_field (str, optional): The key containing the truth labels (optional).

        Returns:
            Dataset: The loaded Dataset object.
        """
        with open(filename, 'r') as f:
            data = json.load(f)

        texts = [item[text_field] for item in data]

        truths = None
        if truth_field:
            truths = [item.get(truth_field) for item in data]

        return cls(texts, truths)

    # print an informative string if printed
    def __repr__(self):
        repr_str = f"<Dataset with {len(self.texts)} texts"

        # If vectors are available, show the shape of the first one
        if self.vectors is not None:
            repr_str += f", vectors: {np.shape(self.vectors)[1]}-dim" if isinstance(self.vectors, np.ndarray) else f", vectors: {len(self.vectors[0])}-dim"

        # If results are available, add the number of results
        if self.results is not None:
            repr_str += f", results: {len(self.results)} items"

        # Add a preview of the first few texts and truths
        repr_str += "\nTexts: "
        if len(self.texts) > 2:
            repr_str += f"{self.texts[:2]}... +{len(self.texts) - 2} more"
        else:
            repr_str += f"{self.texts[:2]}"

        if self.truths is not None:
            repr_str += "\nTruths: "
            if len(self.truths) > 2:
                repr_str += f"{self.truths[:2]}... +{len(self.truths) - 2} more"
            else:
                repr_str += f"{self.truths[:2]}"

        
        if self.results is not None:
            repr_str += "\nResults: "
            if len(self.results) > 2:
                repr_str += f"{self.results[:2]}... +{len(self.results) - 2} more"
            else:
                repr_str += f"{self.results[:2]}"

        repr_str += ">"
        return repr_str


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

    def split(self, labeled: bool = False, splitLabeled: bool = False, count: int = 100, ratio: float = 1.0, seed: Optional[int] = None) -> Tuple['Dataset', 'Dataset']:
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
    
    def to_df(self) -> pd.DataFrame:
        """
        Convert the dataset to a pandas DataFrame.

        Returns:
            pd.DataFrame: A DataFrame representing the Dataset object.
        """
        data = {
            'text': self.texts,
            'original_text': self.original_texts,
        }
        
        if self.truths:
            data['truth'] = self.truths
        
        if self.results:
            data['results'] = self.results

        # Convert the dictionary into a DataFrame
        return pd.DataFrame(data)

    def to_csv(self, filename: str) -> None:
        """
        Write the dataset to a CSV file.

        Args:
            filename (str): Path to the output CSV file.
        """
        df = self.to_df()
        df.to_csv(filename, index=False)

    def to_parquet(self, filename: str) -> None:
        """
        Write the dataset to a Parquet file.

        Args:
            filename (str): Path to the output Parquet file.
        """
        df = self.to_df()
        df.to_parquet(filename, index=False)

    def to_json(self, filename: str) -> None:
        """
        Write the dataset to a JSON file.

        Args:
            filename (str): Path to the output JSON file.
        """
        # Convert dataset into a list of dictionaries
        data = []
        for i in range(len(self.texts)):
            item = {
                'text': self.texts[i],
                'original_text': self.original_texts[i],
            }
            if self.truths:
                item['truth'] = self.truths[i]
            if self.results:
                item['results'] = self.results[i]
            data.append(item)
        
        with open(filename, 'w') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
