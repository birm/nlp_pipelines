from abc import ABC, abstractmethod
import logging
from nlp_pipelines.dataset import Dataset

class BaseVectorizer(ABC):
    """
    Base class for nlp pipeline vectorizers.
    Use fit first then transform. To set dataset.vectors as a list of vectors.
    """
    def __init__(self, supervised=False):
        """
        Initializes the base vectorizeer model.

        Args:
            supervised (bool): Indicates if the vectorizer is supervised. Defaults to False.

        """
        # logging
        self.logger = logging.getLogger(self.__class__.__name__)
        self.supervised = supervised
        self.method_type = "vectorizer"
        # metadata fields
        self.method_name = "BaseVectorizer"
        self.is_fit = False
        self.train_requires_truths = False # I don't think this should ever be true for vectorizer
        self.requires_vectors = False # also always false
        self.requires_embed_possible_labels = False

    def fit(self, dataset):
        """
        Fits the vectorizer on the training data, setting dataset.vectors.

        Args:
            dataset: a text dataset object with at least the .texts; other attributes may be required by some methods

        Returns:
            None
        """
        self.is_fit = True
    
    @abstractmethod
    def transform(self, dataset):
        """
        Transform the text to vector form.

        Args:
            dataset: a text dataset object with at least the .texts; other attributes may be required by some methods

        Returns:
            dataset: A dataset with .vectors set
        """
        return dataset
    
    def transform_labels(self, labels):
        """
        Transform the text of possible labels to vector form.

        Args:
            labels: a list of strings to vectorize

        Returns:
            dataset: A dataset with .vectors set
        """
        # make a dataset for just these labels
        label_dataset = Dataset(labels)
        # transform this dataset under vectorization
        label_dataset = self.transform(label_dataset)
        # then just get the vectors!
        return label_dataset.vectors