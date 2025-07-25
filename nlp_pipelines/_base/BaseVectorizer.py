from abc import ABC, abstractmethod
import logging

class BaseVectorizer(ABC):
    """
    Base class for nlp pipeline vectorizers.
    Use fit first then transform. To set dataset.vectors as a list of vectors.
    """
    def __init__(self, supervised=False):
        """
        Initializes the base clustering model.

        Args:
            supervised (bool): Indicates if the vectorizer is supervised. Defaults to False.

        """
        # logging
        self.logger = logging.getLogger(self.__class__.__name__)
        self.supervised = supervised
        # metadata fields
        self.method_name = "BaseVectorizer"
        self.is_fit = False

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