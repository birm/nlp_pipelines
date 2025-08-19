from abc import ABC, abstractmethod
import logging

class BasePreprocessor(ABC):
    """
    Base class for nlp pipeline vectorizers.
    Use fit first then transform. To set dataset.vectors as a list of vectors.
    """
    def __init__(self, supervised=False):
        """
        Initializes the base preprocessing class.

        """
        # logging
        self.logger = logging.getLogger(self.__class__.__name__)
        self.supervised = supervised
        self.method_type = "processor"
        # metadata fields
        self.method_name = "BasePreprocessor"
        self.is_fit = True # secret; don't need to fit preprocessors but can still call it
        self.train_requires_truths = False
        self.requires_vectors = False
        self.requires_embed_possible_labels = False
        self.requires_possible_labels = False

    def fit(self, dataset):
        """
        Should never be needed truly, but included for compatibility as if a vectorizer.
        """
        self.is_fit = True
    
    @abstractmethod
    def transform(self, dataset):
        """
        Transform the text per the preprocessing rules.
        """
        return dataset