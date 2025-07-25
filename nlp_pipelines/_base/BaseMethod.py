import pickle
from abc import ABC, abstractmethod
import logging

__SUPPORTED_METHOD_TYPES = {
    'clusterer',
    'classifier',
    'labeler',
    'topics'
    }

class BaseMethod(ABC):
    """
    Base class for nlp pipeline methods.
    Univerally, use fit first then predict, at least to set up variables internally, even for methods which do not actually fit.
    """
    def __init__(self, method_type, supervised=False):
        """
        Initializes the base clustering model.

        Args:
            method_type (str): Describes which method type this is. Must be among supported.
            supervised (bool): Indicates if the model is supervised. Defaults to False.

        """
        # logging
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Initialized BartTag")
        self.supervised = supervised
        if not type(method_type) is str:
            raise ValueError("Method type must be a string from the list of supported types: {__SUPPORTED_METHOD_TYPES}")
        self.method_type = method_type.lower()
        if not method_type in __SUPPORTED_METHOD_TYPES:
            raise ValueError(f'{method_type} not supported. Supported methods include {__SUPPORTED_METHOD_TYPES}')
        # metadata fields
        self.method_name = "BaseMethod"
        self.is_fit = False
        self.possible_labels = []

    def fit(self, dataset, possible_labels=[]):
        """
        Fits the model on the training data (X_train).

        Args:
            X_train (list): A list of documents (text data) to train the model.
            y_train (list, optional): Labels for the training data. Defaults to None.
            possible_labels (list, optional): A list of possible labels for prediction. Defaults to None.

        Returns:
            None
        """
        self.possible_labels = possible_labels  # so predict doesn't ever need this arg
        self.is_fit = True
    
    @abstractmethod
    def predict(self, dataset):
        """
        Predict the output for a given set of text samples.

        Args:
            X (list): A list of documents (text data) to predict the labels for.

        Returns:
            list: A list of predicted labels corresponding to each document in X.
        """
        return dataset