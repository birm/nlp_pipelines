from abc import ABC, abstractmethod
import logging



class BaseMethod(ABC):

    __SUPPORTED_METHOD_TYPES = {
    'clusterer',
    'classifier',
    'labeler',
    'topics'
    }
    
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
        self.supervised = supervised
        if not type(method_type) is str:
            raise ValueError("Method type must be a string from the list of supported types: {self.__SUPPORTED_METHOD_TYPES}")
        self.method_type = method_type.lower()
        if not method_type in self.__SUPPORTED_METHOD_TYPES:
            raise ValueError(f'{method_type} not supported. Supported methods include {self.__SUPPORTED_METHOD_TYPES}')
        # metadata fields
        self.method_name = "BaseMethod"
        self.is_fit = False
        self.possible_labels = []
        self.train_requires_truths = False
        self.requires_vectors = False
        self.requires_embed_possible_labels = False

    def fit(self, dataset, possible_labels=[]):
        """
        Fits the model on the training data.

        Args:
            dataset: a text dataset object with at least the .texts; other attributes may be required by some methods
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
            dataset: a text dataset object with at least the .texts; other attributes may be required by some methods

        Returns:
            dataset: A dataset with .results set
        """
        return dataset