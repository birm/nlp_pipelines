import fasttext
import numpy as np
import re
import io

from nlp_pipelines._base.BaseVectorizer import BaseVectorizer



class FastText(BaseVectorizer):

    __SUPPORTED_UNSUP_METHODS = {"skipgram", "cbow"}

    def __init__(self, model_path=None, supervised=False, unsupervised_method="skipgram"):
        super().__init__(supervised)
        self.model_path = model_path
        self.method_name = f"FastText: {"Supervised" if supervised else "Unsupervised"}"
        if not self.unsupervised_method in self.__SUPPORTED_UNSUP_METHODS:
            raise RuntimeError(f"FastText: unsupervised_method {unsupervised_method} not supported; pick one of {self.__SUPPORTED_UNSUP_METHODS}")
        self.unsupervised_method = unsupervised_method
        self.model = None

    def clean_text(self, text):
        text = text.replace('\n', ' ').replace('\r', ' ').strip()
        # Remove excessive whitespace (e.g., multiple spaces between words)
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters or punctuation (optional, depending on your use case)
        text = re.sub(r'[^\w\s]', '', text)  # Keeps only alphanumeric characters and spaces
        text = text.lower()
        return text        

    def fit(self, dataset, model_path=None):
        if self.model_path or model_path:  # Load pre-trained model
            model_path = model_path or self.model_path
            self.model = fasttext.load_model(model_path)
            self.is_fit = True
            self.supervised = False
        elif self.supervised:
            # Clean text and prepare training data
            if not hasattr(dataset, 'truths'):
                raise ValueError("Supervised training requires labels (truths) in the dataset.")
            
            training_data = [f"__label__{label} {self.clean_text(text)}" for text, label in zip(dataset.texts, dataset.truths)]
            
            # Use StringIO to simulate a file-like object in memory
            training_data_str = "\n".join(training_data)
            training_data_io = io.StringIO(training_data_str)
            
            # Train the supervised model
            self.model = fasttext.train_supervised(training_data_io)
            self.is_fit = True
        else:
            # Clean text and prepare training data
            training_data = [f"{self.clean_text(text)}" for text in dataset.texts]
            
            # Use StringIO to simulate a file-like object in memory
            training_data_str = "\n".join(training_data)
            training_data_io = io.StringIO(training_data_str)
            
            # Train the unsupervised model
            self.model = fasttext.train_unsupervised(training_data_io, model=self.unsupervised_method)
            self.is_fit = True


    def transform(self, dataset):
        if not self.is_fit:
            raise RuntimeError("Vectorizer must be fit before transforming.")
        dataset.vectors = np.array([
            np.mean([self.model.get_word_vector(word) for word in text.split() if word in self.model] or [np.zeros(self.model.get_dimension())], axis=0)
            for text in dataset.texts
        ])
        return dataset
