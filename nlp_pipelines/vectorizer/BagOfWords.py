from sklearn.feature_extraction.text import CountVectorizer
from nlp_pipelines._base.BaseVectorizer import BaseVectorizer

class BagOfWords(BaseVectorizer):
    def __init__(self, supervised=False):
        super().__init__(supervised)
        self.vectorizer = CountVectorizer()

    def fit(self, dataset):
        """
        Fits the BoW model to the dataset, learning vocabulary and tokenization.
        """
        super().fit(dataset)
        self.vectorizer.fit(dataset.texts)

    def transform(self, dataset):
        if not self.is_fit:
            raise RuntimeError("Vectorizer must be fit before transforming.")
        
        # Perform the transformation using BoW (using the learned vocabulary)
        dataset.vectors = self.vectorizer.transform(dataset.texts).toarray()
        return dataset