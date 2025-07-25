from sklearn.feature_extraction.text import TfidfVectorizer
from nlp_pipelines._base.BaseVectorizer import BaseVectorizer

class Tfidf(BaseVectorizer):
    def __init__(self, supervised=False):
        super().__init__(supervised)
        self.vectorizer = TfidfVectorizer()

    def fit(self, dataset):
        """
        Fits the TF-IDF model to the dataset, computing the IDF values.
        """
        super().fit(dataset)
        self.vectorizer.fit(dataset.texts)

    def transform(self, dataset):
        if not self.is_fit:
            raise RuntimeError("Vectorizer must be fit before transforming.")
        
        # Perform the transformation using TF-IDF (using the learned IDF values)
        dataset.vectors = self.vectorizer.transform(dataset.texts).toarray()
        return dataset