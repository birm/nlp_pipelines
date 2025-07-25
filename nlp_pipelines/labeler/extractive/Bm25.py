from rank_bm25 import BM25Okapi
from nlp_pipelines._base.BaseMethod import BaseMethod
import numpy as np

class Bm25(BaseMethod):
    def __init__(self, top_k=10):
        super().__init__(method_type="labeler")
        self.method_name = "BestMatch-25"
        self.top_k = top_k

    def fit(self, dataset, possible_labels=None):
        if possible_labels is None:
            raise ValueError("BM25Extract requires a list of possible_labels.")
        self.possible_labels = [l.lower() for l in possible_labels]
        # Pre-tokenize the keyword candidates
        self.tokenized_candidates = [kw.split() for kw in self.possible_labels]
        self.bm25 = BM25Okapi(self.tokenized_candidates)
        self.is_fit = True

    def predict(self, dataset):
        if not self.is_fit:
            raise RuntimeError("Methods must be fit before running predict.")
        keywords = []
        for doc in dataset.texts:
            query = doc.split() # todo, maybe use nltk or spacy?
            scores = self.bm25.get_scores(query)
            top_indices = np.argsort(scores)[::-1][:self.top_k]
            y_i = [(self.possible_labels[i], float(scores[i])) for i in top_indices]
            keywords.append([kwd for kwd, score in y_i])
        dataset.results = keywords
        return dataset
