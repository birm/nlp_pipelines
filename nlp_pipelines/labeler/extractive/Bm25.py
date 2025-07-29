from rank_bm25 import BM25Okapi
from nlp_pipelines._base.BaseMethod import BaseMethod
import numpy as np
from collections import Counter

class Bm25(BaseMethod):
    def __init__(self, top_k=10):
        super().__init__(method_type="labeler")
        self.method_name = "BestMatch-25"
        self.top_k = top_k
        self.bm25 = None
        self.possible_labels = []

    def fit(self, dataset, possible_labels=[]):
        self.possible_labels = [l.lower() for l in possible_labels]
        if len(self.possible_labels) > 0:
            # Pre-tokenize the candidate keywords
            self.tokenized_candidates = [kw.split() for kw in self.possible_labels]
            self.bm25 = BM25Okapi(self.tokenized_candidates)
        else:
            self.bm25 = None
        self.is_fit = True

    def predict(self, dataset):
        if not self.is_fit:
            raise RuntimeError("Methods must be fit before running predict.")

        keywords = []
        if self.bm25 is None:
            # No labels given: fallback to top-k frequent tokens per doc
            for doc in dataset.texts:
                tokens = doc.lower().split()
                counter = Counter(tokens)
                top_tokens = [token for token, _ in counter.most_common(self.top_k)]
                keywords.append(top_tokens)
        else:
            # Use BM25 with given labels
            for doc in dataset.texts:
                query = doc.lower().split()
                scores = self.bm25.get_scores(query)
                top_indices = np.argsort(scores)[::-1][:self.top_k]
                y_i = [(self.possible_labels[i], float(scores[i])) for i in top_indices]
                keywords.append([kwd for kwd, score in y_i])

        dataset.results = keywords
        return dataset
