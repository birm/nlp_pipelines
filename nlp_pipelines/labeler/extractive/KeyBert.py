from keybert import KeyBERT
from nlp_pipelines._base.BaseMethod import BaseMethod

class KeyBert(BaseMethod):
    def __init__(self, top_k=10, include_scores=False):
        super().__init__(method_type="labeler", supervised=False)
        self.extractor = KeyBERT()
        self.include_scores = include_scores
        self.top_k = top_k

    def fit(self, dataset, possible_labels=[]):
        self.possible_labels = [l.lower() for l in possible_labels]
        self.is_fit = True

    def predict(self, dataset):
        if not self.is_fit:
            raise RuntimeError("call `.fit()` before `.predict()`.")
        keywords = []
        for row in dataset.texts:
            if self.possible_labels:
                y_i = self.extractor.extract_keywords(row.lower(), candidates=self.possible_labels)
                if not y_i:
                    print("WARN: Found No Matches in possible keywords.")
                    y_i = self.extractor.extract_keywords(row.lower())
            else:
                y_i = self.extractor.extract_keywords(row.lower())
            if self.include_scores:
                keywords.append([(kwd, score) for kwd, score in y_i[:self.top_k]])
            else:
                keywords.append([kwd for kwd, score in y_i[:self.top_k]])
        dataset.results= keywords
        return dataset
