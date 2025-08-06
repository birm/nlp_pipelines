# does this need a redesign, or it it special enough to actually get its own vectorizer out of necessiry?
from sklearn.feature_extraction.text import TfidfVectorizer
from nlp_pipelines._base.BaseMethod import BaseMethod


class TfidfTopN(BaseMethod):
    def __init__(self, top_k=10, ngram_range=(1, 3), stop_words_set='english'):
        super().__init__(method_type="labeler", supervised=False)
        self.method_name = "Top N words by TFIDF"
        self.top_k = top_k
        self.vectorizer = TfidfVectorizer(ngram_range=ngram_range, stop_words=stop_words_set)

    def fit(self, dataset, possible_labels=None):
        self.vectorizer.fit(dataset.texts)
        self.is_fit = True

    def predict(self, dataset):
        if not self.is_fit:
            raise RuntimeError("Call fit() before predict().")

        X_tfidf = self.vectorizer.transform(dataset.texts)
        feature_names = self.vectorizer.get_feature_names_out()
        keywords = []

        for row in X_tfidf:
            row_array = row.toarray().flatten()
            indices = row_array.argsort()[-self.top_k:][::-1]
            keywords.append([
                feature_names[i]
                for i in indices
            ])
        dataset.results = keywords
        return dataset
