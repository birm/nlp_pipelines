import yake

from nlp_pipelines._base.BaseMethod import BaseMethod

class Yake(BaseMethod):
    def __init__(self, top_k=10):
        super().__init__(method_type="labeler", supervised=False)
        self.extractor = yake.KeywordExtractor()
        self.top_k = top_k

    def predict(self, dataset):
        keywords = []
        for row in dataset.texts:
            y_i = self.extractor.extract_keywords(row)
            keywords.append([kwd for kwd, _ in y_i[:self.top_k]])
        dataset.results = keywords
