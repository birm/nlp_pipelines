from multi_rake import Rake

from nlp_pipelines._base.BaseMethod import BaseMethod

class MultiRake(BaseMethod):
    def __init__(self, top_k=10):
        super().__init__(method_type="labeler", supervised=False)
        self.method_name = "MultiRake"
        self.top_k = top_k
        self.extractor = Rake()

    def predict(self, dataset):
        keywords = []
        for row in dataset.texts:
            y_i = self.extractor.apply(row)
            keywords.append([kwd for kwd, score in y_i[:self.top_k]])
        dataset.results = keywords
        return dataset
