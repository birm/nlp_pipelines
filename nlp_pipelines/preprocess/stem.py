from nlp_pipelines.dataset import Dataset
from nltk.stem import PorterStemmer
from nlp_pipelines._base.BasePreprocessor import BasePreprocessor

class Stem(BasePreprocessor):
    def __init__(self):
        super().__init__()
        self.stemmer = PorterStemmer()
        self.method_name = "Stemming"

    def transform(self, dataset: Dataset) -> Dataset:
        stemmed_texts = []
    
        for text in dataset.texts:
            # Tokenize and stem each word in the text
            stemmed_text = " ".join([self.stemmer.stem(word) for word in text.split()])
            stemmed_texts.append(stemmed_text)
        
        dataset.texts = stemmed_texts
        return dataset
