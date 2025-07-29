import spacy
import spacy.cli
from nlp_pipelines.dataset import Dataset
from nlp_pipelines._base.BasePreprocessor import BasePreprocessor

class PosRemoval(BasePreprocessor):
    def __init__(self, model_name="en_core_web_sm", pos_tags_to_remove=[]):
        super().__init__()
        self.model_name = model_name
        self.method_name = "Part of Speech Filtering"
        self.nlp = None
        self._ensure_spacy_model()
        self.pos_tags_to_remove = pos_tags_to_remove

    def _ensure_spacy_model(self):
        try:
            self.nlp = spacy.load(self.model_name)
        except OSError:
            print(f"Model '{self.model_name}' not found. Installing...")
            spacy.cli.download(self.model_name)
            self.nlp = spacy.load(self.model_name)

    def transform(self, dataset: Dataset) -> Dataset:
        cleaned_texts = []
        
        for text in dataset.texts:
            doc = self.nlp(text)
            cleaned_text = " ".join([token.text for token in doc if token.pos_ not in self.pos_tags_to_remove and not token.is_space and not token.is_punct])
            cleaned_texts.append(cleaned_text)
        
        dataset.texts = cleaned_texts
        return dataset
