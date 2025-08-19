from transformers import pipeline
from nlp_pipelines._base.BaseMethod import BaseMethod

# required input: dataset, optionally labeled. does not use vectors even if present.

class BartTagLabeler(BaseMethod):
    def __init__(self, threshold=0.5):
        super().__init__(method_type="classifier", supervised=False)
        self.method_name = "Zero-Shot BART"
        self.internal_model_name = "facebook/bart-large-mnli"
        self.classifier = pipeline("zero-shot-classification", model=self.internal_model_name)
        self.threshold = threshold
        self.requires_possible_labels = True

    # predict can use default method.
    
    def _generate_tags_bart(self, text):
        """Helper: Classifies text into relevant topics using zero-shot learning."""
        if not isinstance(text, str) or text.strip() == "":
            return []
        try:
            result = self.classifier(text, candidate_labels=self.possible_labels)
            return [label for label, score in zip(result["labels"], result["scores"]) if score > self.threshold]
        except Exception as e:
            self.logger.warning(f"Skipping text due to error: {e}")
            return []

    def predict(self, dataset):
        if not self.is_fit:
            raise RuntimeError("Methods must be fit before running predict.")
        dataset.results = [self._generate_tags_bart(x) for x in dataset.texts]
        return dataset
