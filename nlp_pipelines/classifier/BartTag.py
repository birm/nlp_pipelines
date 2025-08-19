from transformers import pipeline
from nlp_pipelines._base.BaseMethod import BaseMethod

# required input: dataset, optionally labeled. does not use vectors even if present.

class BartTag(BaseMethod):
    def __init__(self, threshold=0.5):
        super().__init__(method_type="classifier", supervised=False)
        self.method_name = "Zero-Shot BART"
        self.internal_model_name = "facebook/bart-large-mnli"
        self.classifier = pipeline("zero-shot-classification", model=self.internal_model_name)
        self.threshold = threshold
        self.requires_possible_labels = True

    # predict can use default method.
    
    def _get_class_bart(self, text):
        """Helper: Classifies text into relevant topics using zero-shot learning."""
        if not isinstance(text, str) or text.strip() == "":
            return ""
        try:
            result = self.classifier(text, candidate_labels=self.possible_labels)
            return result['labels'][0] # top label
        except Exception as e:
            self.logger.warning(f"Skipping text due to error: {e}")
            return ""

    def predict(self, dataset):
        if not self.is_fit:
            raise RuntimeError("Methods must be fit before running predict.")
        results = [self._get_class_bart(x) for x in dataset.texts]
        dataset.results = results

