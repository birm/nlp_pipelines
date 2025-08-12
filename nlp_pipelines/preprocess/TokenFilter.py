import spacy
import spacy.cli
from nlp_pipelines.dataset import Dataset
from nlp_pipelines._base.BasePreprocessor import BasePreprocessor

# remove things if is_X is set for user choices of X
# see https://spacy.io/api/token/#attributes
# you should only use properties which will be true or false. Filtered if true.

class TokenFilter(BasePreprocessor):
    def __init__(self, model_name="en_core_web_sm", remove_if=None):
        """
        Initialize the TokenFilter.

        Args:
            model_name (str): Name of the spaCy model to use.
            remove_if (list): List of `is_X` of `like_x` attributes as strings (e.g., ["is_stop", "is_punct"]).
        """
        super().__init__()
        if remove_if is None:
            remove_if = ["is_stop", "is_punct", "is_space", "like_num", "like_url"]  # Default things to remove

        self.model_name = model_name
        self.nlp = None
        self.remove_if = set(remove_if)
        self._ensure_spacy_model()
        self.method_name = f"Customized Token Filter removing: {remove_if}"

    def _ensure_spacy_model(self):
        """Ensure the required spaCy model is installed."""
        try:
            self.nlp = spacy.load(self.model_name)
        except OSError:
            print(f"Model '{self.model_name}' not found. Installing...")
            spacy.cli.download(self.model_name)
            self.nlp = spacy.load(self.model_name)

    def transform(self, dataset: Dataset) -> Dataset:
        """
        Removes tokens from the dataset's texts based on the `remove_if` attributes.

        Args:
            dataset (Dataset): The Dataset object to be transformed.

        Returns:
            Dataset: The transformed dataset with unwanted tokens removed.
        """
        cleaned_texts = []

        for text in dataset.texts:
            doc = self.nlp(text)
            cleaned_text = " ".join([
                token.text for token in doc
                if not any(getattr(token, condition) for condition in self.remove_if)
            ])
            cleaned_texts.append(cleaned_text)

        dataset.texts = cleaned_texts
        return dataset
