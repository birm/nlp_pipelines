from sentence_transformers import SentenceTransformer
from nlp_pipelines._base.BaseVectorizer import BaseVectorizer

class SentenceEmbedding(BaseVectorizer):
    def __init__(self, model_name='all-MiniLM-L6-v2', supervised=False):
        super().__init__(supervised)
        # Load the Hugging Face model
        self.model = SentenceTransformer(model_name)

    def transform(self, dataset):
        if not self.is_fit:
            raise RuntimeError("Vectorizer must be fit before transforming.")
        
        dataset.vectors = self.model.encode(dataset.texts, convert_to_tensor=True)
        dataset.vectors = dataset.results.cpu().numpy()
        return dataset