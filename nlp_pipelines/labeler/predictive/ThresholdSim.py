#!! PROBLEM; this method needs the vectorizer to also vectorize the candidates!
# keep this in mind to fix vextorizers to support this too.
from nlp_pipelines._base.BaseMethod import BaseMethod
import torch
import numpy as np

class ThresholdSim(BaseMethod):

    __SUPPORTED_METHOD_TYPES = ['cosine', 'l2', 'euclidean', 'l1', 'manhattan', 'jaccard', 'hamming']

    def __init__(self, similarity_method="cosine", include_scores=False):
        super().__init__(method_type="labeler", supervised=False)
        self.method_name = "Selection via Thresholded Similarity"
        if similarity_method not in self.__SUPPORTED_METHOD_TYPES:
            raise ValueError(f"Invalid method type '{similarity_method}'. Supported methods are: {', '.join(self.__SUPPORTED_METHOD_TYPES)}")
        self.similarity_method = similarity_method
        self.requires_vectors = True
        self.requires_embed_possible_labels = True
        
        
    def fit(self, dataset, possible_labels=[], possible_labels_embed=[]):
        if len(possible_labels) == 0 or len(possible_labels_embed) == 0:
            raise ValueError("ThresholdSim requires a list of possible_labels.")
        self.possible_labels = [label.lower() for label in possible_labels]
        # Embed the keywords
        self.possible_labels_embed = possible_labels_embed
        self.is_fit = True
    
    def compute_similarity(self, doc_embedding, keyword_embedding, similarity_method="cosine"):
        if isinstance(doc_embedding, np.ndarray):
            doc_embedding = torch.tensor(doc_embedding)
        if isinstance(keyword_embedding, np.ndarray):
            keyword_embedding = torch.tensor(keyword_embedding)
        # Ensure both embeddings are at least 2D tensors (shape: [1, d])
        doc_embedding = doc_embedding.unsqueeze(0) if doc_embedding.dim() == 1 else doc_embedding
        keyword_embedding = keyword_embedding.unsqueeze(0) if keyword_embedding.dim() == 1 else keyword_embedding
        res = None
        # Cosine similarity
        if similarity_method == "cosine":
            res = torch.nn.functional.cosine_similarity(doc_embedding, keyword_embedding).item()
        
        # L2 (Euclidean) distance, converted to similarity
        elif similarity_method in ["l2", "euclidean"]:
            distance = torch.nn.functional.pairwise_distance(doc_embedding, keyword_embedding).item()
            res = 1 / (1 + distance)
        
        # L1 (Manhattan) distance, converted to similarity
        elif similarity_method in ["l1", "manhattan"]:
            distance = torch.sum(torch.abs(doc_embedding - keyword_embedding)).item()
            res = 1 / (1 + distance)
        
        # Jaccard similarity
        elif similarity_method == "jaccard":
            intersection = torch.sum(torch.min(doc_embedding, keyword_embedding)).item()
            union = torch.sum(torch.max(doc_embedding, keyword_embedding)).item()
            res = intersection / union if union != 0 else 0
        
        # Hamming distance, converted to similarity
        elif similarity_method == "hamming":
            distance = torch.sum(doc_embedding != keyword_embedding).item()
            res = 1 / (1 + distance)
        
        return res


    def predict(self, dataset):
        if not self.is_fit:
            raise RuntimeError("Methods must be fit before running predict.")
        if dataset.vectors is None:
            raise ValueError("Dataset for ThresholdSim needs vectors. Use a vectorizer.")
        keywords = []
        
        for doc_embedding in dataset.vectors:
            
            # Compute the similarity between the document and each keyword
            keyword_scores = []
            for idx, keyword_embedding in enumerate(self.possible_labels_embed):
                similarity = self.compute_similarity(doc_embedding, keyword_embedding, similarity_method=self.similarity_method)
                keyword_scores.append((self.possible_labels[idx], similarity))
            
            # Sort keywords by their similarity scores
            sorted_keywords = sorted(keyword_scores, key=lambda x: x[1], reverse=True)
            keywords.append([kw for kw, _ in sorted_keywords])
        dataset.results = keywords
        return dataset
