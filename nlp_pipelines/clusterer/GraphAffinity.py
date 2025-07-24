from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import SpectralClustering
import networkx as nx
from nlp_pipelines._base.BaseMethod import BaseMethod

class GraphAffinity(BaseMethod):
    """
    Uses affinity in a embedding distance generated graph for clustering.
    """
    def __init__(self, max_features=5000, ngram_range=(1, 2), similarity_threshold=0.2):
        super().__init__(method_type='clusterer', supervised=False)
        self.method_name = "Graph Affinity Clustering"
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.similarity_threshold = similarity_threshold
        self.clustering_model = None # made in fit
        self.num_clusters = None
        self.possible_labels = None # add in fit
    
    def fit(self, dataset, possible_labels=[], num_clusters=None):
        """
        Fits the model on the training data (X_train).

        Args:
            X_train (list): A list of documents (text data) to train the model.
            y_train (list, optional): Labels for the training data. Defaults to None.
            possible_labels (list, optional): A list of possible labels for prediction. Defaults to None.

        Returns:
            None
        """
        if dataset.vectors is None:
            raise ValueError("Dataset for GraphAffinity needs vectors. Use a vectorizer.")
        self.possible_labels = possible_labels
        # Compute cosine similarity between documents
        cosine_sim = cosine_similarity(dataset.vectors)
        
        # Build graph where nodes are documents & edges are weighted by similarity
        G = nx.Graph()
        for i in range(len(dataset.texts)):
            for j in range(i + 1, len(dataset.texts)):  # Avoid duplicate edges
                if cosine_sim[i, j] > self.similarity_threshold:  # Similarity threshold
                    G.add_edge(i, j, weight=cosine_sim[i, j])
        
        # Spectral Clustering on Graph Similarity Matrix
        if num_clusters:
            self.num_clusters = num_clusters
        else:
            self.num_clusters = len(self.possible_labels)
        self.clustering_model = SpectralClustering(n_clusters=self.num_clusters, affinity="precomputed", assign_labels="discretize")
        self.clustering_model.fit(cosine_sim)
        self.is_fit = True
    
    def predict(self, dataset):
        """
        Predict the class for a given set of documents in a dataset.

        Args:
            dataset: a dataset with vectors

        Returns:
            dataset: the above dataset with added results
        """
        if not self.is_fit:
            raise RuntimeError("Methods must be fit before running predict.")
        if dataset.vectors is None:
            raise ValueError("Dataset for GraphAffinity needs vectors. Use a vectorizer.")
        cosine_sim = cosine_similarity(dataset.vectors)
        dataset.results = self.clustering_model.fit_predict(cosine_sim)
        return dataset
