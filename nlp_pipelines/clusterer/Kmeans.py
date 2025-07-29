from sklearn.cluster import KMeans
from nlp_pipelines._base.BaseMethod import BaseMethod


class Kmeans(BaseMethod):
    def __init__(self, num_clusters=3, random_state=None):
        super().__init__(method_type='clusterer', supervised=False)
        self.method_name = "KMeans Clustering"
        # Initialize KMeans with the specified number of clusters
        self.cluster_model = KMeans(n_clusters=num_clusters, random_state=random_state)
        self.requires_vectors = True
        
    def predict(self, dataset):
        if not self.is_fit:
            raise RuntimeError("Methods must be fit before running predict.")
        if dataset.vectors is None:
            raise ValueError("Dataset for KMeansClustering needs vectors. Use a vectorizer.")
        
        # Fit and predict the clusters
        dataset.results = self.cluster_model.fit_predict(dataset.vectors)
        return dataset
