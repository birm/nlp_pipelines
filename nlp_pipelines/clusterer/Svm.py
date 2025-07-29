from sklearn.svm import OneClassSVM
from nlp_pipelines._base.BaseMethod import BaseMethod


class Svm(BaseMethod):
    def __init__(self, kernel='rbf', gamma='scale', nu=0.5):
        super().__init__(method_type='clusterer', supervised=False)
        self.method_name = "SVM Clustering (One-Class)"
        # Initialize One-Class SVM
        self.cluster_model = OneClassSVM(kernel=kernel, gamma=gamma, nu=nu)
        self.requires_vectors = True
        
    def predict(self, dataset):
        if not self.is_fit:
            raise RuntimeError("Methods must be fit before running predict.")
        if dataset.vectors is None:
            raise ValueError("Dataset for SVMClustering needs vectors. Use a vectorizer.")
        
        # Fit the One-Class SVM and predict the clusters (1: inliers, -1: outliers)
        dataset.results = self.cluster_model.fit_predict(dataset.vectors)
        return dataset
