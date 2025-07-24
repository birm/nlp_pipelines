import hdbscan
import umap
from nlp_pipelines._base.BaseMethod import BaseMethod


class UmapHdbscan(BaseMethod):
    def __init__(self):
        super().__init__(method_type='clusterer', supervised=False)
        self.method_name = "Umap With HDBScan"
        #load the umap model
        self.umap_model = umap.UMAP(n_neighbors=15, n_components=5, metric="cosine")
        self.cluster_model = hdbscan.HDBSCAN(min_cluster_size=10, metric="euclidean", cluster_selection_method="eom", prediction_data=True)

    # fit not needed, leave as pass from abc

    def predict(self, dataset):
        if dataset.vectors is None:
            raise ValueError("Dataset for UmapHdbscan needs vectors. Use a vectorizer.")
        # Reduce dimensionality with UMAP
        umapped = self.umap_model.fit_transform(dataset.vectors)
        # Cluster using HDBSCAN
        dataset.results = self.cluster_model.fit_predict(umapped)
        return dataset