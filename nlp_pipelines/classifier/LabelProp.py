from sklearn.semi_supervised import LabelPropagation
from sklearn.preprocessing import LabelEncoder
from nlp_pipelines._base.BaseMethod import BaseMethod

class LabelProp(BaseMethod):
    def __init__(self, n_neighbors=10):
        super().__init__(method_type="classifier", supervised=False)
        self.method_name = "Label Propagation"
        self.n_neighbors = n_neighbors
        self.lp_model = LabelPropagation(kernel="knn", n_neighbors=self.n_neighbors)
        self.le = LabelEncoder()
        self.possible_labels = None
        self.train_requires_truths = True
        self.requires_vectors = True

    def fit(self, dataset, possible_labels=[]):
        """
        Fits the the label propogation and label encoder on a labeled and vectorized set of training data.

        Args:
            dataset: a dataset with labels and vectors
            possible_labels (list, optional): A list of possible labels for prediction. Defaults to None.

        Returns:
            None
        """
        if dataset.vectors is None:
            raise ValueError("Dataset for LabelProp needs vectors. Use a vectorizer.")
        if dataset.truths is None:
            raise ValueError("Dataset for LabelProp needs truth labels for fitting")
        self.possible_labels = possible_labels
        encoded_labels = self.le.fit_transform(dataset.truths)
        self.lp_model.fit(dataset.vectors, encoded_labels)
        self.is_fit = True

    def predict(self, dataset):
        """
        Predict the class for a given set of documents in a dataset.

        Args:
            dataset: a dataset with vectors

        Returns:
            Dataset: the above dataset with added results
        """
        if not self.is_fit:
            raise RuntimeError("Methods must be fit before running predict.")
        if dataset.vectors is None:
            raise ValueError("Dataset for LabelProp needs vectors. Use a vectorizer.")
        # Propagate labels based on the trained model
        predicted_encoded_labels = self.lp_model.predict(dataset.vectors)
        # map them back
        dataset.results = self.le.inverse_transform(predicted_encoded_labels)
        return dataset