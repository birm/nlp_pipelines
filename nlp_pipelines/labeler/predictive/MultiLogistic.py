from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multioutput import MultiOutputClassifier
import numpy as np

from nlp_pipelines._base.BaseMethod import BaseMethod


class MultiLogistic(BaseMethod):
    """
    Multi-class Logistic Regression method for predicting relevant keywords for a document.
    """

    def __init__(self, similarity_method="cosine", threshold=0.5):
        """
        Initialize the MultiLogistic method using logistic regression for multi-class classification.

        Args:
            similarity_method (str): Similarity measure to use ('cosine', 'l2', 'euclidean', etc.).
            threshold (float): The threshold for classifying a keyword as relevant (probability > threshold).
        """
        super().__init__(method_type="classifier", supervised=True)
        self.method_name = "Multi-class Logistic Regression Keyword Prediction"
        self.similarity_method = similarity_method
        self.threshold = threshold
        self.possible_labels = []
        self.possible_labels_embed = []
        self.lr_model = None
        self.mlb = MultiLabelBinarizer()  # Initialize MultiLabelBinarizer here
        self.train_requires_truths = True
        self.requires_vectors = True
        self.requires_embed_possible_labels = True

    def fit(self, dataset, possible_labels=[], possible_labels_embed=[]):
        """
        Fit the MultiLogistic model using the training data (document embeddings and their truth labels).

        Args:
            dataset: The dataset containing the document embeddings and the true labels.
            possible_labels (list): List of candidate labels (keywords).
            possible_labels_embed (list): List of embeddings corresponding to the candidate labels.
        """
        if len(possible_labels) == 0 or len(possible_labels_embed) == 0:
            raise ValueError("MultiLogistic requires a list of possible_labels and their embeddings.")
        
        self.possible_labels = [label.lower() for label in possible_labels]
        self.possible_labels_embed = possible_labels_embed

        # Ensure dataset has .truths (the ground truth labels)
        if not hasattr(dataset, 'truths') or not isinstance(dataset.truths, list):
            raise ValueError("Dataset must have .truths attribute (list of true labels for each document).")
        
        # MultiLabelBinarizer is used for multi-label classification
        y_train = self.mlb.fit_transform(dataset.truths)  # This converts multi-label lists into binary vectors

        # Prepare the feature matrix (X_train) from document embeddings
        X_train = np.array(dataset.vectors)

        # Initialize the logistic regression model for multi-label classification
        lr_model = LogisticRegression(max_iter=1000, multi_class='ovr', solver='lbfgs')

        # Use MultiOutputClassifier to handle multi-label classification
        self.lr_model = MultiOutputClassifier(lr_model)
        self.lr_model.fit(X_train, y_train)  # Fit on document embeddings with multi-label binary vectors
        self.is_fit = True


    def predict(self, dataset):
        """
        Predict relevant keywords for each document in the dataset.

        Args:
            dataset: The dataset containing the document embeddings.

        Returns:
            dataset: The dataset with the predicted keywords.
        """
        if not self.is_fit:
            raise RuntimeError("Methods must be fit before running predict.")
        
        if dataset.vectors is None:
            raise ValueError("Dataset for MultiLogistic needs vectors. Use a vectorizer.")

        # Get the predicted probabilities (one per label)
        predicted_probs = self.lr_model.predict_proba(np.array(dataset.vectors))

        # Get the predicted labels based on the threshold
        predictions = []
        for prob in predicted_probs:
            predicted_labels = [
                self.possible_labels[idx] 
                for idx, p in enumerate(prob) 
                if isinstance(p, (int, float)) and p >= self.threshold
            ]
            predictions.append(predicted_labels)

        # Now inverse transform the predictions back to the original multi-label format
        # Reverse the transformation from binary format back to label sets
        predictions = np.array(predictions)
        predicted_labels_original = self.mlb.inverse_transform(predictions)

        # Store the predictions in the dataset (now with original labels)
        dataset.results = predicted_labels_original
        return dataset
