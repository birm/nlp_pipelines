from nlp_pipelines._base.BaseMethod import BaseMethod
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MultiLabelBinarizer

class MultiLogistic(BaseMethod):
    """
    Multi-label Logistic Regression method for predicting relevant keywords for a document. (Low expectations due to combinatorial explosion of MultiLabelBinarizer/possible_labels)
    """

    def __init__(self, similarity_method="cosine", threshold=0.5):
        """
        Initialize the MultiLogistic method using logistic regression for multi-label classification.

        Args:
            similarity_method (str): Similarity measure to use ('cosine', 'l2', 'euclidean', etc.).
            threshold (float): The threshold for classifying a keyword as relevant (probability > threshold).
        """
        super().__init__(method_type="labeler", supervised=True)
        self.method_name = "Multi-label Logistic Regression Keyword Prediction"
        self.similarity_method = similarity_method
        self.threshold = threshold
        self.is_fitted = False
        self.possible_labels = []
        self.possible_labels_embed = []
        self.lr_model = None
        self.mlb = MultiLabelBinarizer()
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
        
        # Encode the ground truth labels as binary vectors (multi-label)
        y_train = self.mlb.fit_transform(dataset.truths)  # MultiLabelBinarizer converts to 0/1

        # Prepare the feature matrix (X_train) from document embeddings
        X_train = dataset.vectors

        # Train the logistic regression model (one per candidate label)
        self.lr_model = LogisticRegression(max_iter=1000)
        self.lr_model.fit(X_train, y_train)  # Fit on document embeddings with binary labels
        self.is_fitted = True

    def predict(self, dataset):
        """
        Predict relevant keywords for each document in the dataset.

        Args:
            dataset: The dataset containing the document embeddings.

        Returns:
            dataset: The dataset with the predicted keywords.
        """
        if not self.is_fitted:
            raise RuntimeError("Methods must be fit before running predict.")
        
        if dataset.vectors is None:
            raise ValueError("Dataset for MultiLogistic needs vectors. Use a vectorizer.")

        # Get predictions (probabilities for each label)
        predicted_probs = self.lr_model.predict_proba(dataset.vectors)

        # Get the predicted keywords based on the threshold
        predictions = []
        for prob in predicted_probs:
            predicted_labels = [
                self.possible_labels[idx] for idx, p in enumerate(prob) if p >= self.threshold
            ]
            predictions.append(predicted_labels)

        # Store the predictions in the dataset
        dataset.results = predictions
        return dataset
