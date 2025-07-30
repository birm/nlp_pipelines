import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from nlp_pipelines._base.BaseMethod import BaseMethod

class Xgboost(BaseMethod):
    def __init__(self):
        super().__init__(method_type="classifier", supervised=True)
        self.method_name = "XGBoost Classifier"
        self.xgboost_model = xgb.XGBClassifier(eval_metric='mlogloss')
        self.possible_labels = None
        self.le = LabelEncoder()
        self.train_requires_truths = True
        self.requires_vectors = True

    def fit(self, dataset, possible_labels=[]):
        if dataset.vectors is None:
            raise ValueError("Dataset for LabelProp needs vectors. Use a vectorizer.")
        if dataset.truths is None:
            raise ValueError("Dataset for LabelProp needs truth labels for fitting.")
        encoded_labels = self.le.fit_transform(dataset.truths)
        self.xgboost_model.fit(dataset.vectors, encoded_labels)
        self.possible_labels = possible_labels
        self.is_fit = True

    def predict(self, dataset):
        if not self.is_fit:
            raise RuntimeError("Methods must be fit before running predict.")
        if dataset.vectors is None:
            raise ValueError("Dataset for LabelProp needs vectors. Use a vectorizer.")
        predicted_encoded_labels = self.xgboost_model.predict(dataset.vectors)
        predicted_labels = self.le.inverse_transform(predicted_encoded_labels)
        dataset.results = predicted_labels
