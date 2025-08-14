from nlp_pipelines._base.BaseMethod import BaseMethod
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

class MultiLogistic(BaseMethod):

    def __init__(self, threshold=0.5, epochs=10, lr=1e-3, batch_size=32):
        super().__init__(method_type="labeler", supervised=True)
        self.input_dim = None
        self.threshold = threshold
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None  # Built in fit when we know output size

        self.mlb = MultiLabelBinarizer()
        self.is_fit = False

    def fit(self, dataset, possible_labels=None):
        if dataset.vectors is None or dataset.truths is None:
            raise ValueError("Dataset must have .vectors and .labels")

        self.input_dim = np.shape(dataset.vectors)[1]
        y_bin = self.mlb.fit_transform(dataset.truths)
        num_classes = y_bin.shape[1]

        X = torch.tensor(dataset.vectors, dtype=torch.float32).to(self.device)
        y = torch.tensor(y_bin, dtype=torch.float32).to(self.device)

        # Simple logistic regression model
        self.model = nn.Linear(self.input_dim, num_classes).to(self.device)

        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        self.model.train()
        for epoch in range(self.epochs):
            perm = torch.randperm(X.size(0))
            X = X[perm]
            y = y[perm]
            for i in range(0, X.size(0), self.batch_size):
                xb = X[i:i+self.batch_size]
                yb = y[i:i+self.batch_size]
                logits = self.model(xb)
                loss = self.criterion(logits, yb)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        self.is_fit = True
        self.model.eval()

    def predict(self, dataset):
        if not self.is_fit:
            raise RuntimeError("Model must be fit before predict")
        if dataset.vectors is None:
            raise ValueError("Dataset must have .vectors")

        X = torch.tensor(dataset.vectors, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.model(X)
            probs = torch.sigmoid(logits).cpu().numpy()

        preds_bin = (probs >= self.threshold).astype(int)
        labels = self.mlb.inverse_transform(preds_bin)
        dataset.results = labels
        return dataset
