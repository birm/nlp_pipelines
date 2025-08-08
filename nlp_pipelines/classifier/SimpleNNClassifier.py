from nlp_pipelines._base.BaseMethod import BaseMethod
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class SimpleNNClassifier(BaseMethod):

    def __init__(self, hidden_dim=128, epochs=10, lr=1e-3, batch_size=32):
        super().__init__(method_type="labeler", supervised=True)
        self.input_dim = None
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None  # Built in fit
        self.label_to_id = {}
        self.id_to_label = {}
        self.is_fit = False

    def fit(self, dataset, possible_labels=None):
        if dataset.vectors is None or dataset.truths is None:
            raise ValueError("Dataset must have .vectors and .labels")

        # get input dim from vectorization
        self.input_dim = np.shape(dataset.vectors)[1]

        # Map string labels to integer IDs
        unique_labels = sorted(set(dataset.truths))
        self.label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
        self.id_to_label = {idx: label for label, idx in self.label_to_id.items()}

        y_ids = [self.label_to_id[label] for label in dataset.truths]

        X = torch.tensor(dataset.vectors, dtype=torch.float32).to(self.device)
        y = torch.tensor(y_ids, dtype=torch.long).to(self.device)
        num_classes = len(self.label_to_id)

        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim, num_classes)
        ).to(self.device)

        self.criterion = nn.CrossEntropyLoss()
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
            preds = torch.argmax(logits, dim=1).cpu().numpy()

        labels = [self.id_to_label[p] for p in preds]
        dataset.results = labels
        return dataset
