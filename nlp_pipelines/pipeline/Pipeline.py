import logging
import pickle

import importlib
import logging
import inspect

class Pipeline:
    def __init__(self, steps):
        """
        Initializes the pipeline and dynamically loads classes from string paths.

        Args:
            steps (list): Each step is a dict:
                - "name": name of the step
                - "method": class or string like 'vectorizer.Tfidf'
                - "params": optional kwargs for class init
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.steps = []
        self.possible_labels = None # placeholder
        self.possible_labels_embed = None # placeholder

        for step in steps:
            step_name = step.get("name", "unnamed_step")
            method = step["method"]
            params = step.get("params", {})

            if isinstance(method, str):
                method_instance = self._load_method_from_string(method, params)
                self.logger.debug(f"Instantiated {method} for step '{step_name}'")
            elif inspect.isclass(method):
                method_instance = method(**params)
            else:
                method_instance = method  # already instantiated

            self.steps.append({
                "name": step_name,
                "method": method_instance
            })

        self.train_data = None
        self.run_data = None

    def _load_method_from_string(self, path, params):
        """
        Load and instantiate a method class from a string like 'vectorizer.Tfidf'

        Returns:
            An instance of the class with provided params.
        """
        base_pkg = "nlp_pipelines"
        parts = path.split(".")
        if len(parts) < 2:
            raise ValueError(f"Method string '{path}' should be like 'vectorizer.Tfidf'")

        module_path = ".".join([base_pkg] + parts[:-1])
        class_name = parts[-1]

        try:
            mod = importlib.import_module(module_path)
            cls = getattr(mod, class_name)
            return cls(**params)
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Could not load method '{path}': {e}") from e


    def set_data(self, train_data=None, run_data=None, possible_labels=None):
        """
        Set the datasets to be used by the pipeline.
        """
        self.train_data = train_data
        self.run_data = run_data
        self.possible_labels = possible_labels

    def train(self):
        if self.train_data is None:
            raise ValueError("Training data not set. Use `set_data(train_data=...)` first.")

        dataset = self.train_data

        for step in self.steps:
            method = step["method"]

            if getattr(method, "is_fit", False):
                continue  # skip if already fit

            # Check for truths if required
            if method.train_requires_truths and not dataset.truths:
                raise ValueError(f"Step '{step['name']}' requires truths in training data.")

            # Check for vectors if required
            if method.requires_vectors and dataset.vectors is None:
                raise ValueError(f"Step '{step['name']}' requires vectors, but none found. Vectorize first.")

            # check for embedded possible labels if required
            if method.requires_embed_possible_labels and self.possible_labels_embed is None:
                raise ValueError(f"Step '{step['name']}' requires embedded possible labels, but none found. Set possible_labels and vectorize first.")

            if method.requires_embed_possible_labels:
                method.fit(dataset, possible_labels=self.possible_labels, possible_labels_embed=self.possible_labels_embed)
            else:
                method.fit(dataset)

            # If transform exists, run it immediately to update dataset
            if hasattr(method, "transform"):
                dataset = method.transform(dataset)
                if method.method_type == "vectorizer" and self.possible_labels is not None:
                    self.possible_labels_embed = method.transform_labels(self.possible_labels)
                

        # Save updated dataset back to train_data
        self.train_data = dataset


    def predict(self):
        if self.run_data is None:
            raise ValueError("Run data not set. Use `set_data(run_data=...)` first.")

        dataset = self.run_data
        for step in self.steps:
            method = step["method"]

            if not getattr(method, "is_fit", False):
                raise RuntimeError(f"Step '{step['name']}' not trained. Call `train()` or `run()` first.")

            if hasattr(method, "predict"):
                dataset = method.predict(dataset)
            elif hasattr(method, "transform"):
                dataset = method.transform(dataset)
            else:
                raise TypeError(f"Step '{step['name']}' method must implement `predict` or `transform`.")

        return dataset  # final result

    def run(self):
        """Train and then predict (if not already trained)."""
        if not all(getattr(step["method"], "is_fit", False) for step in self.steps):
            self.train()
        return self.predict()

    def save(self, filepath):
        """Save the full pipeline object (including trained methods) to disk."""
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath):
        """Load a full pipeline object from disk."""
        with open(filepath, "rb") as f:
            return pickle.load(f)
