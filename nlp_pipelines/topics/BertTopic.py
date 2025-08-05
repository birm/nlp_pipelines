from nlp_pipelines._base.BaseMethod import BaseMethod
from bertopic import BERTopic

class BertTopic(BaseMethod):
    """
    Topic modeling using BERTopic
    """

    def __init__(self, nr_topics=None, language='english'):
        super().__init__(method_type="unsupervised", supervised=False)
        self.method_name = "BERTopic"
        self.requires_text = True
        self.topic_model = BERTopic(language=language, nr_topics=nr_topics)
        self.is_fit = False

    def fit(self, dataset, possible_labels=[]):
        if not hasattr(dataset, 'texts') or not dataset.texts:
            raise ValueError("Dataset must have a 'texts' attribute with input documents.")

        # Fit the BERTopic model
        self.topics, self.probs = self.topic_model.fit_transform(dataset.texts)
        self.is_fit = True

    def predict(self, dataset):
        if not self.is_fit:
            raise RuntimeError("BertTopic must be fit before prediction.")

        if not hasattr(dataset, 'texts') or not dataset.texts:
            raise ValueError("Dataset must have a 'texts' attribute with input documents.")
        
        # Predict topics for new documents
        predicted_topics, probs = self.topic_model.transform(dataset.texts)
        dataset.results = predicted_topics
        dataset.topic_probabilities = probs
        return dataset

    def get_topic_info(self):
        if not self.is_fit:
            raise RuntimeError("Model must be fit before accessing topic information.")
        return self.topic_model.get_topic_info()

    def get_topic(self, topic_id):
        if not self.is_fit:
            raise RuntimeError("Model must be fit before accessing topics.")
        return self.topic_model.get_topic(topic_id)
