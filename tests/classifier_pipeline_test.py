# tests/classifier_pipeline_test.py
import context
from nlp_pipelines.pipeline.Pipeline import Pipeline
from nlp_pipelines.dataset import Dataset

def test_classifier_pipeline_smoke():
    texts = ["I love this movie", "This is terrible", "Fantastic work", "Awful experience", "It was okay"]
    truths = ["positive", "negative", "positive", "negative", "neutral"]
    dataset = Dataset(texts, truths)
    train, test = dataset.split(count=3, seed=42)

    pipeline = Pipeline([
        {"name": "vec", "method": "vectorizer.Tfidf"},
        {"name": "clf", "method": "classifier.Xgboost"}
    ])
    pipeline.set_data(train_data=train, run_data=test)
    pipeline.run()

    print("Results:", pipeline.run_data.results)
    assert pipeline.run_data.results is not None
    assert len(pipeline.run_data.results) == len(test.texts)

if __name__ == "__main__":
    test_classifier_pipeline_smoke()