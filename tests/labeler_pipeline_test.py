# tests/labeler_pipeline_test.py
import context
from nlp_pipelines.pipeline.Pipeline import Pipeline
from nlp_pipelines.dataset import Dataset

def test_labeler_pipeline_smoke():
    texts = ["Liver disease is common", "Kidney failure treatment", "Cancer diagnosis improved"]
    dataset = Dataset(texts)

    pipeline = Pipeline([
        {"name": "vec", "method": "vectorizer.Tfidf"},
        {"name": "label", "method": "labeler.Bm25", "params": {"top_k": 3}}
    ])
    pipeline.set_data(train_data=dataset, run_data=dataset)
    pipeline.run()

    print("Keywords:", pipeline.run_data.results)
    assert pipeline.run_data.results is not None

if __name__ == "__main__":
    test_labeler_pipeline_smoke()