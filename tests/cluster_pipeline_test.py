# tests/cluster_pipeline_test.py
import context
from nlp_pipelines.pipeline.Pipeline import Pipeline
from nlp_pipelines.dataset import Dataset

def test_clusterer_pipeline_smoke():
    texts = [
        "Apple releases new iPhone",
        "Samsung launches Galaxy model",
        "Google introduces Android update",
        "Microsoft unveils Surface",
        "Nvidia announces new GPU",
        "Facebook updates privacy policy",
        "Tesla announces new model",
        "Amazon opens new warehouse"
    ]
    dataset = Dataset(texts)

    pipeline = Pipeline([
        {"name": "vec", "method": "vectorizer.Tfidf"},
        {"name": "cluster", "method": "clusterer.UmapHdbscan", "params": {"n_neighbors":2, "min_cluster_size":2}}
    ])
    pipeline.set_data(train_data=dataset, run_data=dataset)
    pipeline.run()

    print("Cluster assignments:", pipeline.run_data.results)
    assert pipeline.run_data.results is not None

if __name__ == "__main__":
    test_clusterer_pipeline_smoke()