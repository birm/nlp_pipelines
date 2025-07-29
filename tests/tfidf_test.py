import context
from nlp_pipelines.vectorizer.Tfidf import Tfidf
from nlp_pipelines.dataset import Dataset

def test_tfidf_vectorizer_smoke():
    texts = [
        "I love this movie",
        "This is terrible",
        "Fantastic work",
        "Awful experience",
        "It was okay"
    ]
    dataset = Dataset(texts)

    vectorizer = Tfidf()
    vectorizer.fit(dataset)
    dataset = vectorizer.transform(dataset)

    print("Vectors shape:", None if dataset.vectors is None else (len(dataset.vectors), len(dataset.vectors[0])))
    print("Sample vector (first doc):", dataset.vectors[0])

    assert dataset.vectors is not None
    assert len(dataset.vectors) == len(dataset.texts)
    assert len(dataset.vectors[0]) > 0

if __name__ == "__main__":
    test_tfidf_vectorizer_smoke()