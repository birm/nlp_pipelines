import spacy
from nlp_pipelines.dataset.dataset import Dataset
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

def stem(dataset: Dataset) -> Dataset:
    """
    Applies stemming to the texts in the given Dataset object.
    
    Args:
        dataset (Dataset): The Dataset object whose texts will be processed.
    
    Returns:
        dataset: The dataset; also modifies the Dataset in place.
    """
    stemmed_texts = []
    
    for text in dataset.texts:
        # Tokenize and stem each word in the text
        stemmed_text = " ".join([stemmer.stem(word) for word in text.split()])
        stemmed_texts.append(stemmed_text)
    
    dataset.texts = stemmed_texts
    return dataset