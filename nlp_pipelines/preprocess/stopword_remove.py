import spacy
from nlp_pipelines.dataset.dataset import Dataset

nlp = spacy.load("en_core_web_sm")

# **1. Stopword Removal Function**
def stopword_remove(dataset: Dataset) -> Dataset:
    """
    Removes stopwords from the texts in the given Dataset object.
    
    Args:
        dataset (Dataset): The Dataset object whose texts will be processed.
    
    Returns:
        dataset: The dataset; also modifies the Dataset in place.
    """
    filtered_texts = []
    
    for text in dataset.texts:
        # Process the text with spaCy
        doc = nlp(text)
        # Filter out stopwords
        filtered_text = " ".join([token.text for token in doc if not token.is_stop])
        filtered_texts.append(filtered_text)
    
    dataset.texts = filtered_texts
    return dataset