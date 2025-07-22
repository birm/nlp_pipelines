import spacy
from typing import List
from nlp_pipelines.dataset.dataset import Dataset

nlp = spacy.load("en_core_web_sm")

def pos_removal(dataset: Dataset, pos_tags_to_remove: List[str]) -> Dataset:
    """
    Removes specified parts of speech from the texts in the given Dataset object.
    
    Args:
        dataset (Dataset): The Dataset object whose texts will be processed.
        pos_tags_to_remove (List[str]): List of POS tags to remove (e.g., ['VERB', 'ADJ']).
    
    Returns:
        dataset: The dataset; also modifies the Dataset in place.
    """
    cleaned_texts = []
    
    for text in dataset.texts:
        # Process the text with spaCy
        doc = nlp(text)
        # Filter out tokens with the specified POS tags
        cleaned_text = " ".join([token.text for token in doc if token.pos_ not in pos_tags_to_remove])
        cleaned_texts.append(cleaned_text)
    
    dataset.texts = cleaned_texts
    return dataset
