import spacy
from nlp_pipelines.dataset.dataset import Dataset

# Load spaCy's English language model (you might need to install it first with: python -m spacy download en_core_web_sm)
nlp = spacy.load("en_core_web_sm")

def lemmatize(dataset: Dataset) -> Dataset:
    """
    Takes a Dataset object and modifies its .texts attribute by lemmatizing the text.

    Args:
        dataset (Dataset): The Dataset object whose texts will be lemmatized.

    Returns:
        dataset: The dataset; also modifies the Dataset in place.
    """
    lemmatized_texts = []

    for text in dataset.texts:
        # Process the text with spaCy
        doc = nlp(text)
        # Lemmatize and join tokens back into a sentence
        lemmatized_text = " ".join([token.lemma_ for token in doc])
        lemmatized_texts.append(lemmatized_text)
    
    dataset.texts = lemmatized_texts
    return dataset
