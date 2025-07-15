# NLP Pipelines

**NLP Pipelines** is a modular Python library for building flexible natural language processing pipelines. It supports dataset management, preprocessing, vectorization, keyword extraction, topic modeling, classification, and evaluation.

---


## Object Types

### Dataset

Located in [`dataset/`](nlp_pipelines/dataset/)

A `Dataset` object is the core data container, and includes:
- `text`: a list of documents (required)
- `original_text`: a copy of the original text, set only by the first preprocessing step

### LabeledDataset
A `LabeledDataset` conceptually extends a `Dataset` and adds:
- `truth`: labels such as keywords or class labels

The definition of a `LabeledDataset` is a dataset with nonnull `truth`; it is not a separate class in code.

### VectorizedDataset

A `VectorizedDataset` conceptually extends a `Dataset` and adds:
- `vectors`: a matrix of feature representations

The definition of a `VectorizedDataset` is a dataset with nonnull `vectors`; it is not a separate class in code.

### ResultDataset

A `ResultDataset` conceptually extends a `Dataset` and adds:
- `results`: results from the method

The definition of a `ResultDataset` is a dataset with nonnull `results`; it is not a separate class in code.
`ResultDataset` can be used in a complex pipeline as a `Dataset` by (todo; add/name a method for this: for now it's just by moving the results to the text field.)

### Method

Each method is its own class, but each contains `fit` and `predict` methods, but `predict` should not be run unless the model has been `fit` first. Supervised methods' `fit` generally require input of a `LabeledDataset`, while unsupervised methods may require a list of candidate result-type objects (e.g. labels, classifications), or no input at all.

### TrainedMethod

A `TrainedMethod` conceptually extends a `Method` and logically allows access to `predict`. We do not recommend running `fit` on a `TrainedMethod` (at this point; may reconsider ways to train cleanly multiple times).

---

## Method Types

### Preprocessor

Located in [`preprocessor/`](nlp_pipelines/preprocessor)

Preprocessing methods operate on `Dataset` objects and return new `Dataset` objects with transformed `.text` values. Only the first preprocessing step populates the `.original_text` field.

Available modules:
- [`rm_stopwords.py`](nlp_pipelines/preprocessor/rm_stopwords.py): stopword removal
- [`stem.py`](nlp_pipelines/preprocessor/stem.py): stemming
- [`lemmatize.py`](nlp_pipelines/preprocessor/lemmatize.py): lemmatization
- [`pos_removal.py`](nlp_pipelines/preprocessor/pos_removal.py): part-of-speech filtering


### Vectorizer

Located in [`vectorizer/`](nlp_pipelines/vectorizer)

Vectorization methods transform datasets into `VectorizedDataset` objects by adding `vectors`

Available modules:
- [`tfidf.py`](nlp_pipelines/vectorizer/tfidf.py): TF-IDF vectorization
- [`bow.py`](nlp_pipelines/vectorizer/bow.py): bag-of-words vectorization
- [`sentence_embedding.py`](nlp_pipelines/vectorizer/sentence_embedding.py): sentence-level embeddings


### Keyword Extractior

Located in [`keywords/`](nlp_pipelines/keywords)

Keyword extraction methods assign keyword labels to input documents. All methods must be trained (even unsupervised ones) using the `.train(dataset)` method. Predictions are made using `.predict(dataset)`, and results can be converted into new datasets for further processing.

Modules:
- [`unsupervised/yake.py`](nlp_pipelines/keywords/unsupervised/yake.py): YAKE-based unsupervised extraction
- [`supervised/logistic.py`](nlp_pipelines/keywords/supervised/logistic.py): supervised multi-label classification for known keyword sets


### Classifier

Located in [`classifier/`](nlp_pipelines/classifier)

Classification methods assign labels to documents using supervised or unsupervised strategies. All methods must be trained (even unsupervised ones) using the `.train(dataset)` method. Predictions are made using `.predict(dataset)`, and results can be converted into new datasets for further processing.

Modules:
- [`unsupervised/graph_affinity.py`](nlp_pipelines/classifier/unsupervised/graph_affinity.py): graph-based label propagation
- [`supervised/label_prop.py`](nlp_pipelines/classifier/supervised/label_prop.py): supervised label propagation model

---

### Topic Model

Located in [`topics/`](nlp_pipelines/topics)

Topic modeling methods learn topic distributions from datasets and assign topic labels. Results can be converted into new datasets for further processing.

Modules:
- [`bertopic.py`](nlp_pipelines/topics/bertopic.py): BERTopic-based topic modeling

---

### Evaluator

Located in [`evaluator/evaluator.py`](nlp_pipelines/evaluator/evaluator.py)

Evaluation methods compute metrics such as precision, recall, and classification accuracy using a `LabeledDataset`.

---
