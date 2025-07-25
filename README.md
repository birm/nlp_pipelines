# NLP Pipelines

**NLP Pipelines** is a modular Python library for building flexible natural language processing pipelines. It supports dataset management, preprocessing, vectorization, labeling/keyword extraction, topic modeling, classification, and evaluation.

---


## Object Types

### Dataset

Located in [`dataset/`](nlp_pipelines/dataset/)

A `Dataset` object is the core data container, and includes:
- `texts`: a list of documents (required)
- `original_text`: a copy of the original text, set only by the first preprocessing step

A dataset (including the below) has methods which allow for controlled access of the dataset to enable some filtering, or train/test split.
- `constructor(texts, labels)` - Makes a dataset from a list-like for texts, and optinally a set of labels in the same order.
- `.get(labeled=True, count=100, ratio=0.5, seed=None)` - Makes a new dataset from the original. set labeled=True to only consider labeled documents. If count is set, returns at most count documents. If ratio is set, then returns at most that ratio of documents (if labeled set, then the set is of the labeled documents). Be careful when using both ratio and count. Set seed to seed the random selection process and get consistent splits.
- `.split(labeled=True, splitLabeled=True, count=100, ratio=0.5, seed=None)` - Performs like get, but returns both the selected part and the other. The splitLabeled argument determines 

To use for cross validation, follow this pseudocode:
```python
...
for i in range(5):
    val_set = dataset.get(labeled=True, ratio=0.2, seed=None)
    y_hat = classifier.preduct(val_set)
    evaluator.compare(val_set, y_hat)
    ...
```

### LabeledDataset
A `LabeledDataset` conceptually extends a `Dataset` and adds:
- `truth`: true labels such as keywords or class labels (named 'truth' to not confuse with labeler predictions)

The definition of a `LabeledDataset` is a dataset with nonnull `truth`; it is not a separate class in code.

### VectorizedDataset

A `VectorizedDataset` conceptually extends a `Dataset` and adds:
- `vectors`: a matrix of feature representations

The definition of a `VectorizedDataset` is a dataset with nonnull `vectors`; it is not a separate class in code.

### ResultDataset

A `ResultDataset` conceptually extends a `Dataset` and adds:
- `results`: results from the method

The definition of a `ResultDataset` is a dataset with nonnull `results`; it is not a separate class in code.
`ResultDataset` can be used in a complex pipeline as a `Dataset` by (todo; add/name a method for this: for now it's just by moving the results to the texts field.)

### Method

Each method is its own class, but each contains `fit` and `predict` methods, but `predict` should not be run unless the model has been `fit` first. Supervised methods' `fit` generally require input of a `LabeledDataset`, while unsupervised methods may require a list of candidate result-type objects (e.g. labels, classifications), or no input at all.

### TrainedMethod

A `TrainedMethod` conceptually extends a `Method` and logically allows access to `predict`. We do not recommend running `fit` on a `TrainedMethod` (at this point; may reconsider ways to train cleanly multiple times).

---

## Method Types

### Preproces

Located in [`preprocess/`](nlp_pipelines/preprocess)

Preprocessing methods operate on `Dataset` objects and return new `Dataset` objects with transformed `.text` values. Only the first preprocessing step populates the `.original_text` field.

Available modules:
- [`rm_stopwords.py`](nlp_pipelines/preprocess/rm_stopwords.py): stopword removal
- [`stem.py`](nlp_pipelines/preprocess/stem.py): stemming
- [`lemmatize.py`](nlp_pipelines/preprocess/lemmatize.py): lemmatization
- [`pos_removal.py`](nlp_pipelines/preprocess/pos_removal.py): part-of-speech filtering


### Vectorizer

Located in [`vectorizer/`](nlp_pipelines/vectorizer)

Vectorization methods transform datasets into `VectorizedDataset` objects by adding `vectors`. Vectorzers have a `.fit(dataset)` method and a  `.transform(dataset)` method. You must fit before transforming.

Available modules:
- [`Tfidf.py`](nlp_pipelines/vectorizer/Tfidf.py): TF-IDF vectorization
- [`Bow.py`](nlp_pipelines/vectorizer/Bow.py): bag-of-words vectorization
- [`SentenceEmbedding.py`](nlp_pipelines/vectorizer/SentenceEmbedding.py): pretrained sentence-level embeddings 
- [`FastText.py`](nlp_pipelines/vectorizer/FastText.py): FastText vectorizer; can use pretrained, supervised, or unsupervised.


### Labeler

Located in [`labeler/`](nlp_pipelines/labeler)

Labeler methods assign keyword labels to input documents, specifically one document may have 0+ labels. Some of these methods take a list of 'candidate labels' which are the set of labels which are possible to use, but some do not and simply extract keywords from documents. All methods must be trained (even unsupervised ones) using the `.train(dataset)` method. Predictions are made using `.predict(dataset)`, and results can be converted into new datasets for further processing.

Modules:
- [`predictive/MultiLogistic.py`](nlp_pipelines/labeler/predictive/MultiLogistic.py): supervised multi-label classification for known keyword sets
- [`predictive/ThresholdSim.py`](nlp_pipelines/labeler/predictive/ThresholdSim.py): similarity-based label prediction using thresholding on various distance metrics
- [`extractive/Bm25.py`](nlp_pipelines/labeler/extractive/Bm25.py): BM25-based keyword extraction
- [`extractive/Yake.py`](nlp_pipelines/labeler/extractive/Yake.py): YAKE-based keyword extraction
- [`extractive/MultiRake.py`](nlp_pipelines/labeler/extractive/MultiRake.py): MultiRake-based keyword extraction
- [`extractive/KeyBert.py`](nlp_pipelines/labeler/extractive/KeyBert.py): KeyBERT-based keyword extraction
- [`extractive/TfidfTopN.py`](nlp_pipelines/labeler/extractive/TfidfTopN.py): TF-IDF Top-N keyword extraction


### Classifier

Located in [`classifier/`](nlp_pipelines/classifier)

Classification methods assign labels to documents using a set of input possible labels, specifically each document has exactly one class. The distinction with "Clusterer" is that classifiers group in a way which is aligned with meaning, while clusterers find groups inherent in the data directly. All methods must be trained (even unsupervised ones) using the `.train(dataset)` method. Predictions are made using `.predict(dataset)`, and results can be converted into new datasets for further processing.

Modules:
- [`BartTag.py`](nlp_pipelines/classifier/BartTag.py): BART zero shot classification (fit and predict only require texts)
- [`LabelProp.py`](nlp_pipelines/classifier/LabelProp.py): supervised label propagation model (fit requires: vectors, true labels; predict requires vectors)
- [`Xgboost.py`](nlp_pipelines/classifier/Xgboost.py): supervised xgboost model (fit requires: vectors, labels; predict requires vectors)

---
#### Clusterer

Located in [`clusterer/`](nlp_pipelines/clusterer)

Classification methods assign labels to documents using some similarity in the data (usually a distance metric), specifically each document has exactly one class. The distinction with "Clusterer" is that classifiers group in a way which is aligned with meaning, while clusterers find groups inherent in the data directly. All methods must be trained (even unsupervised ones) using the `.train(dataset)` method. Predictions are made using `.predict(dataset)`, and results can be converted into new datasets for further processing.

Modules:
- [`GraphAffinity.py`](nlp_pipelines/clusterer/GraphAffinity.py): graph-based clustering using spectral clustering (fit and predict require vectors)
- [`UmapHdbscan.py`](nlp_pipelines/clusterer/UmapHdbscan.py): use of UMAP for dimensionalty reduction and hdbscan for clustering (fit and predict require vectors)

---

### Topic Model

Located in [`topics/`](nlp_pipelines/topics)

Topic modeling methods learn topic distributions from datasets and assign topic labels. Results can be converted into new datasets for further processing.

Modules:
- [`bertopic.py`](nlp_pipelines/topics/bertopic.py): BERTopic-based topic modeling

---

### Evaluate

Located in [`evaluate/evaluate.py`](nlp_pipelines/evaluate/evaluate.py)

Evaluation methods compute metrics such as precision, recall, and classification accuracy using a `LabeledDataset` with results.

---
