# NLP Pipelines

**NLP Pipelines** is a modular Python library for building flexible natural language processing pipelines. It supports dataset management, preprocessing, vectorization, labeling/keyword extraction, topic modeling, classification, and evaluation.

---

## Example Usage
```python
from nlp_pipelines.pipeline.Pipeline import Pipeline
from nlp_pipelines.dataset import Dataset

## pick dataset
texts = ["I love this movie", "This is terrible", "Fantastic work", "Awful experience", "It was okay"]
truths = ["positive", "negative", "positive", "negative", "neutral"]
dataset = Dataset(texts, truths)
train, test = dataset.split(count=3, seed=42)

# define the pipeline; methods use the import name (e.g. `from nlp_pipelines.vectorizer import Tfidf` becomes `"vectorizer.Tfidf"`)
pipeline = Pipeline([
    {"name": "vec", "method": "vectorizer.Tfidf"},
    {"name": "clf", "method": "classifier.Xgboost"}
])
# set train and test data
pipeline.set_data(train_data=train, run_data=test)
# run the pipeline
pipeline.run()

# look at the results
print("Results:", pipeline.run_data.results)
```

If a method needs arguments, pass them as "params" like:
```python
    pipeline = Pipeline([
        {"name": "vec", "method": "vectorizer.Tfidf"},
        {"name": "label", "method": "labeler.Bm25", "params": {"top_k": 3}}
    ])
```


## Object Types

### Pipeline

Located in \[`pipeline/`\](nlp_pipelines/pipeline)

A `Pipeline` object defines an ordered sequence of processing steps that can include any number of `Method` objects (e.g., preprocessors, vectorizers, labelers, classifiers). Pipelines allow methods to be chained together with automatic handling of dataset transformations between steps.

Each pipeline is **directional**: the order of steps matters, and each step transforms the dataset before it is passed to the next. Methods must be `fit()` before `predict()` can be called.

**Key Concepts**:
- Each step is specified as a `(step_name, method_instance)` tuple.
- The pipeline handles:
  - Calling `.fit(dataset)` on each method (if needed).
  - Calling `.transform(dataset)` for methods with that capability (e.g., vectorizers).
  - Passing the output of one method to the next.
- Should be able to be trained then saved to run on data later. 

**Training & Prediction**:
- `train(dataset)` will call `fit()` on each method if applicable, and also apply `transform()` for vectorizers.
- `predict(dataset)` must be called only on a pipeline that has already been trained. It applies each method in sequence, forwarding the updated dataset.

**Notes**:
- Vectorizers should appear before any step that depends on vectors (e.g., classifiers, some labelers).
- Methods that require candidate labels should be trained with those provided at `fit` time.


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
- [`preprocess.StopwordRemove`](nlp_pipelines/preprocess/StopwordRemove.py): stopword removal
- [`preprocess.Stem`](nlp_pipelines/preprocess/Stem.py): stemming
- [`preprocess.Lemmatize`](nlp_pipelines/preprocess/Lemmatize.py): lemmatization
- [`preprocess.StopwordRemove`](nlp_pipelines/preprocess/PosRemoval.py): part-of-speech filtering
- [`preprocess.TokenFilter`](nlp_pipelines/preprocess/TokenFilter.py): Filter things on spacy's tokens (e.g. isNumeric etc)


### Vectorizer

Located in [`vectorizer/`](nlp_pipelines/vectorizer)

Vectorization methods transform datasets into `VectorizedDataset` objects by adding `vectors`. Vectorzers have a `.fit(dataset)` method and a  `.transform(dataset)` method. You must fit before transforming.

Available modules:
- [`vectorizer.Tfidf`](nlp_pipelines/vectorizer/Tfidf.py): TF-IDF vectorization
- [`vectorizer.BagOfWords`](nlp_pipelines/vectorizer/BagOfWords.py): bag-of-words vectorization
- [`vectorizer.SentenceEmbedding`](nlp_pipelines/vectorizer/SentenceEmbedding.py): pretrained sentence-level embeddings 
- [`vectorizer.FastText`](nlp_pipelines/vectorizer/FastText.py): FastText vectorizer; can use pretrained, supervised, or unsupervised.


### Labeler

Located in [`labeler/`](nlp_pipelines/labeler)

Labeler methods assign keyword labels to input documents, specifically one document may have 0+ labels. Some of these methods take a list of 'candidate labels' which are the set of labels which are possible to use, but some do not and simply extract keywords from documents. All methods must be trained (even unsupervised ones) using the `.train(dataset)` method. Predictions are made using `.predict(dataset)`, and results can be converted into new datasets for further processing.

Modules:
- [`labeler.MultiLogistic`](nlp_pipelines/labeler/predictive/MultiLogistic.py): supervised multi-label classification for known keyword sets
- [`labeler.ThresholdSim`](nlp_pipelines/labeler/predictive/ThresholdSim.py): similarity-based label prediction using thresholding on various distance metrics
- [`labeler.Bm25`](nlp_pipelines/labeler/extractive/Bm25.py): BM25-based keyword extraction
- [`labeler.Yake`](nlp_pipelines/labeler/extractive/Yake.py): YAKE-based keyword extraction
- [`labeler.MultiRake`](nlp_pipelines/labeler/extractive/MultiRake.py): MultiRake-based keyword extraction
- [`labeler.KeyBert`](nlp_pipelines/labeler/extractive/KeyBert.py): KeyBERT-based keyword extraction
- [`labeler.TfidfTopN`](nlp_pipelines/labeler/extractive/TfidfTopN.py): TF-IDF Top-N keyword extraction
- [`labeler.SimpleNNLabeler`](nlp_pipelines/labeler/extractive/SimpleNNLabeler.py): Using a neural network to get a n dim vector, and picking those over a threhsold. Train with lots of data! (1. linear input dim [auto detected from the input vector] to hidden dim [set, default 128] with relu activation, 2. dropout (0.2), 3. linear from hidden dim to output dim (number of possible labels), 4. argmax)

---

### Classifier

Located in [`classifier/`](nlp_pipelines/classifier)

Classification methods assign labels to documents using a set of input possible labels, specifically each document has exactly one class. The distinction with "Clusterer" is that classifiers group in a way which is aligned with meaning, while clusterers find groups inherent in the data directly. All methods must be trained (even unsupervised ones) using the `.train(dataset)` method. Predictions are made using `.predict(dataset)`, and results can be converted into new datasets for further processing.

Modules:
- [`classifier.BartTag`](nlp_pipelines/classifier/BartTag.py): BART zero shot classification (fit and predict only require texts)
- [`classifier.LabelProp`](nlp_pipelines/classifier/LabelProp.py): supervised label propagation model (fit requires: vectors, true labels; predict requires vectors)
- [`classifier.Xgboost`](nlp_pipelines/classifier/Xgboost.py): supervised xgboost model (fit requires: vectors, labels; predict requires vectors)
- [`classifier.SimpleNNClassifier`](nlp_pipelines/classifier/SimpleNNClassifier.py): Using a neural network to get a n dim vector, and picking the top one. Train with lots of data! (1. linear input dim [auto detected from the input vector] to hidden dim [set, default 128] with relu activation, 2. dropout (0.2), 3. linear from hidden dim to output dim (number of possible labels), 4. sigmoid + thredhold)

---

#### Clusterer

Located in [`clusterer/`](nlp_pipelines/clusterer)

Classification methods assign labels to documents using some similarity in the data (usually a distance metric), specifically each document has exactly one class. The distinction with "Clusterer" is that classifiers group in a way which is aligned with meaning, while clusterers find groups inherent in the data directly. All methods must be trained (even unsupervised ones) using the `.train(dataset)` method. Predictions are made using `.predict(dataset)`, and results can be converted into new datasets for further processing.

Modules:
- [`clusterer.GraphAffinity`](nlp_pipelines/clusterer/GraphAffinity.py): graph-based clustering using spectral clustering (fit and predict require vectors)
- [`clusterer.UmapHdbscan`](nlp_pipelines/clusterer/UmapHdbscan.py): use of UMAP for dimensionalty reduction and hdbscan for clustering (fit and predict require vectors)
- [`clusterer.Svm`](nlp_pipelines/clusterer/Svm.py): use of SVM to split the dataset
- [`clusterer.Kmeans`](nlp_pipelines/clusterer/Kmeans.py): use of k-means for clustering (fit and predict require vectors)

---

### Topic Model

Located in [`topics/`](nlp_pipelines/topics)

Topic modeling methods learn topic distributions from datasets and assign topic labels. Results can be converted into new datasets for further processing.

Modules:
- [`topics.BertTopic`](nlp_pipelines/topics/BertTopic.py): BERTopic-based topic modeling

---

### Evaluate

Located in [`evaluate/evaluate`](nlp_pipelines/evaluate/evaluate.py)

Evaluation methods compute metrics such as precision, recall, and classification accuracy using a `LabeledDataset` with results.

---
