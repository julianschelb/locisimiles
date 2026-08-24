# Training Module

Trainers for every trainable approach in the benchmark: a lexical (LogReg/GBDT)
classifier, a Word2Vec retrieval model, a fine-tuned transformer sequence
classifier, and a fine-tuned SentenceTransformer embedding retriever. Rule-based
and TF-IDF/BM25 retrieval have no learned parameters and therefore no trainer —
TF-IDF/BM25 build their index on the fly at query time instead.

## Input contract

Three of the four trainers (`LexicalClassifierTrainer`, `ClassificationTrainer`,
`EmbeddingTrainer`) share one input type: [`TrainingData`](#trainingdata),
which bundles a query [`Document`](document.md), a source `Document`, and a
[`GroundTruth`](document.md#groundtruth) of labeled pairs. `Word2VecTrainer` is
unsupervised (it learns word embeddings from raw sentences, not labeled pairs),
so it takes plain `Document`s instead.

```python
from locisimiles.document import Document
from locisimiles.ground_truth import GroundTruth
from locisimiles.training.data import TrainingData

query_doc = Document("query.csv")
source_doc = Document("source.csv")
positives = GroundTruth("known_positives.csv")  # query_id, source_id, label

data = TrainingData(query_doc, source_doc, positives)
```

## TrainingData

Iterating a `TrainingData` yields resolved `(query_text, source_text, label)`
triples — text is looked up by segment id once here rather than duplicated
across each trainer. `TrainingData` also carries all four of the paper's
negative-sampling methods as chainable methods, since assembling a training
set and sampling its negatives are naturally the same step.

`sample_random_negatives` (⟨qry,rnd⟩) is the **default** — a random
non-positive source segment per query, with no model loading required. The
paper's own ablation found `sample_mixed_negatives` (⟨qry,mix⟩) the
best-performing method on every reported metric (F1, accuracy, FPR, SMR); it's
offered as an explicit, one-call upgrade from the default rather than the
default itself, since it requires loading a pretrained embedding model to mine
hard negatives:

```python
# Default: no model download needed
data = TrainingData(query_doc, source_doc, positives).sample_random_negatives(
    n_per_query=5,
)

# Best-performing per the paper's ablation — random + embedding-mined hard negatives
data = TrainingData(query_doc, source_doc, positives).sample_mixed_negatives(
    n_random_per_query=5,
    n_hard_per_query=5,
)
```

`sample_random_pairs` (⟨rnd,rnd⟩, fully-random pairs — the paper's weakest
method) and `sample_hard_negatives` (⟨qry,sim⟩ alone) are also available for
completeness/parity with the paper. All four sampling methods return a **new**
`TrainingData` and never mutate the original, so they chain freely.

::: locisimiles.training.data.TrainingData
    options:
      heading_level: 3

## LexicalClassifierTrainer

Trains the TF-IDF/Jaccard/overlap-feature LogReg or GBDT classifier consumed
by [`LexicalClassifierJudge`](judges.md).

```python
from locisimiles.training.lexical import LexicalClassifierTrainer, LexicalClassifierTrainerConfig

config = LexicalClassifierTrainerConfig(output_dir="models/lexical", classifier="logreg")
trainer = LexicalClassifierTrainer(config)
trainer.fit(data=data)
artifact_path = trainer.save()
```

::: locisimiles.training.lexical.trainer.LexicalClassifierTrainerConfig
    options:
      heading_level: 3
::: locisimiles.training.lexical.trainer.LexicalClassifierTrainer
    options:
      heading_level: 3

## Word2VecTrainer

Trains the gensim Word2Vec model consumed by `Word2VecCandidateGenerator`
(the Burns-style retrieval baseline). Unsupervised: takes `Document`s
directly rather than a `TrainingData`, since there's no label to learn from.

```python
from locisimiles.training.word2vec import Word2VecTrainer, Word2VecTrainerConfig

config = Word2VecTrainerConfig(output_dir="models/word2vec")
trainer = Word2VecTrainer(config)
trainer.fit(documents=[query_doc, source_doc])
model_path = trainer.save()
```

::: locisimiles.training.word2vec.trainer.Word2VecTrainerConfig
    options:
      heading_level: 3
::: locisimiles.training.word2vec.trainer.Word2VecTrainer
    options:
      heading_level: 3

## ClassificationTrainer

Fine-tunes the transformer sequence-classification model consumed by
[`ClassificationJudge`](judges.md) — binary (`no_match`/`match`) or
multiclass (`no_match`/`cit`/`cf`), inferred from the distinct labels in
`data`. Mirrors the paper's recipe: a fixed epoch count with no early
stopping, optional balanced class-weighting or focal loss, and the same
pair-truncation strategy `ClassificationJudge` uses at inference time.
`model.config.id2label`/`label2id` are set automatically before saving, so a
freshly trained model is immediately usable by `ClassificationJudge` — no
manual upload/labeling step.

```python
from locisimiles.training.classification import ClassificationTrainer, ClassificationTrainerConfig

config = ClassificationTrainerConfig(
    output_dir="models/classifier-3class",
    model_name="xlm-roberta-base",
    label_names={0: "no_match", 1: "cit", 2: "cf"},
    epochs=4,
    class_weight="balanced",
)
trainer = ClassificationTrainer(config)
trainer.fit(data=data)
model_path = trainer.save()
```

::: locisimiles.training.classification.trainer.ClassificationTrainerConfig
    options:
      heading_level: 3
::: locisimiles.training.classification.trainer.ClassificationTrainer
    options:
      heading_level: 3

### Threshold tuning and application

A trained classifier's raw argmax decision isn't always what you want —
`tune_threshold` sweeps one-vs-rest decision thresholds per positive class on
an evaluation `TrainingData` (with a tie-break rule for when two positive
classes both clear their threshold on the same pair), and `save()` persists
the result as a `threshold.json` sidecar alongside the model.

Applying a tuned `ThresholdSet` is a deliberate **standalone post-processing
step**, not something baked into `ClassificationJudge` — run `judge.judge(...)`
as normal, then optionally apply the tuned thresholds:

```python
trainer.tune_threshold(data=eval_data, method="max_f1")
model_path = trainer.save()  # writes threshold.json alongside the model

# --- later, at inference time ---
from locisimiles.pipeline.judge import ClassificationJudge
from locisimiles.training.classification.threshold import ThresholdSet, apply_thresholds_to_judgments

judge = ClassificationJudge(classification_name=str(model_path))
results = judge.judge(query=query_doc, candidates=candidates)

thresholds = ThresholdSet.from_json(model_path / "threshold.json")
final = apply_thresholds_to_judgments(results, thresholds, negative_label="no_match")
```

::: locisimiles.training.classification.threshold.ThresholdSet
    options:
      heading_level: 4
::: locisimiles.training.classification.threshold.apply_thresholds_to_judgments
    options:
      heading_level: 4
::: locisimiles.training.classification.threshold.apply_thresholds
    options:
      heading_level: 4

## EmbeddingTrainer

Fine-tunes the SentenceTransformer bi-encoder consumed by
[`EmbeddingCandidateGenerator`](generators.md). Uses `OnlineContrastiveLoss` by
default (the paper's production loss); `prompts` maps the `"query"`/`"match"`
sides to asymmetric prefixes (E5-style by default) and is baked into the saved
model, so `EmbeddingCandidateGenerator(embedding_model_name=...,
prompt_name="query"/"match")` works out of the box after training.

```python
from locisimiles.training.embedding import EmbeddingTrainer, EmbeddingTrainerConfig

config = EmbeddingTrainerConfig(
    output_dir="models/embedding",
    model_name="intfloat/multilingual-e5-large",
)
trainer = EmbeddingTrainer(config)
trainer.fit(data=data, eval_data=eval_data)  # eval_data is optional
model_path = trainer.save()
```

::: locisimiles.training.embedding.trainer.EmbeddingTrainerConfig
    options:
      heading_level: 3
::: locisimiles.training.embedding.trainer.EmbeddingTrainer
    options:
      heading_level: 3

## Negative sampling functions

The plain functions underlying `TrainingData`'s sampling methods, for callers
who want them independent of the `TrainingData` wrapper.

::: locisimiles.training.sampling.sample_random_pairs
    options:
      heading_level: 3
::: locisimiles.training.sampling.sample_random_negatives
    options:
      heading_level: 3
::: locisimiles.training.sampling.sample_hard_negatives
    options:
      heading_level: 3

## Migrating from the CSV-based trainer API

`LexicalClassifierTrainer` and `Word2VecTrainer` previously read a flat,
denormalized CSV directly. As of this training-API unification, all trainers
take `Document`/`TrainingData` instead — this is a breaking change:

```python
# Before
config = LexicalClassifierTrainerConfig(train_path="train.csv", output_dir="models/")
trainer = LexicalClassifierTrainer(config)
trainer.fit()

# After
config = LexicalClassifierTrainerConfig(output_dir="models/")
trainer = LexicalClassifierTrainer(config)
trainer.fit(data=TrainingData(query_doc, source_doc, ground_truth))
```

```python
# Before
config = Word2VecTrainerConfig(train_path="train.csv", output_dir="models/")
trainer = Word2VecTrainer(config)
trainer.fit()

# After
config = Word2VecTrainerConfig(output_dir="models/")
trainer = Word2VecTrainer(config)
trainer.fit(documents=[query_doc, source_doc])
```

`IntertextEvaluator`'s `ground_truth_csv` parameter is likewise renamed to
`ground_truth` (still accepts a path, `DataFrame`, or now also a
`GroundTruth`), and `load_example_ground_truth()` returns a `GroundTruth`
instead of a list of dicts.
