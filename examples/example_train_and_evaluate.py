# examples/example_train_and_evaluate.py
"""End-to-end example: train, evaluate, and use your own models as a pipeline.

Walks through the full loop documented at
https://julianschelb.github.io/locisimiles/api/training/, using the
package's bundled example data (Hieronymus queries vs. Vergil sources):

1. Load the query/source corpora and known positive matches.
2. Build training data, sampling negatives (the example ground truth only
   has positive rows).
3. Train a classifier and an embedding model.
4. Combine both into a two-stage pipeline and run it.
5. Evaluate the trained pipeline, including tuning classification
   thresholds.
6. Reproduce the paper's mean+-std-across-folds protocol with K-fold
   cross-validation.

This uses real (small) pretrained backbones and will download ~1.5GB of
model weights the first time it runs. Epoch counts are kept low since the
example dataset is tiny (11 query segments, 10 known matches) -- this is
meant to demonstrate the mechanics end to end, not to produce a
strong model.
"""

from pathlib import Path

from locisimiles import load_example_ground_truth, load_example_query, load_example_source
from locisimiles.evaluator import IntertextEvaluator
from locisimiles.pipeline import Pipeline, TwoStagePipeline, pretty_print
from locisimiles.pipeline.generator.exhaustive import ExhaustiveCandidateGenerator
from locisimiles.pipeline.judge import ClassificationJudge
from locisimiles.training.classification import ClassificationTrainer, ClassificationTrainerConfig
from locisimiles.training.cross_validation import cross_validate, evaluate_with_pipeline
from locisimiles.training.data import TrainingData
from locisimiles.training.embedding import EmbeddingTrainer, EmbeddingTrainerConfig

OUTPUT_DIR = Path("./trained_models")

# ---------------------------------------------------------------------------
# 1. Load the query/source corpora and known positive matches.
# ---------------------------------------------------------------------------
query_doc = load_example_query()
source_doc = load_example_source()
positives = load_example_ground_truth()  # 10 known positive (label=1) pairs

print(f"Query segments: {len(query_doc)}")
print(f"Source segments: {len(source_doc)}")
print(f"Known positive pairs: {len(positives)}")
print("=" * 70)

# ---------------------------------------------------------------------------
# 2. Build training data. The bundled example ground truth only has
#    positive rows, so sample negatives before training -- this uses the
#    default method (sample_random_negatives, i.e. <qry,rnd>). The paper's
#    best-performing method on every reported metric is
#    `sample_mixed_negatives()` (random + embedding-mined hard negatives);
#    swap it in here for a stronger classifier at the cost of loading an
#    extra embedding model to mine hard negatives.
# ---------------------------------------------------------------------------
data = TrainingData(query_doc, source_doc, positives).sample_random_negatives(
    n_per_query=3, label=0, seed=42
)
print(f"Training pairs after negative sampling: {len(data)}")
print("=" * 70)

# ---------------------------------------------------------------------------
# 3. Train a binary classifier and an embedding retriever on the same data.
#    Both trainers default to a fixed epoch count with no checkpoint
#    selection, matching the paper's recipe; pass eval_data and
#    select_best_checkpoint=True (optionally with early_stopping_patience)
#    to instead keep the best-scoring epoch -- see docs/api/training.md.
# ---------------------------------------------------------------------------
classification_trainer = ClassificationTrainer(
    ClassificationTrainerConfig(
        output_dir=OUTPUT_DIR,
        model_name="xlm-roberta-base",
        epochs=2,
        batch_size=4,
    )
)
classification_trainer.fit(data=data)

# Tune a decision threshold instead of using a fixed 0.5 cutoff. The paper
# tunes on the training split itself (not a held-out split, since its
# validation split shares positives with the test split) -- see
# docs/api/training.md's threshold section. `negative_label` must match
# the label used for sampled negatives above ("0", not the default
# "no_match") -- this dataset only has one positive class ("1"), so
# `apply_thresholds_to_judgments`'s one-vs-rest tie-break machinery isn't
# needed here; for a multiclass classifier (e.g. no_match/cit/cf) it would
# be the natural way to apply a tuned ThresholdSet at inference time.
tuned_thresholds = classification_trainer.tune_threshold(
    data=data, method="max_f1", negative_label="0"
)
tuned_threshold = tuned_thresholds.thresholds["1"]
print(f"Tuned decision threshold: {tuned_threshold:.2f}")
classification_model_path = classification_trainer.save()
print(f"Classifier saved to: {classification_model_path}")

embedding_trainer = EmbeddingTrainer(
    EmbeddingTrainerConfig(
        output_dir=OUTPUT_DIR,
        model_name="intfloat/multilingual-e5-small",
        epochs=2,
        batch_size=4,
    )
)
embedding_trainer.fit(data=data)
embedding_model_path = embedding_trainer.save()
print(f"Embedding model saved to: {embedding_model_path}")
print("=" * 70)

# Both trainers wrap standard transformers/sentence-transformers objects, so
# uploading to the Hugging Face Hub needs no extra helper -- it's already
# built in:
#
#   classification_trainer.model.push_to_hub("your-username/your-model-name")
#   embedding_trainer.model.push_to_hub("your-username/your-embedding-model")

# ---------------------------------------------------------------------------
# 4. Combine both trained models into a two-stage pipeline and run it.
# ---------------------------------------------------------------------------
pipeline = TwoStagePipeline(
    classification_name=str(classification_model_path),
    embedding_model_name=str(embedding_model_path),
    device="cpu",
)
results = pipeline.run(query=query_doc, source=source_doc, top_k=5)
pretty_print(results)
print("=" * 70)

# ---------------------------------------------------------------------------
# 5. Evaluate the trained pipeline against the known ground truth, using the
#    tuned threshold from step 3 instead of an arbitrary fixed cutoff.
# ---------------------------------------------------------------------------
evaluator = IntertextEvaluator(
    query_doc=query_doc,
    source_doc=source_doc,
    ground_truth=positives,
    pipeline=pipeline,
    top_k=5,
    threshold=tuned_threshold,
)
print("Evaluation (single train/eval split):")
print(evaluator.evaluate(average="macro").to_string(index=False))
print("=" * 70)

# ---------------------------------------------------------------------------
# 6. Reproduce the paper's mean+-std-across-folds protocol with K-fold CV.
#    Retrains a fresh classifier per fold and evaluates each fold's
#    held-out queries. With only ~10 positive examples split across 3
#    folds, each fold's training set is tiny -- don't expect strong
#    per-fold scores here; the point is to show the mechanics (grouped
#    fold splitting, per-fold retraining, metric aggregation) work
#    correctly, not to produce a strong model from 10 examples.
# ---------------------------------------------------------------------------


def train_fn(fold_train_data: TrainingData) -> Path:
    fold_trainer = ClassificationTrainer(
        ClassificationTrainerConfig(
            output_dir=OUTPUT_DIR / "cv",
            model_name="xlm-roberta-base",
            epochs=1,
            batch_size=4,
        )
    )
    fold_trainer.fit(data=fold_train_data)
    return fold_trainer.save()


def evaluate_fn(model_path: Path, fold_eval_data: TrainingData) -> dict:
    judge = ClassificationJudge(classification_name=str(model_path))
    fold_pipeline = Pipeline(generator=ExhaustiveCandidateGenerator(), judge=judge)
    return evaluate_with_pipeline(fold_pipeline, fold_eval_data, top_k=5)


cv_result = cross_validate(
    query_doc=query_doc,
    source_doc=source_doc,
    ground_truth=data.ground_truth,
    n_folds=3,
    train_fn=train_fn,
    evaluate_fn=evaluate_fn,
)
print("Cross-validation results across 3 folds:")
print(cv_result.to_dataframe().to_string(index=False))
