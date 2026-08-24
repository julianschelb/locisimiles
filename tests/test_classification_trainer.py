"""Tests for ClassificationTrainer.

Trains a real (not mocked) tiny transformer end to end, using a small public
HF test checkpoint, since this is the only way to verify the trainer and
:class:`~locisimiles.pipeline.judge.classification.ClassificationJudge` stay
consistent with each other (matching pair-encoding and id2label
conventions).
"""

from __future__ import annotations

import csv

import pytest

from locisimiles.document import Document
from locisimiles.ground_truth import GroundTruth
from locisimiles.training.data import TrainingData

TINY_MODEL = "hf-internal-testing/tiny-random-bert"


def _build_training_data(temp_dir, rows, name: str = "data") -> TrainingData:
    query_texts: list[str] = []
    corpus_texts: list[str] = []
    for query_text, corpus_text, _label in rows:
        if query_text not in query_texts:
            query_texts.append(query_text)
        if corpus_text not in corpus_texts:
            corpus_texts.append(corpus_text)
    query_id = {text: f"q{i}" for i, text in enumerate(query_texts)}
    corpus_id = {text: f"c{i}" for i, text in enumerate(corpus_texts)}

    query_csv = temp_dir / f"{name}_query.csv"
    with query_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["seg_id", "text"])
        for text in query_texts:
            writer.writerow([query_id[text], text])

    source_csv = temp_dir / f"{name}_source.csv"
    with source_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["seg_id", "text"])
        for text in corpus_texts:
            writer.writerow([corpus_id[text], text])

    ground_truth = GroundTruth(
        [
            {"query_id": query_id[query_text], "source_id": corpus_id[corpus_text], "label": label}
            for query_text, corpus_text, label in rows
        ]
    )
    return TrainingData(Document(query_csv), Document(source_csv), ground_truth)


BINARY_ROWS = [
    ("Arma virumque cano.", "Arma virumque cano qui primus.", "match"),
    ("Italiam fato profugus.", "Fato profugus Italiam venit.", "match"),
    ("Litora multum iactatus.", "Multum terris iactatus et alto.", "match"),
    ("Arma virumque cano.", "Completely unrelated cooking text.", "no_match"),
    ("Italiam fato profugus.", "Another unrelated politics segment.", "no_match"),
    ("Litora multum iactatus.", "Something else about finance entirely.", "no_match"),
]

THREE_CLASS_ROWS = BINARY_ROWS + [
    ("Arma virumque cano.", "Vaguely thematic reference to arms and war.", "cf"),
    ("Italiam fato profugus.", "Loosely related allusion to fate and exile.", "cf"),
]


@pytest.fixture
def tiny_config(temp_dir):
    from locisimiles.training.classification import ClassificationTrainerConfig

    return ClassificationTrainerConfig(
        output_dir=temp_dir,
        model_name=TINY_MODEL,
        epochs=1,
        batch_size=4,
        max_length=32,
    )


class TestClassificationTrainerFitSave:
    def test_fit_and_save_roundtrip_binary(self, temp_dir, tiny_config):
        from locisimiles.training.classification import ClassificationTrainer

        data = _build_training_data(temp_dir, BINARY_ROWS)
        trainer = ClassificationTrainer(tiny_config)
        model = trainer.fit(data=data)
        assert model is not None

        out_path = trainer.save()
        assert out_path.exists()
        assert out_path.name == tiny_config.output_filename
        assert (out_path / "config.json").exists()

    def test_id2label_set_from_resolved_labels(self, temp_dir, tiny_config):
        from transformers import AutoConfig

        from locisimiles.training.classification import ClassificationTrainer

        data = _build_training_data(temp_dir, BINARY_ROWS)
        trainer = ClassificationTrainer(tiny_config)
        trainer.fit(data=data)
        out_path = trainer.save()

        config = AutoConfig.from_pretrained(out_path)
        assert set(config.id2label.values()) == {"match", "no_match"}

    def test_explicit_label_names_override(self, temp_dir):
        from transformers import AutoConfig

        from locisimiles.training.classification import (
            ClassificationTrainer,
            ClassificationTrainerConfig,
        )

        data = _build_training_data(temp_dir, BINARY_ROWS)
        cfg = ClassificationTrainerConfig(
            output_dir=temp_dir,
            model_name=TINY_MODEL,
            epochs=1,
            batch_size=4,
            max_length=32,
            label_names={0: "no_match", 1: "match"},
        )
        trainer = ClassificationTrainer(cfg)
        trainer.fit(data=data)
        out_path = trainer.save()

        config = AutoConfig.from_pretrained(out_path)
        assert config.id2label[0] == "no_match"
        assert config.id2label[1] == "match"

    def test_saved_model_directly_loadable_by_classification_judge(
        self, temp_dir, tiny_config, query_document, source_document
    ):
        from locisimiles.pipeline.generator.bm25 import BM25CandidateGenerator
        from locisimiles.pipeline.judge.classification import ClassificationJudge
        from locisimiles.training.classification import ClassificationTrainer

        data = _build_training_data(temp_dir, BINARY_ROWS)
        trainer = ClassificationTrainer(tiny_config)
        trainer.fit(data=data)
        out_path = trainer.save()

        candidates = BM25CandidateGenerator().generate(
            query=query_document, source=source_document, top_k=2
        )
        judge = ClassificationJudge(classification_name=str(out_path))
        results = judge.judge(query=query_document, candidates=candidates)

        assert set(results.keys()) == set(candidates.keys())
        for judgments in results.values():
            for j in judgments:
                assert 0.0 <= j.judgment_score <= 1.0

    def test_three_class_saved_model_emits_class_metadata_via_judge(
        self, temp_dir, query_document, source_document
    ):
        from locisimiles.pipeline.generator.bm25 import BM25CandidateGenerator
        from locisimiles.pipeline.judge.classification import ClassificationJudge
        from locisimiles.training.classification import (
            ClassificationTrainer,
            ClassificationTrainerConfig,
        )

        data = _build_training_data(temp_dir, THREE_CLASS_ROWS)
        cfg = ClassificationTrainerConfig(
            output_dir=temp_dir,
            model_name=TINY_MODEL,
            epochs=1,
            batch_size=4,
            max_length=32,
        )
        trainer = ClassificationTrainer(cfg)
        trainer.fit(data=data)
        out_path = trainer.save()

        candidates = BM25CandidateGenerator().generate(
            query=query_document, source=source_document, top_k=2
        )
        judge = ClassificationJudge(classification_name=str(out_path))
        results = judge.judge(query=query_document, candidates=candidates)
        for judgments in results.values():
            for j in judgments:
                assert j.class_probabilities is not None
                assert len(j.class_probabilities) == 3
                assert pytest.approx(sum(j.class_probabilities.values()), abs=1e-4) == 1.0

    def test_empty_training_data_raises(self, temp_dir, tiny_config):
        from locisimiles.training.classification import ClassificationTrainer

        empty = _build_training_data(temp_dir, [])
        trainer = ClassificationTrainer(tiny_config)
        with pytest.raises(ValueError, match="No training rows found"):
            trainer.fit(data=empty)

    def test_save_before_fit_raises(self, temp_dir, tiny_config):
        from locisimiles.training.classification import ClassificationTrainer

        trainer = ClassificationTrainer(tiny_config)
        with pytest.raises(ValueError, match="No trained model available"):
            trainer.save()


class TestClassificationTrainerLossVariants:
    def test_balanced_class_weight_runs(self, temp_dir):
        from locisimiles.training.classification import (
            ClassificationTrainer,
            ClassificationTrainerConfig,
        )

        data = _build_training_data(temp_dir, BINARY_ROWS)
        cfg = ClassificationTrainerConfig(
            output_dir=temp_dir,
            model_name=TINY_MODEL,
            epochs=1,
            batch_size=4,
            max_length=32,
            class_weight="balanced",
        )
        trainer = ClassificationTrainer(cfg)
        trainer.fit(data=data)
        assert trainer.model is not None

    def test_focal_loss_runs(self, temp_dir):
        from locisimiles.training.classification import (
            ClassificationTrainer,
            ClassificationTrainerConfig,
        )

        data = _build_training_data(temp_dir, BINARY_ROWS)
        cfg = ClassificationTrainerConfig(
            output_dir=temp_dir,
            model_name=TINY_MODEL,
            epochs=1,
            batch_size=4,
            max_length=32,
            use_focal_loss=True,
            focal_gamma=2.0,
        )
        trainer = ClassificationTrainer(cfg)
        trainer.fit(data=data)
        assert trainer.model is not None


class TestClassificationTrainerThreshold:
    def test_tune_threshold_writes_sidecar_on_save(self, temp_dir):
        from locisimiles.training.classification import (
            ClassificationTrainer,
            ClassificationTrainerConfig,
        )
        from locisimiles.training.classification.threshold import ThresholdSet

        data = _build_training_data(temp_dir, THREE_CLASS_ROWS)
        cfg = ClassificationTrainerConfig(
            output_dir=temp_dir,
            model_name=TINY_MODEL,
            epochs=1,
            batch_size=4,
            max_length=32,
        )
        trainer = ClassificationTrainer(cfg)
        trainer.fit(data=data)
        result = trainer.tune_threshold(data=data, method="max_f1")
        assert set(result.thresholds) <= {"match", "cf"}

        out_path = trainer.save()
        threshold_path = out_path / "threshold.json"
        assert threshold_path.exists()
        loaded = ThresholdSet.from_json(threshold_path)
        assert loaded.thresholds == result.thresholds

    def test_no_threshold_json_when_not_tuned(self, temp_dir, tiny_config):
        from locisimiles.training.classification import ClassificationTrainer

        data = _build_training_data(temp_dir, BINARY_ROWS)
        trainer = ClassificationTrainer(tiny_config)
        trainer.fit(data=data)
        out_path = trainer.save()
        assert not (out_path / "threshold.json").exists()

    def test_tune_threshold_before_fit_raises(self, temp_dir, tiny_config):
        from locisimiles.training.classification import ClassificationTrainer

        data = _build_training_data(temp_dir, BINARY_ROWS)
        trainer = ClassificationTrainer(tiny_config)
        with pytest.raises(ValueError, match="No trained model available"):
            trainer.tune_threshold(data=data)


class TestClassificationTrainerLoadArtifacts:
    def test_load_artifacts_restores_label_mapping(self, temp_dir, tiny_config):
        from locisimiles.training.classification import ClassificationTrainer

        data = _build_training_data(temp_dir, BINARY_ROWS)
        trainer = ClassificationTrainer(tiny_config)
        trainer.fit(data=data)
        out_path = trainer.save()

        reloaded = ClassificationTrainer(tiny_config)
        reloaded.load_artifacts(out_path)
        assert set(reloaded._label_to_id) == {"match", "no_match"}
