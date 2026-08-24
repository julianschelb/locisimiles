"""Tests for the lexical (LogReg/GBDT) classifier trainer and judge.

Trains tiny real classifiers (not mocked) end to end, since the feature
pipeline and CLTK preprocessing are fast/lightweight, and this is the only
way to verify the trainer and judge stay consistent with each other (they
must build byte-identical features for a saved artifact to score correctly).
"""

from __future__ import annotations

import csv

import pytest

pytest.importorskip("cltk", reason="cltk has no release supporting this Python version")

from locisimiles.document import Document
from locisimiles.ground_truth import GroundTruth
from locisimiles.training.data import TrainingData


def _build_training_data(temp_dir, rows) -> TrainingData:
    """Build a TrainingData from (query_text, corpus_text, label) rows.

    Assigns stable segment ids to each distinct query/corpus text and writes
    them as Document CSVs, mirroring how a real caller would already have
    query/source corpora loaded separately from their labels.
    """
    query_texts: list[str] = []
    corpus_texts: list[str] = []
    for query_text, corpus_text, _label in rows:
        if query_text not in query_texts:
            query_texts.append(query_text)
        if corpus_text not in corpus_texts:
            corpus_texts.append(corpus_text)
    query_id = {text: f"q{i}" for i, text in enumerate(query_texts)}
    corpus_id = {text: f"c{i}" for i, text in enumerate(corpus_texts)}

    query_csv = temp_dir / "query_corpus.csv"
    with query_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["seg_id", "text"])
        for text in query_texts:
            writer.writerow([query_id[text], text])

    source_csv = temp_dir / "source_corpus.csv"
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
    (
        "Arma virumque cano Troiae qui primus ab oris.",
        "Arma virumque cano qui primus Troiae.",
        "match",
    ),
    ("Italiam fato profugus Laviniaque venit.", "Fato profugus Italiam venit.", "match"),
    (
        "Litora multum ille et terris iactatus et alto.",
        "Multum terris iactatus et alto litora.",
        "match",
    ),
    (
        "Arma virumque cano Troiae qui primus ab oris.",
        "Completely unrelated cooking recipe text.",
        "no_match",
    ),
    (
        "Italiam fato profugus Laviniaque venit.",
        "Another unrelated modern politics segment.",
        "no_match",
    ),
    (
        "Litora multum ille et terris iactatus et alto.",
        "Something else entirely about finance.",
        "no_match",
    ),
]

THREE_CLASS_ROWS = BINARY_ROWS + [
    (
        "Arma virumque cano Troiae qui primus ab oris.",
        "Vaguely thematic reference to arms and war.",
        "cf",
    ),
    (
        "Italiam fato profugus Laviniaque venit.",
        "Loosely related allusion to fate and exile.",
        "cf",
    ),
]


class TestLexicalClassifierTrainer:
    def test_fit_and_save_roundtrip_binary(self, temp_dir):
        from locisimiles.training.lexical import (
            LexicalClassifierTrainer,
            LexicalClassifierTrainerConfig,
        )

        data = _build_training_data(temp_dir, BINARY_ROWS)
        cfg = LexicalClassifierTrainerConfig(output_dir=temp_dir, classifier="logreg")
        trainer = LexicalClassifierTrainer(cfg)
        model = trainer.fit(data=data)
        assert model is not None

        out_path = trainer.save()
        assert out_path.exists()
        assert out_path.name == cfg.output_filename

    def test_fit_gbdt_classifier(self, temp_dir):
        from locisimiles.training.lexical import (
            LexicalClassifierTrainer,
            LexicalClassifierTrainerConfig,
        )

        data = _build_training_data(temp_dir, BINARY_ROWS)
        cfg = LexicalClassifierTrainerConfig(output_dir=temp_dir, classifier="gbdt")
        trainer = LexicalClassifierTrainer(cfg)
        trainer.fit(data=data)
        out_path = trainer.save()
        assert out_path.exists()

    def test_empty_training_data_raises(self, temp_dir):
        from locisimiles.training.lexical import (
            LexicalClassifierTrainer,
            LexicalClassifierTrainerConfig,
        )

        empty_data = _build_training_data(temp_dir, [])
        cfg = LexicalClassifierTrainerConfig(output_dir=temp_dir)
        trainer = LexicalClassifierTrainer(cfg)
        with pytest.raises(ValueError, match="No training rows found"):
            trainer.fit(data=empty_data)


class TestLexicalClassifierJudge:
    def test_binary_judge_scores_candidates(self, temp_dir, query_document, source_document):
        from locisimiles.pipeline.generator.bm25 import BM25CandidateGenerator
        from locisimiles.pipeline.judge.lexical_classifier import LexicalClassifierJudge
        from locisimiles.training.lexical import (
            LexicalClassifierTrainer,
            LexicalClassifierTrainerConfig,
        )

        data = _build_training_data(temp_dir, BINARY_ROWS)
        cfg = LexicalClassifierTrainerConfig(output_dir=temp_dir)
        trainer = LexicalClassifierTrainer(cfg)
        trainer.fit(data=data)
        artifact_path = trainer.save()

        candidates = BM25CandidateGenerator().generate(
            query=query_document, source=source_document, top_k=3
        )
        judge = LexicalClassifierJudge(artifact_path=str(artifact_path))
        results = judge.judge(query=query_document, candidates=candidates)

        assert set(results.keys()) == set(candidates.keys())
        for qid, judgments in results.items():
            assert len(judgments) == len(candidates[qid])
            for j in judgments:
                assert 0.0 <= j.judgment_score <= 1.0
                # Binary model with default config: no class metadata emitted.
                assert j.predicted_label is None
                assert j.class_probabilities is None

    def test_multiclass_judge_emits_class_metadata(self, temp_dir, query_document, source_document):
        from locisimiles.pipeline.generator.bm25 import BM25CandidateGenerator
        from locisimiles.pipeline.judge.lexical_classifier import LexicalClassifierJudge
        from locisimiles.training.lexical import (
            LexicalClassifierTrainer,
            LexicalClassifierTrainerConfig,
        )

        data = _build_training_data(temp_dir, THREE_CLASS_ROWS)
        cfg = LexicalClassifierTrainerConfig(
            output_dir=temp_dir,
            output_filename="lexical_3class.joblib",
            label_names={0: "cf", 1: "match", 2: "no_match"},
        )
        trainer = LexicalClassifierTrainer(cfg)
        trainer.fit(data=data)
        artifact_path = trainer.save()

        candidates = BM25CandidateGenerator().generate(
            query=query_document, source=source_document, top_k=2
        )
        judge = LexicalClassifierJudge(
            artifact_path=str(artifact_path),
            positive_labels=["match", "cf"],
        )
        results = judge.judge(query=query_document, candidates=candidates)

        for qid, judgments in results.items():
            for j in judgments:
                # Multiclass (or explicitly label-configured) models emit metadata.
                assert j.predicted_label is not None
                assert j.class_probabilities is not None
                assert len(j.class_probabilities) == 3
                assert pytest.approx(sum(j.class_probabilities.values()), abs=1e-6) == 1.0
                # judgment_score sums the configured positive classes.
                expected = sum(
                    p for label, p in j.class_probabilities.items() if label in ("match", "cf")
                )
                assert j.judgment_score == pytest.approx(expected)

    def test_empty_candidates_returns_empty_results(self, temp_dir, query_document):
        from locisimiles.pipeline.judge.lexical_classifier import LexicalClassifierJudge
        from locisimiles.training.lexical import (
            LexicalClassifierTrainer,
            LexicalClassifierTrainerConfig,
        )

        data = _build_training_data(temp_dir, BINARY_ROWS)
        cfg = LexicalClassifierTrainerConfig(output_dir=temp_dir)
        trainer = LexicalClassifierTrainer(cfg)
        trainer.fit(data=data)
        artifact_path = trainer.save()

        judge = LexicalClassifierJudge(artifact_path=str(artifact_path))
        empty_candidates = {str(seg.id): [] for seg in query_document}
        results = judge.judge(query=query_document, candidates=empty_candidates)
        assert all(len(v) == 0 for v in results.values())


class TestBM25LexicalTwoStagePipeline:
    def test_pipeline_end_to_end(self, temp_dir, query_document, source_document):
        from locisimiles.pipeline.bm25_lexical_two_stage import BM25LexicalTwoStagePipeline
        from locisimiles.training.lexical import (
            LexicalClassifierTrainer,
            LexicalClassifierTrainerConfig,
        )

        data = _build_training_data(temp_dir, BINARY_ROWS)
        cfg = LexicalClassifierTrainerConfig(output_dir=temp_dir)
        trainer = LexicalClassifierTrainer(cfg)
        trainer.fit(data=data)
        artifact_path = trainer.save()

        pipeline = BM25LexicalTwoStagePipeline(artifact_path=str(artifact_path))
        results = pipeline.run(query=query_document, source=source_document, top_k=3)
        assert set(results.keys()) == {seg.id for seg in query_document}
