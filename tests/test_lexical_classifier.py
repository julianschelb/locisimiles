"""Tests for the lexical (LogReg/GBDT) classifier trainer and judge.

Trains tiny real classifiers (not mocked) end to end, since the feature
pipeline and CLTK preprocessing are fast/lightweight, and this is the only
way to verify the trainer and judge stay consistent with each other (they
must build byte-identical features for a saved artifact to score correctly).
"""

from __future__ import annotations

import pytest


def _write_training_csv(path, rows):
    lines = ["query_text,corpus_text,label"]
    for query_text, corpus_text, label in rows:
        lines.append(f'"{query_text}","{corpus_text}",{label}')
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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

        train_csv = temp_dir / "train.csv"
        _write_training_csv(train_csv, BINARY_ROWS)

        cfg = LexicalClassifierTrainerConfig(
            train_path=train_csv, output_dir=temp_dir, classifier="logreg"
        )
        trainer = LexicalClassifierTrainer(cfg)
        model = trainer.fit()
        assert model is not None

        out_path = trainer.save()
        assert out_path.exists()
        assert out_path.name == cfg.output_filename

    def test_fit_gbdt_classifier(self, temp_dir):
        from locisimiles.training.lexical import (
            LexicalClassifierTrainer,
            LexicalClassifierTrainerConfig,
        )

        train_csv = temp_dir / "train.csv"
        _write_training_csv(train_csv, BINARY_ROWS)

        cfg = LexicalClassifierTrainerConfig(
            train_path=train_csv, output_dir=temp_dir, classifier="gbdt"
        )
        trainer = LexicalClassifierTrainer(cfg)
        trainer.fit()
        out_path = trainer.save()
        assert out_path.exists()

    def test_missing_columns_raises(self, temp_dir):
        from locisimiles.training.lexical import (
            LexicalClassifierTrainer,
            LexicalClassifierTrainerConfig,
        )

        train_csv = temp_dir / "bad.csv"
        train_csv.write_text("a,b,c\n1,2,3\n", encoding="utf-8")
        cfg = LexicalClassifierTrainerConfig(train_path=train_csv, output_dir=temp_dir)
        trainer = LexicalClassifierTrainer(cfg)
        with pytest.raises(ValueError, match="missing required columns"):
            trainer.fit()


class TestLexicalClassifierJudge:
    def test_binary_judge_scores_candidates(self, temp_dir, query_document, source_document):
        from locisimiles.pipeline.generator.bm25 import BM25CandidateGenerator
        from locisimiles.pipeline.judge.lexical_classifier import LexicalClassifierJudge
        from locisimiles.training.lexical import (
            LexicalClassifierTrainer,
            LexicalClassifierTrainerConfig,
        )

        train_csv = temp_dir / "train.csv"
        _write_training_csv(train_csv, BINARY_ROWS)
        cfg = LexicalClassifierTrainerConfig(train_path=train_csv, output_dir=temp_dir)
        trainer = LexicalClassifierTrainer(cfg)
        trainer.fit()
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

        train_csv = temp_dir / "train_3class.csv"
        _write_training_csv(train_csv, THREE_CLASS_ROWS)
        cfg = LexicalClassifierTrainerConfig(
            train_path=train_csv,
            output_dir=temp_dir,
            output_filename="lexical_3class.joblib",
            label_names={0: "cf", 1: "match", 2: "no_match"},
        )
        trainer = LexicalClassifierTrainer(cfg)
        trainer.fit()
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

        train_csv = temp_dir / "train.csv"
        _write_training_csv(train_csv, BINARY_ROWS)
        cfg = LexicalClassifierTrainerConfig(train_path=train_csv, output_dir=temp_dir)
        trainer = LexicalClassifierTrainer(cfg)
        trainer.fit()
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

        train_csv = temp_dir / "train.csv"
        _write_training_csv(train_csv, BINARY_ROWS)
        cfg = LexicalClassifierTrainerConfig(train_path=train_csv, output_dir=temp_dir)
        trainer = LexicalClassifierTrainer(cfg)
        trainer.fit()
        artifact_path = trainer.save()

        pipeline = BM25LexicalTwoStagePipeline(artifact_path=str(artifact_path))
        results = pipeline.run(query=query_document, source=source_document, top_k=3)
        assert set(results.keys()) == {seg.id for seg in query_document}
