"""Tests for Word2Vec generator, retrieval pipeline, and training helpers."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from locisimiles.pipeline._types import Candidate


class _FakeKeyedVectors:
    def __init__(self):
        self._known = {
            "arma",
            "uirumque",
            "cano",
            "troiae",
            "primus",
            "oris",
            "fato",
            "profugus",
            "italiam",
            "uenit",
        }

    def __contains__(self, key: str) -> bool:
        return key in self._known

    def similarity(self, w1: str, w2: str) -> float:
        if w1 == w2:
            return 1.0
        return 0.2


class _FakeWord2VecModel:
    def __init__(self):
        self.wv = _FakeKeyedVectors()


class TestWord2VecCandidateGenerator:
    @patch("locisimiles.pipeline.generator.word2vec._load_word2vec_model")
    def test_generate_returns_ranked_candidates(
        self,
        mock_loader,
        query_document,
        source_document,
        temp_dir,
    ):
        """Generator should return Candidate lists keyed by query segment id."""
        from locisimiles.pipeline.generator.word2vec import Word2VecCandidateGenerator

        model_path = temp_dir / "latin.model"
        model_path.write_text("stub", encoding="utf-8")
        mock_loader.return_value = _FakeWord2VecModel()

        generator = Word2VecCandidateGenerator(model_path=model_path)
        result = generator.generate(query=query_document, source=source_document, top_k=3)

        assert isinstance(result, dict)
        assert set(result.keys()) == {seg.id for seg in query_document}
        for _qid, candidates in result.items():
            assert len(candidates) <= 3
            assert all(isinstance(c, Candidate) for c in candidates)
            scores = [c.score for c in candidates]
            assert scores == sorted(scores, reverse=True)

    @patch("locisimiles.pipeline.generator.word2vec._load_word2vec_model")
    def test_missing_model_file_raises(self, mock_loader, temp_dir):
        """Missing model path should fail with a clear error."""
        from locisimiles.pipeline.generator.word2vec import Word2VecCandidateGenerator

        with pytest.raises(FileNotFoundError):
            Word2VecCandidateGenerator(model_path=temp_dir / "does_not_exist.model")

        mock_loader.assert_not_called()

    @patch("locisimiles.pipeline.generator.word2vec._load_word2vec_model")
    def test_order_free_and_interval_overrides(self, mock_loader, query_document, source_document, temp_dir):
        """Runtime kwargs should override interval/order settings for generate()."""
        from locisimiles.pipeline.generator.word2vec import Word2VecCandidateGenerator

        model_path = temp_dir / "latin.model"
        model_path.write_text("stub", encoding="utf-8")
        mock_loader.return_value = _FakeWord2VecModel()

        generator = Word2VecCandidateGenerator(model_path=model_path, interval=0, order_free=False)
        result = generator.generate(
            query=query_document,
            source=source_document,
            top_k=2,
            interval=2,
            order_free=True,
        )

        assert len(result) == len(list(query_document))
        assert generator.order_free is True


class TestWord2VecRetrievalPipeline:
    @patch("locisimiles.pipeline.word2vec.Word2VecCandidateGenerator")
    def test_pipeline_composition(self, mock_generator):
        """Pipeline should compose Word2Vec generator with ThresholdJudge."""
        from locisimiles.pipeline.judge.threshold import ThresholdJudge
        from locisimiles.pipeline.word2vec import Word2VecRetrievalPipeline

        mock_generator.return_value = MagicMock()
        pipeline = Word2VecRetrievalPipeline(model_path=Path(__file__), top_k=4)

        assert mock_generator.called
        assert isinstance(pipeline.judge, ThresholdJudge)
        assert pipeline.judge.top_k == 4


class TestWord2VecTrainer:
    def test_fit_and_save_roundtrip(self, temp_dir, monkeypatch):
        """Trainer should fit a model from CSV and save to configured output path."""
        from locisimiles.training.word2vec import Word2VecTrainer, Word2VecTrainerConfig

        train_csv = temp_dir / "train.csv"
        train_csv.write_text(
            "seg_id,text\n"
            "q1,Arma virumque cano\n"
            "q2,Fato profugus Italiam venit\n",
            encoding="utf-8",
        )

        model_instance = MagicMock()

        def _fake_ctor(*args, **kwargs):
            return model_instance

        fake_models = types.SimpleNamespace(Word2Vec=_fake_ctor)
        fake_gensim = types.SimpleNamespace(models=fake_models)

        monkeypatch.setitem(sys.modules, "gensim", fake_gensim)
        monkeypatch.setitem(sys.modules, "gensim.models", fake_models)

        cfg = Word2VecTrainerConfig(train_path=train_csv, output_dir=temp_dir)
        trainer = Word2VecTrainer(cfg)

        trainer.fit()
        out_path = trainer.save()

        model_instance.save.assert_called_once_with(str(out_path))
        assert out_path.name == cfg.output_filename
