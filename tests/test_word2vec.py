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
    def test_order_free_and_interval_overrides(
        self, mock_loader, query_document, source_document, temp_dir
    ):
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


class _PairwiseKeyedVectors:
    """Fake KeyedVectors returning a fixed raw cosine per unordered word pair."""

    def __init__(self, pairwise: dict[frozenset[str], float]):
        self._pairwise = pairwise
        self._known = {w for pair in pairwise for w in pair}

    def __contains__(self, key: str) -> bool:
        return key in self._known

    def similarity(self, w1: str, w2: str) -> float:
        if w1 == w2:
            return 1.0
        return self._pairwise[frozenset((w1, w2))]


class TestWord2VecBigramPairScore:
    """Regression tests for the Burns et al. (2021) bigram-pair formula.

    ``_word_similarity`` rescales each raw cosine from ``[-1, 1]`` to
    ``[0, 1]`` via the affine, monotonically increasing map
    ``f(x) = (x + 1) / 2``. Because the bigram-pair score is built only from
    ``max``/``mean`` over these per-word similarities, and both operations
    commute with an affine monotonic map, the package's rescaled output for
    any bigram pair equals ``f(raw_score)``. Expected values below are
    derived accordingly from raw-cosine worked examples.
    """

    def _make_generator(self, mock_loader, temp_dir, pairwise):
        from locisimiles.pipeline.generator.word2vec import Word2VecCandidateGenerator

        model_path = temp_dir / "latin.model"
        model_path.write_text("stub", encoding="utf-8")
        model = _FakeWord2VecModel()
        model.wv = _PairwiseKeyedVectors(pairwise)
        mock_loader.return_value = model
        return Word2VecCandidateGenerator(model_path=model_path)

    @patch("locisimiles.pipeline.generator.word2vec._load_word2vec_model")
    def test_matches_burns_2021_worked_example(self, mock_loader, temp_dir):
        """flammifero Olympo vs. flammifera nocte -> paper's worked example.

        Raw cosines: flammifer~flammifer = 1.0 (same word), olympus~nox =
        0.35 (the "remaining" pair), both cross pairs low. Paper's raw
        result: (1.0 + 0.35) / 2 = 0.675 -> rescaled: (0.675 + 1) / 2 = 0.8375.
        """
        generator = self._make_generator(
            mock_loader,
            temp_dir,
            pairwise={
                frozenset({"olympus", "nox"}): 0.35,
                frozenset({"flammifer", "nox"}): 0.05,
                frozenset({"olympus", "flammifer"}): 0.05,
            },
        )
        score = generator._bigram_pair_score(("flammifer", "olympus"), ("flammifer", "nox"))
        assert score == pytest.approx(0.8375)

    @patch("locisimiles.pipeline.generator.word2vec._load_word2vec_model")
    def test_picks_higher_scoring_pairing_not_just_global_max(self, mock_loader, temp_dir):
        """The best single cross-cosine need not belong to the better-scoring pairing.

        Raw cosines: q1s1=0.9, q1s2=0.85, q2s1=0.85, q2s2=0.1. The single
        highest value (0.9) belongs to the "direct" pairing (q1s1, q2s2),
        whose mean (0.5) is *lower* than the "crossed" pairing's mean
        (mean(0.85, 0.85) = 0.85). The correct Burns formula still follows
        the global max (0.9) and pairs it with its non-overlapping partner
        (q2s2=0.1): raw = (0.9 + 0.1) / 2 = 0.5 -> rescaled = 0.75. A
        max-of-pairings implementation would incorrectly return the
        crossed pairing's mean instead (rescaled 0.925).
        """
        generator = self._make_generator(
            mock_loader,
            temp_dir,
            pairwise={
                frozenset({"q1", "s1"}): 0.9,
                frozenset({"q1", "s2"}): 0.85,
                frozenset({"q2", "s1"}): 0.85,
                frozenset({"q2", "s2"}): 0.1,
            },
        )
        score = generator._bigram_pair_score(("q1", "q2"), ("s1", "s2"))
        assert score == pytest.approx(0.75)
        assert score != pytest.approx(0.925)  # the old, incorrect formula's result

    @patch("locisimiles.pipeline.generator.word2vec._load_word2vec_model")
    def test_requires_all_four_words_in_vocabulary(self, mock_loader, temp_dir):
        """Returns None (not a partial score) when one of the four words is OOV.

        ``s2`` never appears in the pairwise similarity map, so it is
        out-of-vocabulary; two of the four cross pairs involve it and
        therefore cannot be scored, which should void the whole bigram pair
        rather than silently scoring on the remaining two.
        """
        generator = self._make_generator(
            mock_loader,
            temp_dir,
            pairwise={frozenset({"q1", "s1"}): 0.9, frozenset({"q2", "s1"}): 0.5},
        )
        score = generator._bigram_pair_score(("q1", "q2"), ("s1", "s2"))
        assert score is None


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
        """Trainer should fit a model from Documents and save to configured output path."""
        from locisimiles.document import Document
        from locisimiles.training.word2vec import Word2VecTrainer, Word2VecTrainerConfig

        train_csv = temp_dir / "train.csv"
        train_csv.write_text(
            "seg_id,text\nq1,Arma virumque cano\nq2,Fato profugus Italiam venit\n",
            encoding="utf-8",
        )
        document = Document(train_csv)

        model_instance = MagicMock()

        def _fake_ctor(*args, **kwargs):
            return model_instance

        fake_models = types.SimpleNamespace(Word2Vec=_fake_ctor)
        fake_gensim = types.SimpleNamespace(models=fake_models)

        monkeypatch.setitem(sys.modules, "gensim", fake_gensim)
        monkeypatch.setitem(sys.modules, "gensim.models", fake_models)

        cfg = Word2VecTrainerConfig(output_dir=temp_dir)
        trainer = Word2VecTrainer(cfg)

        trainer.fit(documents=[document])
        out_path = trainer.save()

        model_instance.save.assert_called_once_with(str(out_path))
        assert out_path.name == cfg.output_filename

    def test_fit_multiple_documents(self, temp_dir, monkeypatch):
        """Trainer should pool sentences across multiple Documents."""
        from locisimiles.document import Document
        from locisimiles.training.word2vec import Word2VecTrainer, Word2VecTrainerConfig

        query_csv = temp_dir / "query.csv"
        query_csv.write_text("seg_id,text\nq1,Arma virumque cano\n", encoding="utf-8")
        source_csv = temp_dir / "source.csv"
        source_csv.write_text("seg_id,text\ns1,Fato profugus Italiam venit\n", encoding="utf-8")

        model_instance = MagicMock()
        captured = {}

        def _fake_ctor(*, sentences, **kwargs):
            captured["sentences"] = sentences
            return model_instance

        fake_models = types.SimpleNamespace(Word2Vec=_fake_ctor)
        fake_gensim = types.SimpleNamespace(models=fake_models)
        monkeypatch.setitem(sys.modules, "gensim", fake_gensim)
        monkeypatch.setitem(sys.modules, "gensim.models", fake_models)

        cfg = Word2VecTrainerConfig(output_dir=temp_dir)
        trainer = Word2VecTrainer(cfg)
        trainer.fit(documents=[Document(query_csv), Document(source_csv)])

        assert len(captured["sentences"]) == 2

    def test_fit_no_sentences_raises(self, temp_dir, monkeypatch):
        """Fitting on documents with no tokenizable text should raise clearly."""
        from locisimiles.document import Document
        from locisimiles.training.word2vec import Word2VecTrainer, Word2VecTrainerConfig

        empty_csv = temp_dir / "empty.csv"
        empty_csv.write_text("seg_id,text\n", encoding="utf-8")

        fake_models = types.SimpleNamespace(Word2Vec=MagicMock())
        fake_gensim = types.SimpleNamespace(models=fake_models)
        monkeypatch.setitem(sys.modules, "gensim", fake_gensim)
        monkeypatch.setitem(sys.modules, "gensim.models", fake_models)

        cfg = Word2VecTrainerConfig(output_dir=temp_dir)
        trainer = Word2VecTrainer(cfg)
        with pytest.raises(ValueError, match="No non-empty tokenized"):
            trainer.fit(documents=[Document(empty_csv)])
