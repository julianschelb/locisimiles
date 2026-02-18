"""
Cross-combination tests: every generator × every judge via Pipeline.

Generators:
  - ExhaustiveCandidateGenerator  (no ML)
  - EmbeddingCandidateGenerator   (mocked sentence-transformer + ChromaDB)
  - RuleBasedCandidateGenerator   (no ML, lexical matching)

Judges:
  - IdentityJudge                 (no ML, pass-through)
  - ThresholdJudge                (no ML, top-k / threshold)
  - ClassificationJudge           (mocked transformer)

Each combination: create temp CSVs → load Documents → Pipeline.run() → save
CSV & JSON → validate output files.
"""

import csv
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from locisimiles.document import Document
from locisimiles.pipeline._types import CandidateJudge
from locisimiles.pipeline.generator.embedding import EmbeddingCandidateGenerator
from locisimiles.pipeline.generator.exhaustive import ExhaustiveCandidateGenerator
from locisimiles.pipeline.generator.rule_based import RuleBasedCandidateGenerator
from locisimiles.pipeline.judge.classification import ClassificationJudge
from locisimiles.pipeline.judge.identity import IdentityJudge
from locisimiles.pipeline.judge.threshold import ThresholdJudge
from locisimiles.pipeline.pipeline import Pipeline

# ============== Fixtures ==============

QUERY_CSV = (
    "seg_id,text\n"
    "q1,Arma virumque cano Troiae qui primus ab oris.\n"
    "q2,Italiam fato profugus Laviniaque venit.\n"
)

SOURCE_CSV = (
    "seg_id,text\n"
    "s1,Arma virumque cano qui primus Troiae.\n"
    "s2,Fato profugus Italiam venit litora.\n"
    "s3,Completely unrelated text here indeed.\n"
)


@pytest.fixture
def combo_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def query_doc(combo_dir):
    p = combo_dir / "query.csv"
    p.write_text(QUERY_CSV, encoding="utf-8")
    return Document(p)


@pytest.fixture
def source_doc(combo_dir):
    p = combo_dir / "source.csv"
    p.write_text(SOURCE_CSV, encoding="utf-8")
    return Document(p)


# ============== Mock builders ==============


def _mock_sentence_transformer():
    mock = MagicMock()

    def _encode(texts, **_kw):
        rng = np.random.default_rng(42)
        vecs = rng.standard_normal((len(texts), 384)).astype("float32")
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / norms

    mock.encode.side_effect = _encode
    return mock


class _FakeEncoding(dict):
    def to(self, device):
        return self


def _mock_classifier():
    model = MagicMock()
    tokenizer = MagicMock()

    tokenizer.num_special_tokens_to_add.return_value = 3
    tokenizer.tokenize.side_effect = lambda x: x.split()[:250]
    tokenizer.convert_tokens_to_string.side_effect = lambda x: " ".join(x)

    def _tok_call(*args, **kwargs):
        batch_size = len(args[0]) if args and isinstance(args[0], list) else 1
        return _FakeEncoding(
            input_ids=torch.zeros(batch_size, 64, dtype=torch.long),
            attention_mask=torch.ones(batch_size, 64, dtype=torch.long),
        )

    tokenizer.side_effect = _tok_call
    tokenizer.return_value = _tok_call(["dummy"], ["dummy"])

    def _forward(**kwargs):
        ids = kwargs.get("input_ids", torch.zeros(1, 1))
        result = MagicMock()
        result.logits = torch.randn(ids.shape[0], 2)
        return result

    model.side_effect = _forward
    model.return_value = _forward()
    model.to.return_value = model
    model.eval.return_value = model
    return model, tokenizer


# ============== Validators ==============


def _assert_valid_csv(path: Path, expected_qids: set[str]):
    assert path.exists()
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert set(reader.fieldnames) == {
        "query_id",
        "source_id",
        "source_text",
        "candidate_score",
        "judgment_score",
    }
    found = {r["query_id"] for r in rows}
    assert found == expected_qids
    for r in rows:
        float(r["judgment_score"])


def _assert_valid_json(path: Path, expected_qids: set[str]):
    assert path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert set(data.keys()) == expected_qids
    for matches in data.values():
        for m in matches:
            assert "source_id" in m and "judgment_score" in m


def _run_and_save(pipeline: Pipeline, query_doc, source_doc, out_dir, **run_kw):
    """Run a pipeline, save results, and validate the outputs."""
    results = pipeline.run(query=query_doc, source=source_doc, **run_kw)

    assert isinstance(results, dict)
    for lst in results.values():
        assert all(isinstance(item, CandidateJudge) for item in lst)

    csv_out = out_dir / "results.csv"
    pipeline.to_csv(csv_out)
    assert csv_out.exists()

    json_out = out_dir / "results.json"
    pipeline.to_json(json_out)
    assert json_out.exists()

    return results


# ============== ExhaustiveCandidateGenerator × all judges ==============


class TestExhaustiveWithIdentity:
    def test_run_and_save(self, query_doc, source_doc, combo_dir):
        pipeline = Pipeline(
            generator=ExhaustiveCandidateGenerator(),
            judge=IdentityJudge(),
        )
        results = _run_and_save(pipeline, query_doc, source_doc, combo_dir)
        # Exhaustive: 2 queries × 3 sources
        assert set(results.keys()) == {"q1", "q2"}
        for lst in results.values():
            assert len(lst) == 3
            assert all(j.judgment_score == 1.0 for j in lst)
        _assert_valid_csv(combo_dir / "results.csv", {"q1", "q2"})
        _assert_valid_json(combo_dir / "results.json", {"q1", "q2"})


class TestExhaustiveWithThreshold:
    def test_run_and_save(self, query_doc, source_doc, combo_dir):
        pipeline = Pipeline(
            generator=ExhaustiveCandidateGenerator(),
            judge=ThresholdJudge(top_k=2),
        )
        results = _run_and_save(pipeline, query_doc, source_doc, combo_dir)
        assert set(results.keys()) == {"q1", "q2"}
        for lst in results.values():
            assert len(lst) == 3  # all returned, but only top-2 scored 1.0
            positives = [j for j in lst if j.judgment_score == 1.0]
            assert len(positives) == 2
        _assert_valid_csv(combo_dir / "results.csv", {"q1", "q2"})
        _assert_valid_json(combo_dir / "results.json", {"q1", "q2"})


class TestExhaustiveWithClassification:
    @patch("locisimiles.pipeline.judge.classification.AutoModelForSequenceClassification")
    @patch("locisimiles.pipeline.judge.classification.AutoTokenizer")
    def test_run_and_save(self, mock_tok_cls, mock_mdl_cls, query_doc, source_doc, combo_dir):
        model, tokenizer = _mock_classifier()
        mock_mdl_cls.from_pretrained.return_value = model
        mock_tok_cls.from_pretrained.return_value = tokenizer

        pipeline = Pipeline(
            generator=ExhaustiveCandidateGenerator(),
            judge=ClassificationJudge(device="cpu"),
        )
        results = _run_and_save(pipeline, query_doc, source_doc, combo_dir)
        assert set(results.keys()) == {"q1", "q2"}
        for lst in results.values():
            assert len(lst) == 3
        _assert_valid_csv(combo_dir / "results.csv", {"q1", "q2"})
        _assert_valid_json(combo_dir / "results.json", {"q1", "q2"})


# ============== EmbeddingCandidateGenerator × all judges ==============


class TestEmbeddingWithIdentity:
    @patch("locisimiles.pipeline.generator.embedding.SentenceTransformer")
    def test_run_and_save(self, mock_st_cls, query_doc, source_doc, combo_dir):
        mock_st_cls.return_value = _mock_sentence_transformer()

        pipeline = Pipeline(
            generator=EmbeddingCandidateGenerator(device="cpu"),
            judge=IdentityJudge(),
        )
        results = _run_and_save(pipeline, query_doc, source_doc, combo_dir, top_k=2)
        assert set(results.keys()) == {"q1", "q2"}
        for lst in results.values():
            assert len(lst) == 2
            assert all(j.judgment_score == 1.0 for j in lst)
        _assert_valid_csv(combo_dir / "results.csv", {"q1", "q2"})
        _assert_valid_json(combo_dir / "results.json", {"q1", "q2"})


class TestEmbeddingWithThreshold:
    @patch("locisimiles.pipeline.generator.embedding.SentenceTransformer")
    def test_run_and_save(self, mock_st_cls, query_doc, source_doc, combo_dir):
        mock_st_cls.return_value = _mock_sentence_transformer()

        pipeline = Pipeline(
            generator=EmbeddingCandidateGenerator(device="cpu"),
            judge=ThresholdJudge(top_k=1),
        )
        # top_k is forwarded to both stages; generator returns 3 candidates,
        # judge receives top_k=3 from kwargs which overrides instance top_k.
        # Use a separate run to exercise the threshold judge properly.
        candidates = pipeline.generate_candidates(
            query=query_doc,
            source=source_doc,
            top_k=3,
        )
        results = pipeline.judge_candidates(
            query=query_doc,
            candidates=candidates,
            top_k=1,
        )
        pipeline.to_csv(combo_dir / "results.csv", results=results)
        pipeline.to_json(combo_dir / "results.json", results=results)

        assert set(results.keys()) == {"q1", "q2"}
        for lst in results.values():
            assert len(lst) == 3
            positives = [j for j in lst if j.judgment_score == 1.0]
            assert len(positives) == 1
        _assert_valid_csv(combo_dir / "results.csv", {"q1", "q2"})
        _assert_valid_json(combo_dir / "results.json", {"q1", "q2"})


class TestEmbeddingWithClassification:
    @patch("locisimiles.pipeline.generator.embedding.SentenceTransformer")
    @patch("locisimiles.pipeline.judge.classification.AutoModelForSequenceClassification")
    @patch("locisimiles.pipeline.judge.classification.AutoTokenizer")
    def test_run_and_save(
        self, mock_tok_cls, mock_mdl_cls, mock_st_cls, query_doc, source_doc, combo_dir
    ):
        mock_st_cls.return_value = _mock_sentence_transformer()
        model, tokenizer = _mock_classifier()
        mock_mdl_cls.from_pretrained.return_value = model
        mock_tok_cls.from_pretrained.return_value = tokenizer

        pipeline = Pipeline(
            generator=EmbeddingCandidateGenerator(device="cpu"),
            judge=ClassificationJudge(device="cpu"),
        )
        results = _run_and_save(pipeline, query_doc, source_doc, combo_dir, top_k=2)
        assert set(results.keys()) == {"q1", "q2"}
        for lst in results.values():
            assert len(lst) == 2
        _assert_valid_csv(combo_dir / "results.csv", {"q1", "q2"})
        _assert_valid_json(combo_dir / "results.json", {"q1", "q2"})


# ============== RuleBasedCandidateGenerator × all judges ==============


class TestRuleBasedWithIdentity:
    def test_run_and_save(self, query_doc, source_doc, combo_dir):
        pipeline = Pipeline(
            generator=RuleBasedCandidateGenerator(
                min_shared_words=1,
                min_complura=2,
                max_distance=5,
            ),
            judge=IdentityJudge(),
        )
        results = _run_and_save(pipeline, query_doc, source_doc, combo_dir)
        assert isinstance(results, dict)
        for qid, lst in results.items():
            assert qid in {"q1", "q2"}
            for j in lst:
                assert j.judgment_score == 1.0
        _assert_valid_csv_or_empty(combo_dir / "results.csv")
        _assert_valid_json_or_empty(combo_dir / "results.json")


class TestRuleBasedWithThreshold:
    def test_run_and_save(self, query_doc, source_doc, combo_dir):
        pipeline = Pipeline(
            generator=RuleBasedCandidateGenerator(
                min_shared_words=1,
                min_complura=2,
                max_distance=5,
            ),
            judge=ThresholdJudge(top_k=1),
        )
        results = _run_and_save(pipeline, query_doc, source_doc, combo_dir)
        assert isinstance(results, dict)
        for qid, lst in results.items():
            assert qid in {"q1", "q2"}
            positives = [j for j in lst if j.judgment_score == 1.0]
            assert len(positives) <= 1
        _assert_valid_csv_or_empty(combo_dir / "results.csv")
        _assert_valid_json_or_empty(combo_dir / "results.json")


class TestRuleBasedWithClassification:
    @patch("locisimiles.pipeline.judge.classification.AutoModelForSequenceClassification")
    @patch("locisimiles.pipeline.judge.classification.AutoTokenizer")
    def test_run_and_save(self, mock_tok_cls, mock_mdl_cls, query_doc, source_doc, combo_dir):
        model, tokenizer = _mock_classifier()
        mock_mdl_cls.from_pretrained.return_value = model
        mock_tok_cls.from_pretrained.return_value = tokenizer

        pipeline = Pipeline(
            generator=RuleBasedCandidateGenerator(
                min_shared_words=1,
                min_complura=2,
                max_distance=5,
            ),
            judge=ClassificationJudge(device="cpu"),
        )
        results = _run_and_save(pipeline, query_doc, source_doc, combo_dir)
        assert isinstance(results, dict)
        for qid in results:
            assert qid in {"q1", "q2"}
        _assert_valid_csv_or_empty(combo_dir / "results.csv")
        _assert_valid_json_or_empty(combo_dir / "results.json")


# ============== Relaxed validators for rule-based (may find zero matches) ==


def _assert_valid_csv_or_empty(path: Path):
    """Validate CSV structure; rule-based may produce zero matches."""
    assert path.exists()
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert set(reader.fieldnames) == {
        "query_id",
        "source_id",
        "source_text",
        "candidate_score",
        "judgment_score",
    }
    for r in rows:
        float(r["judgment_score"])


def _assert_valid_json_or_empty(path: Path):
    """Validate JSON structure; rule-based may produce zero matches."""
    assert path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    for matches in data.values():
        for m in matches:
            assert "source_id" in m and "judgment_score" in m
