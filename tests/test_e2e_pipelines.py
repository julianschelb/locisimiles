"""
End-to-end tests for all preconfigured pipelines.

Each test creates temporary CSV input files, loads them as Documents,
runs the pipeline (with mocked ML models where needed), and saves the
results to both CSV and JSON — verifying the full read → run → save flow.
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
from locisimiles.pipeline._types import CandidateJudge, CandidateJudgeOutput


# ============== Shared helpers ==============


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
def e2e_dir():
    """Provide a fresh temporary directory for every test."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def query_csv(e2e_dir):
    """Write a small query CSV and return its path."""
    p = e2e_dir / "query.csv"
    p.write_text(QUERY_CSV, encoding="utf-8")
    return p


@pytest.fixture
def source_csv(e2e_dir):
    """Write a small source CSV and return its path."""
    p = e2e_dir / "source.csv"
    p.write_text(SOURCE_CSV, encoding="utf-8")
    return p


# ---------- mock builders ----------


def _mock_sentence_transformer():
    """Return a mock SentenceTransformer whose encode() gives unit vectors."""
    mock = MagicMock()

    def _encode(texts, **_kw):
        rng = np.random.default_rng(42)
        vecs = rng.standard_normal((len(texts), 384)).astype("float32")
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / norms

    mock.encode.side_effect = _encode
    return mock


class _FakeEncoding(dict):
    """Dict-like object that mimics a HuggingFace BatchEncoding."""

    def to(self, device):
        return self


def _mock_classifier():
    """Return (model, tokenizer) mocks for a sequence-classification model."""
    model = MagicMock()
    tokenizer = MagicMock()

    # Tokenizer behaviour (used by _truncate_pair)
    tokenizer.num_special_tokens_to_add.return_value = 3
    tokenizer.tokenize.side_effect = lambda x: x.split()[:250]
    tokenizer.convert_tokens_to_string.side_effect = lambda x: " ".join(x)

    def _tok_call(*args, **kwargs):
        # args[0] is a list of query texts, args[1] is a list of cand texts
        batch_size = len(args[0]) if args and isinstance(args[0], list) else 1
        enc = _FakeEncoding(
            input_ids=torch.zeros(batch_size, 64, dtype=torch.long),
            attention_mask=torch.ones(batch_size, 64, dtype=torch.long),
        )
        return enc

    tokenizer.side_effect = _tok_call
    tokenizer.return_value = _tok_call(["dummy"], ["dummy"])

    # Model forward pass → random logits
    def _forward(**kwargs):
        ids = kwargs.get("input_ids", torch.zeros(1, 1))
        batch_size = ids.shape[0]
        result = MagicMock()
        result.logits = torch.randn(batch_size, 2)
        return result

    model.side_effect = _forward
    model.return_value = _forward()
    model.to.return_value = model
    model.eval.return_value = model
    return model, tokenizer


# ---------- result validators ----------


def _assert_valid_csv(path: Path, expected_query_ids: set[str]):
    """Check that the CSV file has the right columns and expected queries."""
    assert path.exists(), f"CSV not found: {path}"
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) > 0, "CSV is empty"
    assert set(reader.fieldnames) == {
        "query_id", "source_id", "source_text",
        "candidate_score", "judgment_score",
    }
    found_qids = {r["query_id"] for r in rows}
    assert found_qids == expected_query_ids, (
        f"Expected query IDs {expected_query_ids}, got {found_qids}"
    )
    # Every judgment_score should be a parseable float
    for r in rows:
        float(r["judgment_score"])


def _assert_valid_json(path: Path, expected_query_ids: set[str]):
    """Check that the JSON file has the right structure and expected queries."""
    assert path.exists(), f"JSON not found: {path}"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    assert set(data.keys()) == expected_query_ids
    for qid, matches in data.items():
        assert isinstance(matches, list)
        for m in matches:
            assert "source_id" in m
            assert "source_text" in m
            assert "judgment_score" in m
            assert isinstance(m["judgment_score"], (int, float))


# ============== TwoStagePipeline (ClassificationPipelineWithCandidategeneration) ==============


class TestTwoStagePipelineE2E:
    """End-to-end: read CSVs → run → save CSV & JSON."""

    @patch("locisimiles.pipeline.generator.embedding.SentenceTransformer")
    @patch("locisimiles.pipeline.judge.classification.AutoModelForSequenceClassification")
    @patch("locisimiles.pipeline.judge.classification.AutoTokenizer")
    def test_run_and_save(self, mock_tok_cls, mock_mdl_cls, mock_st_cls,
                          query_csv, source_csv, e2e_dir):
        from locisimiles.pipeline.two_stage import TwoStagePipeline

        # Wire mocks
        mock_st_cls.return_value = _mock_sentence_transformer()
        model, tokenizer = _mock_classifier()
        mock_mdl_cls.from_pretrained.return_value = model
        mock_tok_cls.from_pretrained.return_value = tokenizer

        # Load documents from temp files
        query = Document(query_csv)
        source = Document(source_csv)
        assert len(query) == 2
        assert len(source) == 3

        # Run pipeline
        pipeline = TwoStagePipeline(device="cpu")
        results = pipeline.run(query=query, source=source, top_k=2)

        # Verify results structure
        assert isinstance(results, dict)
        assert set(results.keys()) == {"q1", "q2"}
        for lst in results.values():
            assert all(isinstance(item, CandidateJudge) for item in lst)

        # Save to CSV
        csv_out = e2e_dir / "results.csv"
        pipeline.to_csv(csv_out)
        _assert_valid_csv(csv_out, {"q1", "q2"})

        # Save to JSON
        json_out = e2e_dir / "results.json"
        pipeline.to_json(json_out)
        _assert_valid_json(json_out, {"q1", "q2"})


# ============== ExhaustiveClassificationPipeline (ClassificationPipeline) ==============


class TestExhaustiveClassificationPipelineE2E:
    """End-to-end: read CSVs → run → save CSV & JSON."""

    @patch("locisimiles.pipeline.judge.classification.AutoModelForSequenceClassification")
    @patch("locisimiles.pipeline.judge.classification.AutoTokenizer")
    def test_run_and_save(self, mock_tok_cls, mock_mdl_cls,
                          query_csv, source_csv, e2e_dir):
        from locisimiles.pipeline.classification import ExhaustiveClassificationPipeline

        model, tokenizer = _mock_classifier()
        mock_mdl_cls.from_pretrained.return_value = model
        mock_tok_cls.from_pretrained.return_value = tokenizer

        query = Document(query_csv)
        source = Document(source_csv)

        pipeline = ExhaustiveClassificationPipeline(device="cpu")
        results = pipeline.run(query=query, source=source)

        # Exhaustive: every query paired with every source → 2 queries
        assert set(results.keys()) == {"q1", "q2"}
        for lst in results.values():
            # Each query sees all 3 source segments
            assert len(lst) == 3
            assert all(isinstance(item, CandidateJudge) for item in lst)

        csv_out = e2e_dir / "results.csv"
        pipeline.to_csv(csv_out)
        _assert_valid_csv(csv_out, {"q1", "q2"})

        json_out = e2e_dir / "results.json"
        pipeline.to_json(json_out)
        _assert_valid_json(json_out, {"q1", "q2"})


# ============== RetrievalPipeline ==============


class TestRetrievalPipelineE2E:
    """End-to-end: read CSVs → run → save CSV & JSON."""

    @patch("locisimiles.pipeline.generator.embedding.SentenceTransformer")
    def test_run_and_save(self, mock_st_cls,
                          query_csv, source_csv, e2e_dir):
        from locisimiles.pipeline.retrieval import RetrievalPipeline

        mock_st_cls.return_value = _mock_sentence_transformer()

        query = Document(query_csv)
        source = Document(source_csv)

        pipeline = RetrievalPipeline(device="cpu")
        results = pipeline.run(query=query, source=source, top_k=2)

        assert set(results.keys()) == {"q1", "q2"}
        for lst in results.values():
            assert len(lst) == 2  # top_k=2
            assert all(isinstance(item, CandidateJudge) for item in lst)

        csv_out = e2e_dir / "results.csv"
        pipeline.to_csv(csv_out)
        _assert_valid_csv(csv_out, {"q1", "q2"})

        json_out = e2e_dir / "results.json"
        pipeline.to_json(json_out)
        _assert_valid_json(json_out, {"q1", "q2"})


# ============== RuleBasedPipeline ==============


class TestRuleBasedPipelineE2E:
    """End-to-end: read CSVs → run → save CSV & JSON.

    RuleBasedPipeline requires no ML models, so no mocking is needed.
    """

    def test_run_and_save(self, query_csv, source_csv, e2e_dir):
        from locisimiles.pipeline.rule_based import RuleBasedPipeline

        query = Document(query_csv)
        source = Document(source_csv)

        pipeline = RuleBasedPipeline(
            min_shared_words=1,
            min_complura=2,
            max_distance=5,
        )
        results = pipeline.run(query=query, source=source)

        # Rule-based may or may not find matches depending on lexical overlap,
        # but the structure must be correct.
        assert isinstance(results, dict)
        for qid, lst in results.items():
            assert qid in {"q1", "q2"}
            assert all(isinstance(item, CandidateJudge) for item in lst)

        # Save even when some queries may have zero matches
        csv_out = e2e_dir / "results.csv"
        pipeline.to_csv(csv_out)
        assert csv_out.exists()
        with csv_out.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            assert set(reader.fieldnames) == {
                "query_id", "source_id", "source_text",
                "candidate_score", "judgment_score",
            }

        json_out = e2e_dir / "results.json"
        pipeline.to_json(json_out)
        assert json_out.exists()
        data = json.loads(json_out.read_text(encoding="utf-8"))
        assert isinstance(data, dict)


# ============== Cross-pipeline: saving via explicit results arg ==============


class TestExplicitResultsSave:
    """Verify to_csv / to_json accept an explicit results dict."""

    @patch("locisimiles.pipeline.generator.embedding.SentenceTransformer")
    def test_save_explicit_results(self, mock_st_cls,
                                   query_csv, source_csv, e2e_dir):
        from locisimiles.pipeline.retrieval import RetrievalPipeline

        mock_st_cls.return_value = _mock_sentence_transformer()

        query = Document(query_csv)
        source = Document(source_csv)

        pipeline = RetrievalPipeline(device="cpu")
        results = pipeline.run(query=query, source=source, top_k=2)

        # Pass results explicitly instead of relying on _last_judgments
        csv_out = e2e_dir / "explicit.csv"
        pipeline.to_csv(csv_out, results=results)
        _assert_valid_csv(csv_out, {"q1", "q2"})

        json_out = e2e_dir / "explicit.json"
        pipeline.to_json(json_out, results=results)
        _assert_valid_json(json_out, {"q1", "q2"})
