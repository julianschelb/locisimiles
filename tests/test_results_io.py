"""Tests for pipeline result I/O: to_csv, to_json, results_to_csv, results_to_json."""
import csv
import json

import pytest

from locisimiles.document import TextSegment
from locisimiles.pipeline._types import (
    CandidateJudge,
    CandidateJudgeOutput,
    results_to_csv,
    results_to_json,
)
from locisimiles.pipeline.pipeline import Pipeline
from locisimiles.pipeline.generator._base import CandidateGeneratorBase
from locisimiles.pipeline.judge._base import JudgeBase


# ============== FIXTURES ==============


@pytest.fixture
def sample_results() -> CandidateJudgeOutput:
    """Minimal CandidateJudgeOutput for testing."""
    return {
        "q1": [
            CandidateJudge(
                segment=TextSegment("Arma virumque cano", seg_id="s1", row_id=0),
                candidate_score=0.85,
                judgment_score=0.95,
            ),
            CandidateJudge(
                segment=TextSegment("Italiam fato profugus", seg_id="s2", row_id=1),
                candidate_score=0.60,
                judgment_score=0.30,
            ),
        ],
        "q2": [
            CandidateJudge(
                segment=TextSegment("Litora multum ille", seg_id="s3", row_id=2),
                candidate_score=None,
                judgment_score=0.70,
            ),
        ],
    }


@pytest.fixture
def empty_results() -> CandidateJudgeOutput:
    """Empty results dict."""
    return {}


# ============== results_to_csv ==============


class TestResultsToCsv:
    def test_basic(self, tmp_path, sample_results):
        out = tmp_path / "results.csv"
        results_to_csv(sample_results, out)

        with out.open(newline="", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))

        assert len(reader) == 3
        assert reader[0]["query_id"] == "q1"
        assert reader[0]["source_id"] == "s1"
        assert reader[0]["source_text"] == "Arma virumque cano"
        assert float(reader[0]["candidate_score"]) == pytest.approx(0.85)
        assert float(reader[0]["judgment_score"]) == pytest.approx(0.95)

    def test_none_candidate_score(self, tmp_path, sample_results):
        out = tmp_path / "results.csv"
        results_to_csv(sample_results, out)

        with out.open(newline="", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))

        # q2/s3 has candidate_score=None → empty string
        row = reader[2]
        assert row["query_id"] == "q2"
        assert row["candidate_score"] == ""

    def test_empty_results(self, tmp_path, empty_results):
        out = tmp_path / "results.csv"
        results_to_csv(empty_results, out)

        with out.open(newline="", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))

        assert len(reader) == 0

    def test_string_path(self, tmp_path, sample_results):
        out = str(tmp_path / "results.csv")
        results_to_csv(sample_results, out)
        assert (tmp_path / "results.csv").exists()

    def test_columns_order(self, tmp_path, sample_results):
        out = tmp_path / "results.csv"
        results_to_csv(sample_results, out)

        with out.open(encoding="utf-8") as f:
            header = f.readline().strip()

        assert header == "query_id,source_id,source_text,candidate_score,judgment_score"


# ============== results_to_json ==============


class TestResultsToJson:
    def test_basic(self, tmp_path, sample_results):
        out = tmp_path / "results.json"
        results_to_json(sample_results, out)

        data = json.loads(out.read_text(encoding="utf-8"))

        assert set(data.keys()) == {"q1", "q2"}
        assert len(data["q1"]) == 2
        assert len(data["q2"]) == 1
        assert data["q1"][0]["source_id"] == "s1"
        assert data["q1"][0]["source_text"] == "Arma virumque cano"
        assert data["q1"][0]["candidate_score"] == pytest.approx(0.85)
        assert data["q1"][0]["judgment_score"] == pytest.approx(0.95)

    def test_none_candidate_score(self, tmp_path, sample_results):
        out = tmp_path / "results.json"
        results_to_json(sample_results, out)

        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["q2"][0]["candidate_score"] is None

    def test_empty_results(self, tmp_path, empty_results):
        out = tmp_path / "results.json"
        results_to_json(empty_results, out)

        data = json.loads(out.read_text(encoding="utf-8"))
        assert data == {}

    def test_custom_indent(self, tmp_path, sample_results):
        out = tmp_path / "results.json"
        results_to_json(sample_results, out, indent=4)

        text = out.read_text(encoding="utf-8")
        # With indent=4, lines should start with 4-space indentation
        assert "    " in text

    def test_string_path(self, tmp_path, sample_results):
        out = str(tmp_path / "results.json")
        results_to_json(sample_results, out)
        assert (tmp_path / "results.json").exists()


# ============== Pipeline.to_csv / Pipeline.to_json ==============


class TestPipelineSaveMethods:
    """Test the convenience methods on the Pipeline class."""

    def _make_pipeline(self):
        """Create a Pipeline with mock generator and judge."""
        gen = type("MockGen", (CandidateGeneratorBase,), {
            "generate": lambda self, **kw: {}
        })()
        judge = type("MockJudge", (JudgeBase,), {
            "judge": lambda self, **kw: {}
        })()
        return Pipeline(generator=gen, judge=judge)

    def test_to_csv_with_explicit_results(self, tmp_path, sample_results):
        pipeline = self._make_pipeline()
        out = tmp_path / "results.csv"
        pipeline.to_csv(out, results=sample_results)

        with out.open(newline="", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
        assert len(reader) == 3

    def test_to_json_with_explicit_results(self, tmp_path, sample_results):
        pipeline = self._make_pipeline()
        out = tmp_path / "results.json"
        pipeline.to_json(out, results=sample_results)

        data = json.loads(out.read_text(encoding="utf-8"))
        assert len(data) == 2

    def test_to_csv_uses_last_results(self, tmp_path, sample_results):
        pipeline = self._make_pipeline()
        pipeline._last_judgments = sample_results
        out = tmp_path / "results.csv"
        pipeline.to_csv(out)

        with out.open(newline="", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
        assert len(reader) == 3

    def test_to_json_uses_last_results(self, tmp_path, sample_results):
        pipeline = self._make_pipeline()
        pipeline._last_judgments = sample_results
        out = tmp_path / "results.json"
        pipeline.to_json(out)

        data = json.loads(out.read_text(encoding="utf-8"))
        assert len(data) == 2

    def test_to_csv_raises_without_results(self, tmp_path):
        pipeline = self._make_pipeline()
        with pytest.raises(ValueError, match="No results to save"):
            pipeline.to_csv(tmp_path / "results.csv")

    def test_to_json_raises_without_results(self, tmp_path):
        pipeline = self._make_pipeline()
        with pytest.raises(ValueError, match="No results to save"):
            pipeline.to_json(tmp_path / "results.json")

    def test_explicit_results_override_cached(self, tmp_path, sample_results):
        """Explicit results take precedence over cached ones."""
        pipeline = self._make_pipeline()
        # Cache some different results
        pipeline._last_judgments = {"q_cached": []}
        out = tmp_path / "results.json"
        pipeline.to_json(out, results=sample_results)

        data = json.loads(out.read_text(encoding="utf-8"))
        assert "q1" in data
        assert "q_cached" not in data
