"""Tests for the locisimiles_gui package."""

from __future__ import annotations

import csv
import os

import pytest

# ── Skip the entire module if gradio is not installed ──────────────────
gr = pytest.importorskip("gradio", reason="GUI tests require gradio")

from locisimiles_gui.app import build_interface  # noqa: E402
from locisimiles_gui.config_stage import (  # noqa: E402
    PIPELINE_CHOICES,
    PIPELINE_EXHAUSTIVE,
    PIPELINE_RETRIEVAL,
    PIPELINE_RULE_BASED,
    PIPELINE_TWO_STAGE,
    _update_pipeline_visibility,
)
from locisimiles_gui.results_stage import (  # noqa: E402
    _export_results_to_csv,
    _extract_numeric_from_html,
    _format_metric_with_bar,
)
from locisimiles_gui.utils import load_csv_preview, validate_csv  # noqa: E402

# ── Fixtures ───────────────────────────────────────────────────────────


@pytest.fixture()
def valid_csv(tmp_path):
    """Write a minimal valid CSV and return its path."""
    p = tmp_path / "valid.csv"
    p.write_text("seg_id,text\n1,hello world\n2,foo bar\n", encoding="utf-8")
    return str(p)


@pytest.fixture()
def empty_csv(tmp_path):
    """CSV with a header but no data rows."""
    p = tmp_path / "empty.csv"
    p.write_text("seg_id,text\n", encoding="utf-8")
    return str(p)


@pytest.fixture()
def wrong_columns_csv(tmp_path):
    """CSV with incorrect column names."""
    p = tmp_path / "wrong.csv"
    p.write_text("id,content\n1,hello\n", encoding="utf-8")
    return str(p)


@pytest.fixture()
def three_columns_csv(tmp_path):
    """CSV with too many columns."""
    p = tmp_path / "three.csv"
    p.write_text("seg_id,text,extra\n1,hello,world\n", encoding="utf-8")
    return str(p)


# ── validate_csv ───────────────────────────────────────────────────────


class TestValidateCsv:
    """Tests for the CSV validation helper."""

    def test_valid_csv(self, valid_csv):
        ok, msg = validate_csv(valid_csv)
        assert ok is True
        assert "valid" in msg.lower()

    def test_none_path(self):
        ok, msg = validate_csv(None)
        assert ok is False

    def test_empty_file(self, tmp_path):
        p = tmp_path / "blank.csv"
        p.write_text("", encoding="utf-8")
        ok, msg = validate_csv(str(p))
        assert ok is False

    def test_no_data_rows(self, empty_csv):
        ok, msg = validate_csv(empty_csv)
        assert ok is False
        assert "no data" in msg.lower()

    def test_wrong_columns(self, wrong_columns_csv):
        ok, msg = validate_csv(wrong_columns_csv)
        assert ok is False
        assert "expected" in msg.lower()

    def test_wrong_column_count(self, three_columns_csv):
        ok, msg = validate_csv(three_columns_csv)
        assert ok is False
        assert "2 columns" in msg.lower()

    def test_nonexistent_file(self):
        ok, msg = validate_csv("/tmp/does_not_exist_12345.csv")
        assert ok is False
        assert "error" in msg.lower()


# ── load_csv_preview ───────────────────────────────────────────────────


class TestLoadCsvPreview:
    """Tests for the CSV preview loader."""

    def test_returns_visible_dict_for_valid_csv(self, valid_csv):
        result = load_csv_preview(valid_csv)
        # Gradio update dicts have __type__ == "update" and a visible key
        assert result["visible"] is True

    def test_returns_hidden_for_none(self):
        result = load_csv_preview(None)
        assert result["visible"] is False

    def test_returns_hidden_for_empty(self, empty_csv):
        result = load_csv_preview(empty_csv)
        assert result["visible"] is False

    def test_max_rows_limits_output(self, tmp_path):
        p = tmp_path / "many.csv"
        lines = ["seg_id,text"] + [f"{i},row {i}" for i in range(50)]
        p.write_text("\n".join(lines), encoding="utf-8")
        result = load_csv_preview(str(p), max_rows=5)
        assert result["visible"] is True
        assert len(result["value"]) == 5


# ── _extract_numeric_from_html ─────────────────────────────────────────


class TestExtractNumericFromHtml:
    """Tests for HTML metric value extraction."""

    def test_extracts_from_span(self):
        html = '<span style="font-weight: bold;">0.789</span>'
        assert _extract_numeric_from_html(html) == pytest.approx(0.789)

    def test_plain_number_string(self):
        assert _extract_numeric_from_html("0.5") == pytest.approx(0.5)

    def test_non_numeric_returns_zero(self):
        assert _extract_numeric_from_html("not a number") == 0.0


# ── _format_metric_with_bar ───────────────────────────────────────────


class TestFormatMetricWithBar:
    """Tests for the metric progress bar formatter."""

    def test_returns_html_string(self):
        html = _format_metric_with_bar(0.75)
        assert "<div" in html
        assert "0.750" in html

    def test_above_threshold_uses_blue(self):
        html = _format_metric_with_bar(0.9, is_above_threshold=True)
        assert "#6B9BD1" in html

    def test_below_threshold_uses_gray(self):
        html = _format_metric_with_bar(0.3, is_above_threshold=False)
        assert "#B0B0B0" in html


# ── _export_results_to_csv ────────────────────────────────────────────


class TestExportResultsToCsv:
    """Tests for CSV export of results."""

    def test_export_creates_valid_csv(self):
        query_segments = [["q1", "query text"]]
        matches_dict = {
            "q1": [["s1", "source text", "0.85", "0.92"]],
        }
        path = _export_results_to_csv(query_segments, matches_dict, threshold=0.5)

        assert os.path.isfile(path)
        with open(path, encoding="utf-8") as f:
            rows = list(csv.reader(f))
        # header + 1 data row
        assert len(rows) == 2
        assert rows[0][0] == "Query_Segment_ID"
        assert rows[1][0] == "q1"
        assert rows[1][6] == "Yes"  # above threshold
        os.unlink(path)

    def test_export_with_no_matches(self):
        query_segments = [["q1", "lonely query"]]
        matches_dict = {}
        path = _export_results_to_csv(query_segments, matches_dict, threshold=0.5)

        with open(path, encoding="utf-8") as f:
            rows = list(csv.reader(f))
        assert len(rows) == 2  # header + 1 empty-match row
        assert rows[1][2] == ""  # no source id
        os.unlink(path)

    def test_export_below_threshold(self):
        query_segments = [["q1", "query text"]]
        matches_dict = {
            "q1": [["s1", "source text", "0.3", "0.2"]],
        }
        path = _export_results_to_csv(query_segments, matches_dict, threshold=0.5)

        with open(path, encoding="utf-8") as f:
            rows = list(csv.reader(f))
        assert rows[1][6] == "No"  # below threshold
        os.unlink(path)


# ── build_interface ───────────────────────────────────────────────────


class TestBuildInterface:
    """Smoke tests for the Gradio interface construction."""

    def test_build_returns_blocks(self):
        demo = build_interface()
        assert isinstance(demo, gr.Blocks)

    def test_interface_has_title(self):
        demo = build_interface()
        assert demo.title == "Loci Similes Demo"


# ── Pipeline visibility toggling ──────────────────────────────────────


class TestPipelineVisibility:
    """Tests for pipeline-type dependent parameter visibility."""

    def test_all_pipeline_choices_are_handled(self):
        """Every pipeline choice should produce a 5-tuple without errors."""
        for choice in PIPELINE_CHOICES:
            result = _update_pipeline_visibility(choice)
            assert len(result) == 5

    def test_two_stage_shows_both_models(self):
        desc, emb, cls, retr, rb = _update_pipeline_visibility(PIPELINE_TWO_STAGE)
        assert emb["visible"] is True
        assert cls["visible"] is True
        assert retr["visible"] is True
        assert rb["visible"] is False

    def test_exhaustive_shows_classification_only(self):
        desc, emb, cls, retr, rb = _update_pipeline_visibility(PIPELINE_EXHAUSTIVE)
        assert emb["visible"] is False
        assert cls["visible"] is True
        assert retr["visible"] is False
        assert rb["visible"] is False

    def test_retrieval_shows_embedding_only(self):
        desc, emb, cls, retr, rb = _update_pipeline_visibility(PIPELINE_RETRIEVAL)
        assert emb["visible"] is True
        assert cls["visible"] is False
        assert retr["visible"] is True
        assert rb["visible"] is False

    def test_rule_based_shows_rule_params(self):
        desc, emb, cls, retr, rb = _update_pipeline_visibility(PIPELINE_RULE_BASED)
        assert emb["visible"] is False
        assert cls["visible"] is False
        assert retr["visible"] is False
        assert rb["visible"] is True
