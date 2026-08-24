"""Tests for the GroundTruth/GroundTruthEntry container."""

from __future__ import annotations

import pandas as pd
import pytest

from locisimiles.ground_truth import GroundTruth, GroundTruthEntry

ROWS = [
    {"query_id": "q1", "source_id": "s1", "label": "cit"},
    {"query_id": "q1", "source_id": "s2", "label": "no_match"},
    {"query_id": "q2", "source_id": "s1", "label": "cf"},
]


class TestGroundTruthConstruction:
    def test_empty_by_default(self):
        gt = GroundTruth()
        assert len(gt) == 0
        assert list(gt) == []

    def test_from_list_of_dicts(self):
        gt = GroundTruth(ROWS)
        assert len(gt) == 3
        assert all(isinstance(entry, GroundTruthEntry) for entry in gt)
        assert gt[0].query_id == "q1"
        assert gt[0].source_id == "s1"
        assert gt[0].label == "cit"

    def test_from_list_of_entries(self):
        entries = [GroundTruthEntry(query_id="q1", source_id="s1", label="cit")]
        gt = GroundTruth(entries)
        assert len(gt) == 1
        assert gt[0].query_id == "q1"
        # Entries are copied, not aliased.
        entries[0].label = "cf"
        assert gt[0].label == "cit"

    def test_from_dataframe(self):
        df = pd.DataFrame(ROWS)
        gt = GroundTruth(df)
        assert len(gt) == 3
        assert {entry.query_id for entry in gt} == {"q1", "q2"}

    def test_from_csv_path(self, temp_dir):
        path = temp_dir / "gt.csv"
        path.write_text("query_id,source_id,label\nq1,s1,cit\nq2,s2,cf\n", encoding="utf-8")
        gt = GroundTruth(path)
        assert len(gt) == 2
        assert gt[0].query_id == "q1"
        assert gt[1].label == "cf"

    def test_from_another_ground_truth(self):
        original = GroundTruth(ROWS)
        copy = GroundTruth(original)
        assert len(copy) == len(original)
        copy.append(GroundTruthEntry(query_id="q3", source_id="s3", label="cf"))
        assert len(original) == 3  # original untouched
        assert len(copy) == 4

    def test_missing_required_columns_raises(self):
        with pytest.raises(ValueError, match="missing"):
            GroundTruth([{"query_id": "q1", "label": "cit"}])

    def test_missing_columns_in_dataframe_raises(self):
        df = pd.DataFrame([{"query_id": "q1", "label": "cit"}])
        with pytest.raises(ValueError, match="missing"):
            GroundTruth(df)

    def test_unsupported_type_raises(self):
        with pytest.raises(TypeError):
            GroundTruth(12345)


class TestGroundTruthContainerProtocol:
    def test_len_and_iter(self):
        gt = GroundTruth(ROWS)
        assert len(gt) == 3
        assert len(list(gt)) == 3

    def test_getitem(self):
        gt = GroundTruth(ROWS)
        assert gt[1].source_id == "s2"

    def test_append(self):
        gt = GroundTruth(ROWS)
        gt.append(GroundTruthEntry(query_id="q3", source_id="s3", label="no_match"))
        assert len(gt) == 4
        assert gt[-1].query_id == "q3"


class TestGroundTruthAdd:
    def test_add_concatenates(self):
        a = GroundTruth(ROWS[:1])
        b = GroundTruth(ROWS[1:])
        combined = a + b
        assert len(combined) == 3
        assert list(combined)[0].source_id == "s1"
        assert list(combined)[-1].label == "cf"

    def test_add_does_not_mutate_operands(self):
        a = GroundTruth(ROWS[:1])
        b = GroundTruth(ROWS[1:])
        _ = a + b
        assert len(a) == 1
        assert len(b) == 2

    def test_add_non_ground_truth_returns_not_implemented(self):
        a = GroundTruth(ROWS)
        with pytest.raises(TypeError):
            a + 5  # noqa: B018


class TestGroundTruthConveniences:
    def test_query_ids(self):
        gt = GroundTruth(ROWS)
        assert gt.query_ids() == {"q1", "q2"}

    def test_source_ids(self):
        gt = GroundTruth(ROWS)
        assert gt.source_ids() == {"s1", "s2"}

    def test_label_counts(self):
        gt = GroundTruth(ROWS)
        assert gt.label_counts() == {"cit": 1, "no_match": 1, "cf": 1}

    def test_filter_single_label(self):
        gt = GroundTruth(ROWS)
        filtered = gt.filter(label="cit")
        assert len(filtered) == 1
        assert filtered[0].source_id == "s1"

    def test_filter_multiple_labels(self):
        gt = GroundTruth(ROWS)
        filtered = gt.filter(label=["cit", "cf"])
        assert len(filtered) == 2
        assert {entry.label for entry in filtered} == {"cit", "cf"}

    def test_filter_none_returns_everything(self):
        gt = GroundTruth(ROWS)
        assert len(gt.filter()) == len(gt)

    def test_filter_does_not_mutate_original(self):
        gt = GroundTruth(ROWS)
        gt.filter(label="cit")
        assert len(gt) == 3


class TestGroundTruthExport:
    def test_to_dataframe_round_trip(self):
        gt = GroundTruth(ROWS)
        df = gt.to_dataframe()
        assert list(df.columns) == ["query_id", "source_id", "label", "meta"]
        assert len(df) == 3
        rebuilt = GroundTruth(df)
        assert len(rebuilt) == 3

    def test_to_csv_round_trip(self, temp_dir):
        gt = GroundTruth(ROWS)
        path = gt.to_csv(temp_dir / "out.csv")
        assert path.exists()
        rebuilt = GroundTruth(path)
        assert len(rebuilt) == 3
        assert {entry.query_id for entry in rebuilt} == {"q1", "q2"}
