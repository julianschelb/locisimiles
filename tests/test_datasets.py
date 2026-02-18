"""Tests for the built-in example datasets."""

from locisimiles.datasets import (
    load_example_ground_truth,
    load_example_query,
    load_example_source,
)
from locisimiles.document import Document

# ── Query document ──────────────────────────────────────────────


class TestLoadExampleQuery:
    def test_returns_document(self) -> None:
        doc = load_example_query()
        assert isinstance(doc, Document)

    def test_default_author(self) -> None:
        doc = load_example_query()
        assert doc.author == "Hieronymus"

    def test_custom_author(self) -> None:
        doc = load_example_query(author="Custom")
        assert doc.author == "Custom"

    def test_segment_count(self) -> None:
        doc = load_example_query()
        assert len(doc) == 11

    def test_ids_are_strings(self) -> None:
        doc = load_example_query()
        for seg in doc:
            assert isinstance(seg.id, str)
            assert seg.id.startswith("hier.")

    def test_text_not_empty(self) -> None:
        doc = load_example_query()
        for seg in doc:
            assert len(seg.text) > 0


# ── Source document ─────────────────────────────────────────────


class TestLoadExampleSource:
    def test_returns_document(self) -> None:
        doc = load_example_source()
        assert isinstance(doc, Document)

    def test_default_author(self) -> None:
        doc = load_example_source()
        assert doc.author == "Vergil"

    def test_segment_count(self) -> None:
        doc = load_example_source()
        assert len(doc) == 10

    def test_ids_are_strings(self) -> None:
        doc = load_example_source()
        for seg in doc:
            assert isinstance(seg.id, str)
            assert seg.id.startswith("verg.")


# ── Ground truth ────────────────────────────────────────────────


class TestLoadExampleGroundTruth:
    def test_returns_list(self) -> None:
        gt = load_example_ground_truth()
        assert isinstance(gt, list)

    def test_row_count(self) -> None:
        gt = load_example_ground_truth()
        assert len(gt) == 10

    def test_row_keys(self) -> None:
        gt = load_example_ground_truth()
        for row in gt:
            assert set(row.keys()) == {"query_id", "source_id", "label"}

    def test_label_is_int(self) -> None:
        gt = load_example_ground_truth()
        for row in gt:
            assert isinstance(row["label"], int)
            assert row["label"] in (0, 1)

    def test_query_ids_reference_hieronymus(self) -> None:
        gt = load_example_ground_truth()
        for row in gt:
            assert row["query_id"].startswith("hier.")

    def test_source_ids_reference_vergil(self) -> None:
        gt = load_example_ground_truth()
        for row in gt:
            assert row["source_id"].startswith("verg.")


# ── Top-level import ────────────────────────────────────────────


class TestTopLevelImport:
    """Verify the loaders are accessible from ``locisimiles`` directly."""

    def test_import_load_example_query(self) -> None:
        from locisimiles import load_example_query as fn

        assert callable(fn)

    def test_import_load_example_source(self) -> None:
        from locisimiles import load_example_source as fn

        assert callable(fn)

    def test_import_load_example_ground_truth(self) -> None:
        from locisimiles import load_example_ground_truth as fn

        assert callable(fn)
