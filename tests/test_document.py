"""
Unit tests for locisimiles.document module.
Tests TextSegment and Document classes.
"""
import pytest
from pathlib import Path

from locisimiles.document import TextSegment, Document


# =================== TEXTSEGMENT TESTS ===================

class TestTextSegment:
    """Tests for the TextSegment class."""

    def test_text_segment_creation(self):
        """Test creating a TextSegment with all parameters."""
        segment = TextSegment(
            text="Hello world",
            seg_id="seg1",
            row_id=5,
            meta={"author": "Test"}
        )
        assert segment.text == "Hello world"
        assert segment.id == "seg1"
        assert segment.row_id == 5
        assert segment.meta == {"author": "Test"}

    def test_text_segment_minimal(self):
        """Test creating a TextSegment with minimal parameters."""
        segment = TextSegment("Test text", "id1")
        assert segment.text == "Test text"
        assert segment.id == "id1"
        assert segment.row_id is None
        assert segment.meta == {}

    def test_text_segment_int_id(self):
        """Test creating a TextSegment with integer ID."""
        segment = TextSegment("Content", 42, row_id=0)
        assert segment.id == 42
        assert isinstance(segment.id, int)

    def test_text_segment_repr(self):
        """Test TextSegment string representation."""
        segment = TextSegment("Hello world", "seg1", row_id=3)
        repr_str = repr(segment)
        assert "TextSegment" in repr_str
        assert "seg1" in repr_str
        assert "row_id=3" in repr_str
        assert "len=11" in repr_str  # len("Hello world") = 11

    def test_text_segment_empty_text(self):
        """Test TextSegment with empty text."""
        segment = TextSegment("", "empty_seg")
        assert segment.text == ""
        assert len(segment.text) == 0

    def test_text_segment_meta_mutation(self):
        """Test that meta dict can be mutated after creation."""
        segment = TextSegment("Text", "id1", meta={"key": "value"})
        segment.meta["new_key"] = "new_value"
        assert "new_key" in segment.meta


# =================== DOCUMENT TESTS ===================

class TestDocumentCreation:
    """Tests for Document creation and loading."""

    def test_document_from_csv(self, sample_csv_file):
        """Test loading document from CSV file."""
        doc = Document(sample_csv_file)
        assert len(doc) == 3
        assert "seg1" in doc.segments
        assert "seg2" in doc.segments
        assert "seg3" in doc.segments

    def test_document_from_plain_text(self, sample_plain_text_file):
        """Test loading document from plain text file."""
        doc = Document(sample_plain_text_file)
        assert len(doc) == 3
        # Plain text uses integer IDs starting from 0
        assert 0 in doc.segments
        assert 1 in doc.segments
        assert 2 in doc.segments

    def test_document_from_tsv(self, sample_tsv_file):
        """Test loading document from TSV file."""
        doc = Document(sample_tsv_file)
        assert len(doc) == 2
        assert "tsv1" in doc.segments
        assert "tsv2" in doc.segments

    def test_document_csv_missing_columns(self, sample_csv_missing_columns):
        """Test that CSV without required columns raises ValueError."""
        with pytest.raises(ValueError, match="CSV must contain 'seg_id' and 'text' columns"):
            Document(sample_csv_missing_columns)

    def test_document_nonexistent_path(self, temp_dir):
        """Test creating document with non-existent path (empty document)."""
        doc = Document(temp_dir / "nonexistent.txt")
        assert len(doc) == 0
        assert doc.segments == {}

    def test_document_with_author(self, sample_csv_file):
        """Test document creation with author metadata."""
        doc = Document(sample_csv_file, author="Vergil")
        assert doc.author == "Vergil"

    def test_document_with_meta(self, sample_csv_file):
        """Test document creation with custom metadata."""
        doc = Document(sample_csv_file, meta={"year": 19, "genre": "epic"})
        assert doc.meta == {"year": 19, "genre": "epic"}

    def test_document_custom_delimiter(self, temp_dir):
        """Test loading plain text with custom delimiter."""
        txt_path = temp_dir / "custom.txt"
        txt_path.write_text("Part1|||Part2|||Part3", encoding="utf-8")
        doc = Document(txt_path, segment_delimiter="|||")
        assert len(doc) == 3


class TestDocumentDunderMethods:
    """Tests for Document dunder methods."""

    def test_document_len(self, sample_document):
        """Test len(document) returns segment count."""
        assert len(sample_document) == 3

    def test_document_iter(self, sample_document):
        """Test iteration yields segments in row_id order."""
        segments = list(sample_document)
        assert len(segments) == 3
        # Check order by row_id
        for i, seg in enumerate(segments):
            assert seg.row_id == i

    def test_document_getitem(self, sample_document):
        """Test document[seg_id] retrieves correct segment."""
        segment = sample_document["seg1"]
        assert segment.id == "seg1"
        assert "first segment" in segment.text.lower()

    def test_document_getitem_missing(self, sample_document):
        """Test document[missing_id] raises KeyError."""
        with pytest.raises(KeyError):
            _ = sample_document["nonexistent"]

    def test_document_repr(self, sample_csv_file):
        """Test document string representation."""
        doc = Document(sample_csv_file, author="Vergil", meta={"year": 19})
        repr_str = repr(doc)
        assert "Document" in repr_str
        assert "sample.csv" in repr_str
        assert "segments=3" in repr_str
        assert "Vergil" in repr_str


class TestDocumentMethods:
    """Tests for Document public methods."""

    def test_document_ids_order(self, sample_document):
        """Test ids() returns IDs in original (row_id) order."""
        ids = sample_document.ids()
        assert ids == ["seg1", "seg2", "seg3"]

    def test_document_get_text(self, sample_document):
        """Test get_text() returns correct text."""
        text = sample_document.get_text("seg1")
        assert text == "This is the first segment."

    def test_document_get_text_missing(self, sample_document):
        """Test get_text() with missing ID raises KeyError."""
        with pytest.raises(KeyError):
            sample_document.get_text("nonexistent")

    def test_document_segments_property(self, sample_document):
        """Test segments property returns internal dict."""
        segments = sample_document.segments
        assert isinstance(segments, dict)
        assert len(segments) == 3


class TestDocumentAddRemoveSegments:
    """Tests for add_segment and remove_segment methods."""

    def test_document_add_segment(self, sample_document):
        """Test adding a new segment."""
        sample_document.add_segment(
            text="New segment text",
            seg_id="seg4",
            row_id=3,
            meta={"new": True}
        )
        assert len(sample_document) == 4
        assert "seg4" in sample_document.segments
        assert sample_document["seg4"].text == "New segment text"
        assert sample_document["seg4"].meta == {"new": True}

    def test_document_add_segment_auto_row_id(self, sample_document):
        """Test adding segment without row_id auto-assigns one."""
        sample_document.add_segment("No row_id", "seg_auto")
        assert sample_document["seg_auto"].row_id == 3  # Next after 0,1,2

    def test_document_add_duplicate_segment(self, sample_document):
        """Test adding segment with duplicate ID raises ValueError."""
        with pytest.raises(ValueError, match="already exists"):
            sample_document.add_segment("Duplicate", "seg1")

    def test_document_remove_segment(self, sample_document):
        """Test removing an existing segment."""
        assert "seg2" in sample_document.segments
        sample_document.remove_segment("seg2")
        assert "seg2" not in sample_document.segments
        assert len(sample_document) == 2

    def test_document_remove_nonexistent(self, sample_document):
        """Test removing non-existent segment doesn't raise."""
        sample_document.remove_segment("nonexistent")  # Should not raise
        assert len(sample_document) == 3


class TestDocumentEdgeCases:
    """Edge case tests for Document class."""

    def test_document_empty_csv_rows(self, temp_dir):
        """Test CSV with only header (no data rows)."""
        csv_path = temp_dir / "empty.csv"
        csv_path.write_text("seg_id,text\n", encoding="utf-8")
        doc = Document(csv_path)
        assert len(doc) == 0

    def test_document_plain_text_empty_lines(self, temp_dir):
        """Test plain text with empty lines (should be skipped)."""
        txt_path = temp_dir / "sparse.txt"
        txt_path.write_text("Line1\n\n\nLine2\n\n", encoding="utf-8")
        doc = Document(txt_path)
        assert len(doc) == 2  # Empty lines should be skipped

    def test_document_unicode_content(self, temp_dir):
        """Test document with unicode content."""
        csv_path = temp_dir / "unicode.csv"
        csv_path.write_text(
            "seg_id,text\n"
            "u1,Κόσμος Greek text\n"
            "u2,日本語 Japanese\n"
            "u3,Émoji 🎉 text\n",
            encoding="utf-8"
        )
        doc = Document(csv_path)
        assert len(doc) == 3
        assert "Κόσμος" in doc.get_text("u1")
        assert "日本語" in doc.get_text("u2")
        assert "🎉" in doc.get_text("u3")

    def test_document_path_property(self, sample_csv_file):
        """Test that path property returns Path object."""
        doc = Document(sample_csv_file)
        assert isinstance(doc.path, Path)
        assert doc.path.exists()
