"""
Unit tests for locisimiles.document module.
Tests TextSegment and Document classes.
"""

from pathlib import Path

import pytest

from locisimiles.document import Document, TextSegment

# =================== TEXTSEGMENT TESTS ===================


class TestTextSegment:
    """Tests for the TextSegment class."""

    def test_text_segment_creation(self):
        """Test creating a TextSegment with all parameters."""
        segment = TextSegment(text="Hello world", seg_id="seg1", row_id=5, meta={"author": "Test"})
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
            text="New segment text", seg_id="seg4", row_id=3, meta={"new": True}
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
            "seg_id,text\nu1,Κόσμος Greek text\nu2,日本語 Japanese\nu3,Émoji 🎉 text\n",
            encoding="utf-8",
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


# =================== STATISTICS TESTS ===================


class TestStatistics:
    """Tests for Document.statistics() method."""

    @pytest.fixture()
    def temp_dir(self, tmp_path):
        return tmp_path

    def test_statistics_basic(self, temp_dir):
        """statistics() returns correct values for a simple document."""
        csv_path = temp_dir / "stats.csv"
        csv_path.write_text(
            "seg_id,text\ns1,Hello world\ns2,Foo\n",
            encoding="utf-8",
        )
        doc = Document(csv_path)
        stats = doc.statistics()

        assert stats["num_segments"] == 2
        assert stats["total_chars"] == len("Hello world") + len("Foo")
        assert stats["total_words"] == 3  # Hello, world, Foo
        assert stats["min_segment_chars"] == 3  # "Foo"
        assert stats["max_segment_chars"] == 11  # "Hello world"
        assert stats["avg_chars_per_segment"] == round((11 + 3) / 2, 2)
        assert stats["avg_words_per_segment"] == round(3 / 2, 2)

    def test_statistics_empty_document(self, temp_dir):
        """statistics() on an empty document returns all zeros."""
        csv_path = temp_dir / "empty.csv"
        csv_path.write_text("seg_id,text\n", encoding="utf-8")
        doc = Document(csv_path)
        stats = doc.statistics()

        assert stats["num_segments"] == 0
        assert stats["total_chars"] == 0
        assert stats["total_words"] == 0
        assert stats["avg_chars_per_segment"] == 0.0
        assert stats["avg_words_per_segment"] == 0.0
        assert stats["min_segment_chars"] == 0
        assert stats["max_segment_chars"] == 0

    def test_statistics_single_segment(self, temp_dir):
        """statistics() with one segment has min == max == total."""
        csv_path = temp_dir / "one.csv"
        csv_path.write_text("seg_id,text\ns1,One two three\n", encoding="utf-8")
        doc = Document(csv_path)
        stats = doc.statistics()

        assert stats["num_segments"] == 1
        assert stats["min_segment_chars"] == stats["max_segment_chars"]
        assert stats["total_words"] == 3


# =================== SENTENCIZE TESTS ===================


class TestSentencize:
    """Tests for Document.sentencize() method."""

    def test_single_sentence_segments_unchanged(self, temp_dir):
        """Segments that already have one sentence stay as-is."""
        csv_path = temp_dir / "single.csv"
        csv_path.write_text(
            "seg_id,text\ns1,Arma virumque cano.\ns2,Troiae qui primus ab oris.\n",
            encoding="utf-8",
        )
        doc = Document(csv_path)
        doc.sentencize()
        assert len(doc) == 2
        assert "s1.1" in doc.segments
        assert "s2.1" in doc.segments

    def test_multi_sentence_segment_split(self, temp_dir):
        """A segment with multiple sentences is split into separate segments."""
        csv_path = temp_dir / "multi.csv"
        csv_path.write_text(
            "seg_id,text\ns1,First sentence. Second sentence. Third sentence.\n",
            encoding="utf-8",
        )
        doc = Document(csv_path)
        doc.sentencize()
        assert len(doc) == 3
        assert doc["s1.1"].text == "First sentence."
        assert doc["s1.2"].text == "Second sentence."
        assert doc["s1.3"].text == "Third sentence."

    def test_split_preserves_original_id_in_meta(self, temp_dir):
        """Split segments store original_seg_id and sentence_index in meta."""
        csv_path = temp_dir / "meta.csv"
        csv_path.write_text(
            "seg_id,text\nseg1,Hello world. Goodbye world.\n",
            encoding="utf-8",
        )
        doc = Document(csv_path)
        doc.sentencize()
        assert doc["seg1.1"].meta["original_seg_id"] == "seg1"
        assert doc["seg1.1"].meta["sentence_index"] == 1
        assert doc["seg1.2"].meta["sentence_index"] == 2

    def test_mixed_segments(self, temp_dir):
        """Mix of single and multi-sentence segments."""
        csv_path = temp_dir / "mixed.csv"
        csv_path.write_text(
            "seg_id,text\na,Solo sentence.\nb,One. Two. Three.\nc,Another solo.\n",
            encoding="utf-8",
        )
        doc = Document(csv_path)
        doc.sentencize()
        # a stays (suffixed), b splits into 3, c stays (suffixed) → total 5
        assert len(doc) == 5
        assert "a.1" in doc.segments
        assert "b.1" in doc.segments
        assert "b.2" in doc.segments
        assert "b.3" in doc.segments
        assert "c.1" in doc.segments

    def test_row_ids_are_sequential(self, temp_dir):
        """Row IDs should be sequential after sentencization."""
        csv_path = temp_dir / "seq.csv"
        csv_path.write_text(
            "seg_id,text\ns1,A. B.\ns2,C.\n",
            encoding="utf-8",
        )
        doc = Document(csv_path)
        doc.sentencize()
        row_ids = [seg.row_id for seg in doc]
        assert row_ids == list(range(len(doc)))

    def test_custom_id_separator(self, temp_dir):
        """Custom id_separator is used in generated IDs."""
        csv_path = temp_dir / "sep.csv"
        csv_path.write_text(
            "seg_id,text\nx,Foo. Bar.\n",
            encoding="utf-8",
        )
        doc = Document(csv_path)
        doc.sentencize(id_separator="_s")
        assert "x_s1" in doc.segments
        assert "x_s2" in doc.segments

    def test_custom_splitter(self, temp_dir):
        """A custom splitter function is used when provided."""
        csv_path = temp_dir / "custom.csv"
        csv_path.write_text(
            "seg_id,text\ns1,alpha|beta|gamma\n",
            encoding="utf-8",
        )
        doc = Document(csv_path)
        doc.sentencize(splitter=lambda t: t.split("|"))
        assert len(doc) == 3
        assert doc["s1.1"].text == "alpha"
        assert doc["s1.2"].text == "beta"
        assert doc["s1.3"].text == "gamma"

    def test_inplace_modifies_and_returns_self(self, temp_dir):
        """sentencize() modifies and returns the same Document."""
        csv_path = temp_dir / "ip.csv"
        csv_path.write_text(
            "seg_id,text\ns1,A. B.\n",
            encoding="utf-8",
        )
        doc = Document(csv_path)
        result = doc.sentencize()
        assert result is doc
        assert len(doc) == 2

    def test_semicolon_splitting(self, temp_dir):
        """Default splitter handles semicolons as sentence boundaries."""
        csv_path = temp_dir / "semi.csv"
        csv_path.write_text(
            "seg_id,text\ns1,Clause one; clause two; clause three.\n",
            encoding="utf-8",
        )
        doc = Document(csv_path)
        doc.sentencize()
        assert len(doc) == 3

    def test_sentence_spanning_multiple_rows(self, temp_dir):
        """A sentence split across two rows is merged into one segment."""
        csv_path = temp_dir / "cross.csv"
        csv_path.write_text(
            "seg_id,text\n"
            "s1,Arma virumque cano\n"  # no ending punctuation
            "s2,Troiae qui primus ab oris.\n",
            encoding="utf-8",
        )
        doc = Document(csv_path)
        doc.sentencize()
        # Both fragments form a single sentence → 1 segment
        assert len(doc) == 1
        seg = list(doc)[0]
        assert "Arma virumque cano" in seg.text
        assert "Troiae qui primus ab oris." in seg.text

    def test_merge_and_split_combined(self, temp_dir):
        """Rows where fragments merge AND a row with multiple sentences."""
        csv_path = temp_dir / "combo.csv"
        csv_path.write_text(
            "seg_id,text\n"
            "s1,Start of sentence\n"  # fragment → merges with s2
            "s2,end of sentence. New one.\n"  # finishes first sent + starts second
            "s3,Solo sentence.\n",
            encoding="utf-8",
        )
        doc = Document(csv_path)
        doc.sentencize()
        # Sentence 1: "Start of sentence end of sentence." (spans s1+s2)
        # Sentence 2: "New one." (from s2)
        # Sentence 3: "Solo sentence." (s3 alone)
        assert len(doc) == 3
        texts = [seg.text for seg in doc]
        assert any("Start of sentence" in t and "end of sentence." in t for t in texts)
        assert "New one." in texts
        assert "Solo sentence." in texts

    def test_many_fragments_one_sentence(self, temp_dir):
        """Multiple rows that are all fragments of a single sentence."""
        csv_path = temp_dir / "frags.csv"
        csv_path.write_text(
            "seg_id,text\ns1,word1 word2\ns2,word3 word4\ns3,word5 word6.\n",
            encoding="utf-8",
        )
        doc = Document(csv_path)
        doc.sentencize()
        assert len(doc) == 1
        seg = list(doc)[0]
        assert "word1 word2 word3 word4 word5 word6." == seg.text

    def test_empty_document_sentencize(self, temp_dir):
        """Sentencize on an empty document is a no-op."""
        csv_path = temp_dir / "empty.csv"
        csv_path.write_text("seg_id,text\n", encoding="utf-8")
        doc = Document(csv_path)
        result = doc.sentencize()
        assert len(result) == 0
        assert result is doc


# =================== EXPORT TESTS ===================


class TestSaveCSV:
    """Tests for Document.save_csv."""

    @pytest.fixture()
    def temp_dir(self, tmp_path):
        return tmp_path

    def test_save_csv_roundtrip(self, temp_dir):
        """Saving and reloading a CSV preserves segments."""
        src = temp_dir / "src.csv"
        src.write_text("seg_id,text\ns1,Hello.\ns2,World.\n", encoding="utf-8")
        doc = Document(src)

        out = temp_dir / "out.csv"
        returned = doc.save_csv(out)
        assert returned == out
        assert out.exists()

        reloaded = Document(out)
        assert len(reloaded) == 2
        assert reloaded["s1"].text == "Hello."
        assert reloaded["s2"].text == "World."

    def test_save_csv_after_sentencize(self, temp_dir):
        """Exported CSV reflects sentencized segments."""
        src = temp_dir / "src.csv"
        src.write_text("seg_id,text\ns1,One. Two.\n", encoding="utf-8")
        doc = Document(src)
        doc.sentencize()

        out = temp_dir / "sent.csv"
        doc.save_csv(out)
        reloaded = Document(out)
        assert len(reloaded) == 2
        assert reloaded["s1.1"].text == "One."
        assert reloaded["s1.2"].text == "Two."

    def test_save_csv_sentencize_roundtrip(self, temp_dir):
        """Full roundtrip: load → sentencize → save → reload preserves content."""
        src = temp_dir / "mixed.csv"
        src.write_text(
            "seg_id,text\n"
            "a,First sentence. Second sentence.\n"
            "b,Third sentence spanning\n"
            "c,multiple rows.\n"
            "d,Solo.\n",
            encoding="utf-8",
        )

        # Load and sentencize.
        doc = Document(src)
        doc.sentencize()
        sentencized_texts = [seg.text for seg in doc]
        sentencized_ids = [seg.id for seg in doc]

        # Save and reload.
        out = temp_dir / "roundtrip.csv"
        doc.save_csv(out)
        reloaded = Document(out)

        # Reloaded document must match the sentencized version exactly.
        assert len(reloaded) == len(doc)
        assert [seg.id for seg in reloaded] == sentencized_ids
        assert [seg.text for seg in reloaded] == sentencized_texts

    def test_save_csv_creates_parent_dirs(self, temp_dir):
        """Missing parent directories are created automatically."""
        src = temp_dir / "src.csv"
        src.write_text("seg_id,text\na,Hi.\n", encoding="utf-8")
        doc = Document(src)

        out = temp_dir / "sub" / "dir" / "out.csv"
        doc.save_csv(out)
        assert out.exists()

    def test_save_csv_empty_document(self, temp_dir):
        """An empty document produces a CSV with only a header."""
        src = temp_dir / "empty.csv"
        src.write_text("seg_id,text\n", encoding="utf-8")
        doc = Document(src)

        out = temp_dir / "empty_out.csv"
        doc.save_csv(out)
        content = out.read_text(encoding="utf-8")
        assert content.strip() == "seg_id,text"


class TestSavePlain:
    """Tests for Document.save_plain."""

    @pytest.fixture()
    def temp_dir(self, tmp_path):
        return tmp_path

    def test_save_plain_roundtrip(self, temp_dir):
        """Saving and reloading a plain-text file preserves segments."""
        src = temp_dir / "src.txt"
        src.write_text("Line one\nLine two\nLine three", encoding="utf-8")
        doc = Document(src)

        out = temp_dir / "out.txt"
        returned = doc.save_plain(out)
        assert returned == out
        assert out.read_text(encoding="utf-8") == "Line one\nLine two\nLine three"

    def test_save_plain_custom_delimiter(self, temp_dir):
        """A custom delimiter is used between segments."""
        src = temp_dir / "src.csv"
        src.write_text("seg_id,text\na,Alpha\nb,Beta\n", encoding="utf-8")
        doc = Document(src)

        out = temp_dir / "out.txt"
        doc.save_plain(out, delimiter=" | ")
        assert out.read_text(encoding="utf-8") == "Alpha | Beta"

    def test_save_plain_empty_document(self, temp_dir):
        """An empty document produces an empty file."""
        src = temp_dir / "empty.csv"
        src.write_text("seg_id,text\n", encoding="utf-8")
        doc = Document(src)

        out = temp_dir / "empty_out.txt"
        doc.save_plain(out)
        assert out.read_text(encoding="utf-8") == ""
