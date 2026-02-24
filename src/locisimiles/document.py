import csv
import re
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Union

ID = Union[str, int]

# =================== TEXT SEGMENT ===================


class TextSegment:
    """
    Atomic unit of text inside a document.

    A TextSegment represents a single passage, sentence, or verse from a larger
    document. Each segment has a unique identifier and optional metadata.

    Attributes:
        text: The raw text content of the segment.
        id: Unique identifier for the segment (e.g., "verg. aen. 1.1").
        row_id: Position of the segment in the original document (0-indexed).
        meta: Optional dictionary of additional metadata.

    Example:
        ```python
        segment = TextSegment(
            text="Arma virumque cano, Troiae qui primus ab oris",
            seg_id="verg. aen. 1.1",
            row_id=0,
            meta={"book": 1, "line": 1}
        )
        print(segment.text)  # "Arma virumque cano..."
        print(segment.id)    # "verg. aen. 1.1"
        ```
    """

    def __init__(
        self,
        text: str,
        seg_id: ID,
        *,
        row_id: int | None = None,
        meta: Dict[str, Any] | None = None,
    ):
        self.text: str = text
        self.id: ID = seg_id
        self.row_id: int | None = row_id
        self.meta: Dict[str, Any] = meta or {}

    def __repr__(self) -> str:
        return f"TextSegment(id={self.id!r}, row_id={self.row_id}, len={len(self.text)})"


# =================== DOCUMENT ===================


class Document:
    """
    Collection of text segments representing a document.

    A Document is a container for TextSegments loaded from a file. It supports
    CSV/TSV files with 'seg_id' and 'text' columns, or plain text files where
    segments are separated by a delimiter.

    Attributes:
        path: Path to the source file.
        author: Optional author name for the document.
        meta: Optional dictionary of document-level metadata.

    Example:
        ```python
        from locisimiles.document import Document

        # Load from CSV (must have 'seg_id' and 'text' columns)
        vergil = Document("vergil_samples.csv", author="Vergil")

        # Access segments
        print(len(vergil))           # Number of segments
        print(vergil.ids())          # List of segment IDs
        print(vergil.get_text("verg. aen. 1.1"))  # Get text by ID

        # Iterate over segments
        for segment in vergil:
            print(f"{segment.id}: {segment.text[:50]}...")

        # Add custom segments
        vergil.add_segment(
            text="Custom text",
            seg_id="custom.1",
            meta={"source": "manual"}
        )
        ```
    """

    def __init__(
        self,
        path: str | Path,
        *,
        author: str | None = None,
        meta: Dict[str, Any] | None = None,
        segment_delimiter: str = "\n",
    ):
        self.path: Path = Path(path)
        self.author: str | None = author
        self.meta: Dict[str, Any] = meta or {}
        self._segments: Dict[ID, TextSegment] = {}

        if self.path.exists():
            if self.path.suffix.lower() in {".csv", ".tsv"}:
                self._load_csv()
            else:
                self._load_plain(segment_delimiter)

    # ---------- DUNDER HELPERS ----------

    def __len__(self) -> int:
        return len(self._segments)

    def __iter__(self) -> Iterator[TextSegment]:
        return iter(sorted(self._segments.values(), key=lambda s: s.row_id or 0))

    def __getitem__(self, seg_id: ID) -> TextSegment:
        return self._segments[seg_id]

    def __repr__(self) -> str:
        return (
            f"Document({self.path.name!r}, segments={len(self)}, "
            f"author={self.author!r}, meta={self.meta})"
        )

    # ---------- CONVENIENCE ----------

    def ids(self) -> List[ID]:
        """Return segment IDs in original order."""
        return [s.id for s in self]

    def get_text(self, seg_id: ID) -> str:
        """Return raw text of a segment."""
        return self._segments[seg_id].text

    # ---------- PUBLIC API ----------
    @property
    def segments(self) -> Dict[ID, TextSegment]:
        return self._segments

    def add_segment(
        self,
        text: str,
        seg_id: ID,
        *,
        row_id: int | None = None,
        meta: Dict[str, Any] | None = None,
    ) -> None:
        """Add a new text segment to the document."""
        if seg_id in self._segments:
            raise ValueError(f"Segment id {seg_id!r} already exists in document")
        if row_id is None:
            row_id = len(self._segments)
        self._segments[seg_id] = TextSegment(text, seg_id, row_id=row_id, meta=meta)

    def remove_segment(self, seg_id: ID) -> None:
        """Delete a segment if present."""
        self._segments.pop(seg_id, None)

    # ---------- SENTENCIZATION ----------

    # Type alias for the offset map entries used during sentencization.
    # Each tuple is (start_char, end_char, original_segment).
    _OffsetEntry = tuple[int, int, "TextSegment"]

    @staticmethod
    def _default_sentence_splitter(text: str) -> List[str]:
        """Split *text* into sentences using punctuation heuristics.

        Handles common Latin (and general) sentence-ending punctuation.
        For higher accuracy pass a custom *splitter* (e.g. spaCy) to
        :meth:`sentencize`.
        """
        parts = re.split(r"(?<=[.!?;])\s+", text)
        return [p.strip() for p in parts if p.strip()]

    # -- private helpers used by sentencize() --

    def _join_segments(
        self,
        separator: str = " ",
    ) -> tuple[str, List["Document._OffsetEntry"]]:
        """Join all segment texts into one string and build a char-offset map.

        Args:
            separator: String inserted between consecutive segments.

        Returns:
            A tuple of (*full_text*, *offset_map*) where each entry in
            *offset_map* is ``(start_char, end_char, original_segment)``.
        """
        texts: List[str] = []
        offset_map: List[Document._OffsetEntry] = []
        cursor = 0

        # Iterate segments in original order, appending text and recording offsets.
        for seg in self:
            if texts:
                cursor += len(separator)
            start = cursor
            cursor += len(seg.text)
            texts.append(seg.text)
            offset_map.append((start, cursor, seg))

        full_text = separator.join(texts)
        return full_text, offset_map

    @staticmethod
    def _find_origin_segment(
        sent_start: int,
        offset_map: List["Document._OffsetEntry"],
    ) -> "TextSegment":
        """Return the original segment whose span contains *sent_start*."""
        for seg_start, seg_end, seg in offset_map:
            if seg_start <= sent_start < seg_end:
                return seg
        return offset_map[-1][2]  # fallback to last segment

    @staticmethod
    def _map_sentences_to_segments(
        sentences: List[str],
        full_text: str,
        offset_map: List["Document._OffsetEntry"],
        id_separator: str,
    ) -> tuple[Dict[ID, "TextSegment"], Dict[ID, int]]:
        """Create new ``TextSegment`` objects from sentences, mapping each
        back to its originating segment for ID derivation.

        Returns:
            A tuple of (*new_segments*, *origin_counts*) where
            *origin_counts* records how many sentences each original
            segment produced.
        """
        new_segments: Dict[ID, TextSegment] = {}
        origin_counts: Dict[ID, int] = {}
        search_start = 0

        for sentence in sentences:
            sent_start = full_text.find(sentence, search_start)
            if sent_start == -1:
                sent_start = search_start
            search_start = sent_start + len(sentence)

            origin_seg = Document._find_origin_segment(sent_start, offset_map)
            origin_counts[origin_seg.id] = origin_counts.get(origin_seg.id, 0) + 1
            count = origin_counts[origin_seg.id]

            meta = dict(origin_seg.meta)
            meta["original_seg_id"] = origin_seg.id
            meta["sentence_index"] = count

            new_id = f"{origin_seg.id}{id_separator}{count}"
            new_segments[new_id] = TextSegment(
                sentence,
                new_id,
                row_id=len(new_segments),
                meta=meta,
            )

        return new_segments, origin_counts

    def _clone_empty(self) -> "Document":
        """Return an empty ``Document`` with the same metadata."""
        new_doc = Document.__new__(Document)
        new_doc.path = self.path
        new_doc.author = self.author
        new_doc.meta = dict(self.meta)
        new_doc._segments = {}
        return new_doc

    # -- public API --

    def sentencize(
        self,
        *,
        splitter: Optional[Callable[[str], List[str]]] = None,
        id_separator: str = ".",
    ) -> "Document":
        """Re-segment this document so that each segment contains exactly one sentence.

        All segment texts are first joined (in row-id order) and then
        sentence-split as a single block.  This correctly handles:

        * Segments containing **multiple sentences** → split into separate
          segments.
        * A single sentence **spanning multiple rows** → merged into one
          segment.

        New segment IDs are derived from the original segment whose text
        *starts* the sentence, with a numeric suffix appended (e.g.
        ``"seg1.1"``, ``"seg1.2"``).

        Args:
            splitter: A callable that takes a ``str`` and returns a list
                of sentence strings.  When ``None`` a simple
                punctuation-based splitter is used.  To use spaCy::

                    import spacy
                    nlp = spacy.load("la_core_web_lg")
                    doc.sentencize(splitter=lambda t: [s.text for s in nlp(t).sents])

            id_separator: Separator inserted between the original
                segment ID and the sentence index when a segment is
                split (e.g. ``"seg1"`` → ``"seg1.1"``, ``"seg1.2"``).

        Returns:
            The modified ``Document`` with one sentence per segment.

        Example:
            ```python
            doc = Document("mixed.csv")
            doc.sentencize()
            ```
        """
        if not list(self):
            return self

        # 1. Join all segment texts and build a character-offset map.
        full_text, offset_map = self._join_segments()

        # 2. Split the joined text into sentences.
        split_fn = splitter or self._default_sentence_splitter
        sentences = split_fn(full_text)
        if not sentences:
            return self

        # 3. Map each sentence back to its originating segment.
        new_segments, _ = self._map_sentences_to_segments(
            sentences, full_text, offset_map, id_separator
        )

        self._segments = new_segments
        return self

    # ---------- INTERNAL LOADERS ----------

    def _load_plain(self, delimiter: str) -> None:
        """Load from plain-text file split by delimiter."""
        for row_id, seg_text in enumerate(self.path.read_text(encoding="utf-8").split(delimiter)):
            if seg_text.strip():
                self.add_segment(seg_text, seg_id=row_id, row_id=row_id)

    def _load_csv(self) -> None:
        """Load from CSV with columns seg_id,text."""
        with self.path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            required_columns = {"seg_id", "text"}
            if not required_columns.issubset(set(reader.fieldnames or [])):
                raise ValueError("CSV must contain 'seg_id' and 'text' columns")
            for row_id, row in enumerate(reader):
                self.add_segment(row["text"], row["seg_id"], row_id=row_id)

    # ---------- EXPORT ----------

    def save_plain(self, path: str | Path, *, delimiter: str = "\n") -> Path:
        """Write all segment texts to a plain-text file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            delimiter.join(seg.text for seg in self),
            encoding="utf-8",
        )
        return path

    def save_csv(self, path: str | Path) -> Path:
        """Write all segments to a CSV file with ``seg_id`` and ``text`` columns."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["seg_id", "text"])
            writer.writeheader()
            for seg in self:
                writer.writerow({"seg_id": seg.id, "text": seg.text})
        return path


# =================== MAIN DEMO ===================

if __name__ == "__main__":
    doc_txt = Document("../data/hieronymus_samples.txt")
    doc_txt.add_segment("This is a test segment.", "segX")
    print(doc_txt.ids())
    print(doc_txt.get_text("segX"))

    doc_csv = Document("../data/vergil_samples.csv", author="Vergil")
    print(doc_csv)
    print(len(doc_csv))
    for seg in doc_csv:
        print(seg)

    doc_csv.remove_segment("verg. ecl. 8.75")
    print(len(doc_csv))
