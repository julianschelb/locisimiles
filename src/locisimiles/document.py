import csv
from pathlib import Path
from typing import Any, Dict, Iterator, List, Union

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
