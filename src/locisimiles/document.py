from pathlib import Path
from typing import Dict, Iterator, Any, Union, List
import csv

ID = Union[str, int]

# =================== TEXT SEGMENT ===================


class TextSegment:
    """Atomic unit of text inside a document."""

    def __init__(
        self,
        text: str,
        seg_id: ID,
        *,
        row_id: int | None = None,
        meta: Dict[str, Any] | None = None,
    ):
        self.text: str = text
        self.id: ID = seg_id                  # Unique key in parent document
        self.row_id: int | None = row_id      # Original positional index
        self.meta: Dict[str, Any] = meta or {}

    def __repr__(self) -> str:
        return f"TextSegment(id={self.id!r}, row_id={self.row_id}, len={len(self.text)})"


# =================== DOCUMENT ===================

class Document:
    """Container holding text segments built from a plain-text or CSV file."""

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
        return iter(sorted(self._segments.values(), key=lambda s: s.row_id))

    def __getitem__(self, seg_id: ID) -> TextSegment:
        return self._segments[seg_id]

    def __repr__(self) -> str:
        return f"Document({self.path.name!r}, segments={len(self)})"

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
        if seg_id in self._segments:
            raise ValueError(
                f"Segment id {seg_id!r} already exists in document")
        if row_id is None:
            row_id = len(self._segments)
        self._segments[seg_id] = TextSegment(
            text, seg_id, row_id=row_id, meta=meta)

    def remove_segment(self, seg_id: ID) -> None:
        """Delete a segment if present."""
        self._segments.pop(seg_id, None)

    # ---------- INTERNAL LOADERS ----------
    def _load_plain(self, delimiter: str) -> None:
        """Load from plain-text file split by delimiter."""
        for row_id, seg_text in enumerate(
            self.path.read_text(encoding="utf-8").split(delimiter)
        ):
            if seg_text.strip():
                self.add_segment(seg_text, seg_id=row_id, row_id=row_id)

    def _load_csv(self) -> None:
        """Load from CSV with columns seg_id,text."""
        with self.path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if {"seg_id", "text"} - set(reader.fieldnames or []):
                raise ValueError(
                    "CSV must contain 'seg_id' and 'text' columns")
            for row_id, row in enumerate(reader):
                self.add_segment(row["text"], row["seg_id"], row_id=row_id)


# =================== MAIN DEMO ===================

if __name__ == "__main__":
    doc_txt = Document("../data/hieronymus_samples.txt")
    doc_txt.add_segment("This is a test segment.", "segX")
    print(doc_txt.ids())
    print(doc_txt.get_text("segX"))

    doc_csv = Document("../data/vergil_samples.csv")
    print(len(doc_csv))
    for seg in doc_csv:
        print(seg)

    doc_csv.remove_segment("verg. ecl. 8.75")
    print(len(doc_csv))
