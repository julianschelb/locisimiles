# ground_truth.py
"""Ground-truth labels for query/source segment pairs.

Purpose-built counterpart to :class:`~locisimiles.document.Document`: where a
``Document`` is a collection of ``TextSegment``s (one corpus), ``GroundTruth``
is a collection of ``GroundTruthEntry`` rows — labeled relationships
*between* two corpora, referenced by segment id rather than duplicating text.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Sequence, Set, Union

import numpy as np
import pandas as pd

from locisimiles.document import ID

LabelValue = Union[int, str]

REQUIRED_COLUMNS = {"query_id", "source_id", "label"}


# =============================================================================
# Data model
# =============================================================================


@dataclass
class GroundTruthEntry:
    """One labeled relationship between a query segment and a source segment.

    Attributes:
        query_id: Id of the query segment (matches a ``Document`` segment id).
        source_id: Id of the source segment (matches a ``Document`` segment id).
        label: Class label for this pair (e.g. ``"no_match"``, ``"cit"``,
            ``"cf"``, or an integer class id).
        meta: Optional additional metadata for this entry.
    """

    query_id: ID
    source_id: ID
    label: LabelValue
    meta: Dict[str, Any] = field(default_factory=dict)


def _native(value: Any) -> Any:
    """Coerce numpy scalar types (from a DataFrame) to plain Python types."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


# =============================================================================
# GroundTruth
# =============================================================================


class GroundTruth:
    """Collection of labeled query/source segment pairs.

    Can be constructed from a CSV/TSV path, a ``pandas.DataFrame``, a list of
    dicts (or ``GroundTruthEntry`` objects), or another ``GroundTruth``.
    Required columns/keys: ``query_id``, ``source_id``, ``label``. An
    optional ``meta`` column/key is preserved per row.

    Example:
        ```python
        from locisimiles.ground_truth import GroundTruth

        gt = GroundTruth("labels.csv")
        gt = GroundTruth([
            {"query_id": "q1", "source_id": "s1", "label": "cit"},
            {"query_id": "q2", "source_id": "s2", "label": "cf"},
        ])

        # Concatenate positives with sampled negatives
        combined = positives + negatives
        ```
    """

    def __init__(
        self,
        source: Union[
            str,
            Path,
            pd.DataFrame,
            Sequence[Mapping[str, Any]],
            Sequence[GroundTruthEntry],
            GroundTruth,
            None,
        ] = None,
    ):
        self._entries: List[GroundTruthEntry] = self._load(source) if source is not None else []

    # ---------- Loading ----------

    @staticmethod
    def _coerce_entry(row: Mapping[str, Any]) -> GroundTruthEntry:
        """Build one entry from a dict-like row, validating required columns."""
        # every row must carry query_id/source_id/label; meta is optional
        missing = REQUIRED_COLUMNS - set(row.keys())
        if missing:
            raise ValueError(f"Ground-truth row is missing required columns: {sorted(missing)}")
        meta = row.get("meta")
        if not isinstance(meta, dict):
            meta = {}
        return GroundTruthEntry(
            query_id=_native(row["query_id"]),
            source_id=_native(row["source_id"]),
            label=_native(row["label"]),
            meta=dict(meta),
        )

    @classmethod
    def _load(
        cls,
        source: Union[
            str,
            Path,
            pd.DataFrame,
            Sequence[Mapping[str, Any]],
            Sequence[GroundTruthEntry],
            GroundTruth,
        ],
    ) -> List[GroundTruthEntry]:
        """Dispatch on source type and return the loaded entries."""
        # another GroundTruth: copy its entries rather than alias them
        if isinstance(source, GroundTruth):
            return [
                GroundTruthEntry(entry.query_id, entry.source_id, entry.label, dict(entry.meta))
                for entry in source
            ]

        # in-memory table
        if isinstance(source, pd.DataFrame):
            missing = REQUIRED_COLUMNS - set(source.columns)
            if missing:
                raise ValueError(f"Ground-truth DataFrame is missing columns: {sorted(missing)}")
            return [cls._coerce_entry(row) for row in source.to_dict(orient="records")]

        # CSV/TSV path
        if isinstance(source, (str, Path)):
            df = pd.read_csv(source)
            missing = REQUIRED_COLUMNS - set(df.columns)
            if missing:
                raise ValueError(f"Ground-truth CSV is missing columns: {sorted(missing)}")
            return [cls._coerce_entry(row) for row in df.to_dict(orient="records")]

        # list of dicts or GroundTruthEntry objects
        if isinstance(source, Iterable):
            entries: List[GroundTruthEntry] = []
            for item in source:
                if isinstance(item, GroundTruthEntry):
                    entries.append(
                        GroundTruthEntry(item.query_id, item.source_id, item.label, dict(item.meta))
                    )
                elif isinstance(item, Mapping):
                    entries.append(cls._coerce_entry(item))
                else:
                    raise TypeError(f"Unsupported ground-truth row type: {type(item)!r}")
            return entries
        raise TypeError(f"Unsupported GroundTruth source type: {type(source)!r}")

    # ---------- Container protocol ----------

    def __iter__(self) -> Iterator[GroundTruthEntry]:
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, index: int) -> GroundTruthEntry:
        return self._entries[index]

    def __repr__(self) -> str:
        return f"GroundTruth(entries={len(self)})"

    def __add__(self, other: GroundTruth) -> GroundTruth:
        """Concatenate two ``GroundTruth`` tables (e.g. positives + sampled negatives)."""
        if not isinstance(other, GroundTruth):
            return NotImplemented
        combined = GroundTruth()
        combined._entries = list(self._entries) + list(other._entries)
        return combined

    # ---------- Conveniences ----------

    def append(self, entry: GroundTruthEntry) -> None:
        """Append one entry in place."""
        self._entries.append(entry)

    def query_ids(self) -> Set[ID]:
        """Return the set of distinct query ids present."""
        return {entry.query_id for entry in self._entries}

    def source_ids(self) -> Set[ID]:
        """Return the set of distinct source ids present."""
        return {entry.source_id for entry in self._entries}

    def label_counts(self) -> Dict[LabelValue, int]:
        """Return a mapping from label to number of entries with that label."""
        counts: Dict[LabelValue, int] = {}
        for entry in self._entries:
            counts[entry.label] = counts.get(entry.label, 0) + 1
        return counts

    def filter(self, *, label: Union[LabelValue, Sequence[LabelValue], None] = None) -> GroundTruth:
        """Return a new ``GroundTruth`` restricted to the given label(s)."""
        if label is None:
            labels: Set[LabelValue] | None = None
        elif isinstance(label, (list, tuple, set, frozenset)):
            labels = set(label)
        else:
            labels = {label}  # type: ignore[arg-type]
        filtered = GroundTruth()
        filtered._entries = [
            entry for entry in self._entries if labels is None or entry.label in labels
        ]
        return filtered

    # ---------- Export ----------

    def to_dataframe(self) -> pd.DataFrame:
        """Return entries as a pandas DataFrame with ``query_id``/``source_id``/``label``/``meta``."""
        return pd.DataFrame(
            [
                {
                    "query_id": entry.query_id,
                    "source_id": entry.source_id,
                    "label": entry.label,
                    "meta": dict(entry.meta),
                }
                for entry in self._entries
            ],
            columns=["query_id", "source_id", "label", "meta"],
        )

    def to_csv(self, path: Union[str, Path]) -> Path:
        """Write entries to a CSV file with ``query_id``, ``source_id``, ``label`` columns."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["query_id", "source_id", "label"])
            writer.writeheader()
            for entry in self._entries:
                writer.writerow(
                    {
                        "query_id": entry.query_id,
                        "source_id": entry.source_id,
                        "label": entry.label,
                    }
                )
        return path
