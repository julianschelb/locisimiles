"""Built-in example datasets for quick experimentation.

This module provides ready-to-use documents and ground-truth data so users
can try out the LociSimiles pipelines without supplying their own files.

Example:
    ```python
    from locisimiles import load_example_query, load_example_source, load_example_ground_truth

    query = load_example_query()       # Hieronymus passages
    source = load_example_source()     # Vergil passages
    ground_truth = load_example_ground_truth()  # list of (query_id, source_id, label) dicts
    ```
"""

from __future__ import annotations

import csv
import importlib.resources
from pathlib import Path
from typing import Any, Dict, List

from locisimiles.document import Document

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_EXAMPLES_PACKAGE = "locisimiles.data.examples"


def _example_path(filename: str) -> Path:
    """Return the filesystem path to a bundled example file.

    Uses :mod:`importlib.resources` so the lookup works regardless of how the
    package was installed (editable, wheel, zip, …).
    """
    ref = importlib.resources.files(_EXAMPLES_PACKAGE).joinpath(filename)
    # as_file() handles both filesystem and zip installs
    with importlib.resources.as_file(ref) as p:
        # Return a concrete Path; the context manager keeps it valid for the
        # lifetime of the process when the file is already on disk.
        return Path(p)


# ---------------------------------------------------------------------------
# Public loaders
# ---------------------------------------------------------------------------


def load_example_query(*, author: str | None = "Hieronymus") -> Document:
    """Load the example query document (Hieronymus *adversus Iovinianum*).

    The dataset contains 11 passages from Hieronymus that are suspected of
    echoing Vergil.

    Args:
        author: Optional author label attached to the ``Document``.

    Returns:
        A ``Document`` with the query segments.

    Example:
        ```python
        query = load_example_query()
        for seg in query:
            print(seg.id, seg.text[:60])
        ```
    """
    return Document(_example_path("hieronymus_samples.csv"), author=author)


def load_example_source(*, author: str | None = "Vergil") -> Document:
    """Load the example source document (Vergil).

    The dataset contains 10 passages from Vergil's works that serve as
    potential sources for the query passages.

    Args:
        author: Optional author label attached to the ``Document``.

    Returns:
        A ``Document`` with the source segments.

    Example:
        ```python
        source = load_example_source()
        for seg in source:
            print(seg.id, seg.text[:60])
        ```
    """
    return Document(_example_path("vergil_samples.csv"), author=author)


def load_example_ground_truth() -> List[Dict[str, Any]]:
    """Load the example ground-truth labels.

    Each entry maps a ``query_id`` / ``source_id`` pair to a binary
    ``label`` (``1`` = true intertext, ``0`` = no intertext).

    Returns:
        A list of dictionaries with keys ``query_id``, ``source_id``, and
        ``label`` (int).

    Example:
        ```python
        gt = load_example_ground_truth()
        for row in gt:
            print(row["query_id"], "->", row["source_id"], ":", row["label"])
        ```
    """
    path = _example_path("ground_truth.csv")
    rows: List[Dict[str, Any]] = []
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(
                {
                    "query_id": row["query_id"],
                    "source_id": row["source_id"],
                    "label": int(row["label"]),
                }
            )
    return rows
