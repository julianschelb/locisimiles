"""Utility functions for the Gradio GUI."""

from __future__ import annotations

import csv
from pathlib import Path

try:
    import gradio as gr
except ImportError as exc:
    raise ImportError("Gradio is required for GUI utilities") from exc


def validate_csv(file_path: str | None) -> tuple[bool, str]:
    """Validate that a CSV file has the required format with 'seg_id' and 'text' columns.
    
    Returns:
        A tuple of (is_valid, message) where is_valid is True if valid, False otherwise.
    """
    if not file_path:
        return False, "No file provided"
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            
            if not header:
                return False, "Empty file"
            
            # Check if header has exactly 2 columns: seg_id and text
            if len(header) != 2:
                return False, f"Expected 2 columns, found {len(header)}"
            
            if header[0] != "seg_id" or header[1] != "text":
                return False, f"Expected columns 'seg_id' and 'text', found {header}"
            
            # Check if there's at least one data row
            first_row = next(reader, None)
            if not first_row:
                return False, "No data rows found"
            
            return True, "Valid CSV format"
    except Exception as e:
        return False, f"Error reading file: {str(e)}"


def load_csv_preview(file_path: str | None, max_rows: int | None = None) -> dict:
    """Load and preview all rows (or first max_rows) of a CSV file.
    
    Args:
        file_path: Path to the CSV file
        max_rows: Maximum number of rows to load (None = load all rows)
    
    Returns:
        A Gradio update dict with the preview data and visibility set to True if valid,
        or hidden if invalid/empty.
    """
    if not file_path:
        return gr.update(visible=False)
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            
            if not header:
                return gr.update(visible=False)
            
            # Load rows (all or up to max_rows)
            rows = []
            for i, row in enumerate(reader):
                if max_rows is not None and i >= max_rows:
                    break
                rows.append(row)
            
            if not rows:
                return gr.update(visible=False)
            
            return gr.update(value=rows, visible=True, headers=header)
    except Exception:
        return gr.update(visible=False)


def validate_and_notify(file_path: str | None, doc_type: str = "Document") -> str | None:
    """Validate a document on upload and show notification.
    
    Args:
        file_path: Path to the CSV file
        doc_type: Type of document (e.g., "Query document", "Source document")
    
    Returns:
        The file path if valid, None otherwise
    """
    if not file_path:
        return None
    
    is_valid, message = validate_csv(file_path)
    filename = Path(file_path).name
    
    if is_valid:
        gr.Info(f"{doc_type} is valid!")
    else:
        gr.Warning(f"{doc_type} is invalid: {message}")
    
    return file_path
