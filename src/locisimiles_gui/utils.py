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


def load_csv_preview(file_path: str | None, max_rows: int = 10) -> list[list[str]] | None:
    """Load and preview the first few rows of a CSV file."""
    if not file_path:
        return None
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            
            if not header:
                return None
            
            # Load first max_rows (excluding header)
            rows = []
            for i, row in enumerate(reader):
                if i >= max_rows:
                    break
                rows.append(row)
            
            return rows
    except Exception:
        return None


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
