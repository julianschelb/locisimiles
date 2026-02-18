"""
Shared fixtures for locisimiles tests.
"""
import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np

from locisimiles.document import Document, TextSegment


# ============== TEMPORARY FILE FIXTURES ==============

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_csv_file(temp_dir):
    """Create a sample CSV file with seg_id and text columns."""
    csv_path = temp_dir / "sample.csv"
    csv_path.write_text(
        "seg_id,text\n"
        "seg1,This is the first segment.\n"
        "seg2,This is the second segment.\n"
        "seg3,This is the third segment.\n",
        encoding="utf-8"
    )
    return csv_path


@pytest.fixture
def sample_csv_missing_columns(temp_dir):
    """Create a CSV file missing required columns."""
    csv_path = temp_dir / "bad.csv"
    csv_path.write_text(
        "id,content\n"
        "1,Some text\n",
        encoding="utf-8"
    )
    return csv_path


@pytest.fixture
def sample_plain_text_file(temp_dir):
    """Create a sample plain text file."""
    txt_path = temp_dir / "sample.txt"
    txt_path.write_text(
        "First paragraph of text.\n"
        "Second paragraph of text.\n"
        "Third paragraph of text.",
        encoding="utf-8"
    )
    return txt_path


@pytest.fixture
def sample_tsv_file(temp_dir):
    """Create a sample TSV file (uses same CSV format, just different extension)."""
    # Note: Document treats .tsv same as .csv (comma-delimited)
    tsv_path = temp_dir / "sample.tsv"
    tsv_path.write_text(
        "seg_id,text\n"
        "tsv1,First TSV segment.\n"
        "tsv2,Second TSV segment.\n",
        encoding="utf-8"
    )
    return tsv_path


@pytest.fixture
def ground_truth_csv(temp_dir):
    """Create a ground truth CSV file for evaluation."""
    csv_path = temp_dir / "ground_truth.csv"
    csv_path.write_text(
        "query_id,source_id,label\n"
        "q1,s1,1\n"
        "q1,s2,0\n"
        "q1,s3,1\n"
        "q2,s1,0\n"
        "q2,s2,1\n"
        "q2,s3,0\n"
        "q3,s1,0\n"
        "q3,s2,0\n"
        "q3,s3,0\n",
        encoding="utf-8"
    )
    return csv_path


# ============== DOCUMENT FIXTURES ==============

@pytest.fixture
def empty_document(temp_dir):
    """Create an empty document (non-existent path)."""
    return Document(temp_dir / "nonexistent.txt")


@pytest.fixture
def sample_document(sample_csv_file):
    """Create a document loaded from CSV."""
    return Document(sample_csv_file)


@pytest.fixture
def query_document(temp_dir):
    """Create a query document for pipeline tests."""
    csv_path = temp_dir / "query.csv"
    csv_path.write_text(
        "seg_id,text\n"
        "q1,Arma virumque cano Troiae qui primus ab oris.\n"
        "q2,Italiam fato profugus Laviniaque venit.\n"
        "q3,Litora multum ille et terris iactatus et alto.\n",
        encoding="utf-8"
    )
    return Document(csv_path)


@pytest.fixture
def source_document(temp_dir):
    """Create a source document for pipeline tests."""
    csv_path = temp_dir / "source.csv"
    csv_path.write_text(
        "seg_id,text\n"
        "s1,Arma virumque cano qui primus Troiae.\n"
        "s2,Fato profugus Italiam venit.\n"
        "s3,Multum terris iactatus et alto litora.\n"
        "s4,Completely unrelated text here.\n"
        "s5,Another unrelated segment.\n",
        encoding="utf-8"
    )
    return Document(csv_path)


# ============== TEXT SEGMENT FIXTURES ==============

@pytest.fixture
def sample_segment():
    """Create a sample TextSegment."""
    return TextSegment(
        text="Sample text content",
        seg_id="test_seg_1",
        row_id=0,
        meta={"source": "test"}
    )


@pytest.fixture
def sample_segments():
    """Create a list of sample TextSegments."""
    return [
        TextSegment("First segment", "seg1", row_id=0),
        TextSegment("Second segment", "seg2", row_id=1),
        TextSegment("Third segment", "seg3", row_id=2),
    ]


# ============== MOCK FIXTURES FOR ML MODELS ==============

@pytest.fixture
def mock_embedder():
    """Mock SentenceTransformer for embedding tests."""
    mock = MagicMock()
    # Return normalized random embeddings
    def mock_encode(texts, **kwargs):
        embeddings = np.random.randn(len(texts), 384).astype("float32")
        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / norms
    mock.encode.side_effect = mock_encode
    return mock


@pytest.fixture
def mock_classifier():
    """Mock classifier model for classification tests."""
    import torch
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    
    # Mock tokenizer
    mock_tokenizer.num_special_tokens_to_add.return_value = 3
    mock_tokenizer.tokenize.side_effect = lambda x: x.split()[:250]
    mock_tokenizer.convert_tokens_to_string.side_effect = lambda x: " ".join(x)
    
    def mock_call(*args, **kwargs):
        encoding = MagicMock()
        batch_size = len(args[0]) if args else 1
        encoding.__getitem__ = lambda self, key: torch.zeros(batch_size, 512)
        encoding.to = lambda device: encoding
        return encoding
    mock_tokenizer.side_effect = mock_call
    mock_tokenizer.return_value = mock_call()
    
    # Mock model forward pass
    def mock_forward(**kwargs):
        result = MagicMock()
        batch_size = kwargs.get("input_ids", torch.zeros(1)).shape[0] if "input_ids" in kwargs else 1
        result.logits = torch.randn(batch_size, 2)
        return result
    mock_model.side_effect = mock_forward
    mock_model.return_value = mock_forward()
    mock_model.to.return_value = mock_model
    mock_model.eval.return_value = mock_model
    
    return mock_model, mock_tokenizer


@pytest.fixture
def mock_chroma_collection():
    """Mock ChromaDB collection."""
    mock = MagicMock()
    mock.add.return_value = None
    
    def mock_query(query_embeddings, n_results):
        # Return fake results
        return {
            "ids": [["s1", "s2", "s3"][:n_results]],
            "distances": [[0.1, 0.3, 0.5][:n_results]],
        }
    mock.query.side_effect = mock_query
    return mock


# ============== PIPELINE RESULT FIXTURES ==============

@pytest.fixture
def sample_fulldict(sample_segments):
    """Create a sample CandidateJudgeOutput result."""
    from locisimiles.pipeline._types import CandidateJudge
    return {
        "q1": [
            CandidateJudge(segment=sample_segments[0], candidate_score=0.95, judgment_score=0.85),
            CandidateJudge(segment=sample_segments[1], candidate_score=0.75, judgment_score=0.45),
            CandidateJudge(segment=sample_segments[2], candidate_score=0.55, judgment_score=0.25),
        ],
        "q2": [
            CandidateJudge(segment=sample_segments[1], candidate_score=0.88, judgment_score=0.72),
            CandidateJudge(segment=sample_segments[0], candidate_score=0.65, judgment_score=0.38),
            CandidateJudge(segment=sample_segments[2], candidate_score=0.45, judgment_score=0.15),
        ],
    }


@pytest.fixture
def sample_simdict(sample_segments):
    """Create a sample CandidateGeneratorOutput result."""
    from locisimiles.pipeline._types import Candidate
    return {
        "q1": [
            Candidate(segment=sample_segments[0], score=0.95),
            Candidate(segment=sample_segments[1], score=0.75),
            Candidate(segment=sample_segments[2], score=0.55),
        ],
        "q2": [
            Candidate(segment=sample_segments[1], score=0.88),
            Candidate(segment=sample_segments[0], score=0.65),
            Candidate(segment=sample_segments[2], score=0.45),
        ],
    }
