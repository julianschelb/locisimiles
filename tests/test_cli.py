"""
Unit tests for locisimiles.cli module.
Tests command-line interface functionality.
"""
import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
from io import StringIO

from locisimiles.pipeline._types import CandidateJudge


class TestCLIArgumentParsing:
    """Tests for CLI argument parsing."""

    def test_cli_help_output(self, capsys):
        """Test that help text displays correctly."""
        from locisimiles.cli import main
        
        with patch.object(sys, 'argv', ['locisimiles', '--help']):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0
        
        captured = capsys.readouterr()
        assert "locisimiles" in captured.out
        assert "query" in captured.out
        assert "source" in captured.out
        assert "--output" in captured.out

    def test_cli_missing_required_args(self, capsys):
        """Test error when missing required arguments."""
        from locisimiles.cli import main
        
        with patch.object(sys, 'argv', ['locisimiles']):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code != 0


class TestCLIFileValidation:
    """Tests for CLI file validation."""

    def test_cli_missing_query_file(self, temp_dir, capsys):
        """Test error when query file doesn't exist."""
        from locisimiles.cli import main
        
        source_csv = temp_dir / "source.csv"
        source_csv.write_text("seg_id,text\ns1,Source\n", encoding="utf-8")
        output_path = temp_dir / "output.csv"
        
        with patch.object(sys, 'argv', [
            'locisimiles',
            str(temp_dir / "nonexistent.csv"),
            str(source_csv),
            '-o', str(output_path),
        ]):
            result = main()
        
        assert result == 1
        captured = capsys.readouterr()
        assert "Query file not found" in captured.err

    def test_cli_missing_source_file(self, temp_dir, capsys):
        """Test error when source file doesn't exist."""
        from locisimiles.cli import main
        
        query_csv = temp_dir / "query.csv"
        query_csv.write_text("seg_id,text\nq1,Query\n", encoding="utf-8")
        output_path = temp_dir / "output.csv"
        
        with patch.object(sys, 'argv', [
            'locisimiles',
            str(query_csv),
            str(temp_dir / "nonexistent.csv"),
            '-o', str(output_path),
        ]):
            result = main()
        
        assert result == 1
        captured = capsys.readouterr()
        assert "Source file not found" in captured.err


class TestCLIDeviceSelection:
    """Tests for CLI device selection."""

    @patch("locisimiles.cli.ClassificationPipelineWithCandidategeneration")
    @patch("locisimiles.cli.Document")
    def test_cli_device_auto_cpu(self, mock_doc, mock_pipeline, temp_dir, capsys):
        """Test auto device selection falls back to CPU when no GPU."""
        from locisimiles.cli import main
        
        # Setup mocks
        mock_doc.return_value = MagicMock()
        mock_doc.return_value.__iter__ = MagicMock(return_value=iter([]))
        mock_pipeline_instance = MagicMock()
        mock_pipeline_instance.run.return_value = {}
        mock_pipeline.return_value = mock_pipeline_instance
        
        query_csv = temp_dir / "query.csv"
        query_csv.write_text("seg_id,text\nq1,Query\n", encoding="utf-8")
        source_csv = temp_dir / "source.csv"
        source_csv.write_text("seg_id,text\ns1,Source\n", encoding="utf-8")
        output_path = temp_dir / "output.csv"
        
        with patch.object(sys, 'argv', [
            'locisimiles',
            str(query_csv),
            str(source_csv),
            '-o', str(output_path),
            '--device', 'auto',
            '-v',
        ]):
            with patch('torch.cuda.is_available', return_value=False):
                with patch('torch.backends.mps.is_available', return_value=False):
                    main()
        
        captured = capsys.readouterr()
        assert "cpu" in captured.out.lower()

    @patch("locisimiles.cli.ClassificationPipelineWithCandidategeneration")
    @patch("locisimiles.cli.Document")
    def test_cli_device_explicit(self, mock_doc, mock_pipeline, temp_dir):
        """Test explicit device selection is passed to pipeline."""
        from locisimiles.cli import main
        
        mock_doc.return_value = MagicMock()
        mock_doc.return_value.__iter__ = MagicMock(return_value=iter([]))
        mock_pipeline_instance = MagicMock()
        mock_pipeline_instance.run.return_value = {}
        mock_pipeline.return_value = mock_pipeline_instance
        
        query_csv = temp_dir / "query.csv"
        query_csv.write_text("seg_id,text\nq1,Query\n", encoding="utf-8")
        source_csv = temp_dir / "source.csv"
        source_csv.write_text("seg_id,text\ns1,Source\n", encoding="utf-8")
        output_path = temp_dir / "output.csv"
        
        with patch.object(sys, 'argv', [
            'locisimiles',
            str(query_csv),
            str(source_csv),
            '-o', str(output_path),
            '--device', 'cpu',
        ]):
            main()
        
        # Check pipeline was initialized with correct device
        mock_pipeline.assert_called_once()
        call_kwargs = mock_pipeline.call_args[1]
        assert call_kwargs['device'] == 'cpu'


class TestCLIOutputGeneration:
    """Tests for CLI output file generation."""

    @patch("locisimiles.cli.ClassificationPipelineWithCandidategeneration")
    @patch("locisimiles.cli.Document")
    def test_cli_output_csv_created(self, mock_doc_class, mock_pipeline_class, temp_dir):
        """Test that output CSV file is created."""
        from locisimiles.cli import main
        from locisimiles.document import TextSegment
        
        # Create mock query document
        mock_query_doc = MagicMock()
        query_segment = TextSegment("Query text", "q1", row_id=0)
        mock_query_doc.__iter__ = MagicMock(return_value=iter([query_segment]))
        mock_query_doc.__len__ = MagicMock(return_value=1)
        
        # Create mock source document
        mock_source_doc = MagicMock()
        source_segment = TextSegment("Source text", "s1", row_id=0)
        mock_source_doc.__iter__ = MagicMock(return_value=iter([source_segment]))
        mock_source_doc.__len__ = MagicMock(return_value=1)
        
        # Configure Document mock to return different docs
        mock_doc_class.side_effect = [mock_query_doc, mock_source_doc]
        
        # Configure pipeline mock
        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = {
            "q1": [CandidateJudge(segment=source_segment, candidate_score=0.9, judgment_score=0.8)],
        }
        mock_pipeline_class.return_value = mock_pipeline
        
        query_csv = temp_dir / "query.csv"
        query_csv.write_text("seg_id,text\nq1,Query text\n", encoding="utf-8")
        source_csv = temp_dir / "source.csv"
        source_csv.write_text("seg_id,text\ns1,Source text\n", encoding="utf-8")
        output_path = temp_dir / "output.csv"
        
        with patch.object(sys, 'argv', [
            'locisimiles',
            str(query_csv),
            str(source_csv),
            '-o', str(output_path),
        ]):
            result = main()
        
        assert result == 0 or result is None
        assert output_path.exists()

    @patch("locisimiles.cli.ClassificationPipelineWithCandidategeneration")
    @patch("locisimiles.cli.Document")
    def test_cli_output_csv_headers(self, mock_doc_class, mock_pipeline_class, temp_dir):
        """Test that output CSV has correct headers."""
        from locisimiles.cli import main
        from locisimiles.document import TextSegment
        import csv
        
        mock_query_doc = MagicMock()
        query_segment = TextSegment("Query", "q1", row_id=0)
        mock_query_doc.__iter__ = MagicMock(return_value=iter([query_segment]))
        mock_query_doc.__len__ = MagicMock(return_value=1)
        
        mock_source_doc = MagicMock()
        source_segment = TextSegment("Source", "s1", row_id=0)
        mock_source_doc.__iter__ = MagicMock(return_value=iter([source_segment]))
        mock_source_doc.__len__ = MagicMock(return_value=1)
        
        mock_doc_class.side_effect = [mock_query_doc, mock_source_doc]
        
        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = {
            "q1": [CandidateJudge(segment=source_segment, candidate_score=0.9, judgment_score=0.8)],
        }
        mock_pipeline_class.return_value = mock_pipeline
        
        query_csv = temp_dir / "query.csv"
        query_csv.write_text("seg_id,text\nq1,Query\n", encoding="utf-8")
        source_csv = temp_dir / "source.csv"
        source_csv.write_text("seg_id,text\ns1,Source\n", encoding="utf-8")
        output_path = temp_dir / "output.csv"
        
        with patch.object(sys, 'argv', [
            'locisimiles',
            str(query_csv),
            str(source_csv),
            '-o', str(output_path),
        ]):
            main()
        
        # Read output and check headers
        with open(output_path, newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            headers = next(reader)
        
        expected_headers = [
            'query_id', 'query_text', 'source_id', 'source_text',
            'similarity', 'probability', 'above_threshold'
        ]
        assert headers == expected_headers


class TestCLIVerboseMode:
    """Tests for CLI verbose mode."""

    @patch("locisimiles.cli.ClassificationPipelineWithCandidategeneration")
    @patch("locisimiles.cli.Document")
    def test_cli_verbose_flag(self, mock_doc_class, mock_pipeline_class, temp_dir, capsys):
        """Test verbose output includes expected information."""
        from locisimiles.cli import main
        from locisimiles.document import TextSegment
        
        mock_query_doc = MagicMock()
        mock_query_doc.__iter__ = MagicMock(return_value=iter([]))
        mock_query_doc.__len__ = MagicMock(return_value=1)
        
        mock_source_doc = MagicMock()
        mock_source_doc.__iter__ = MagicMock(return_value=iter([]))
        mock_source_doc.__len__ = MagicMock(return_value=2)
        
        mock_doc_class.side_effect = [mock_query_doc, mock_source_doc]
        
        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = {}
        mock_pipeline_class.return_value = mock_pipeline
        
        query_csv = temp_dir / "query.csv"
        query_csv.write_text("seg_id,text\nq1,Query\n", encoding="utf-8")
        source_csv = temp_dir / "source.csv"
        source_csv.write_text("seg_id,text\ns1,S1\ns2,S2\n", encoding="utf-8")
        output_path = temp_dir / "output.csv"
        
        with patch.object(sys, 'argv', [
            'locisimiles',
            str(query_csv),
            str(source_csv),
            '-o', str(output_path),
            '-v',
        ]):
            with patch('torch.cuda.is_available', return_value=False):
                with patch('torch.backends.mps.is_available', return_value=False):
                    main()
        
        captured = capsys.readouterr()
        # Should see verbose output
        assert "Loading" in captured.out or "device" in captured.out.lower()


class TestCLIPipelineParameters:
    """Tests for CLI pipeline parameters."""

    @patch("locisimiles.cli.ClassificationPipelineWithCandidategeneration")
    @patch("locisimiles.cli.Document")
    def test_cli_topk_parameter(self, mock_doc_class, mock_pipeline_class, temp_dir):
        """Test that top-k parameter is passed to pipeline."""
        from locisimiles.cli import main
        
        mock_doc_class.return_value = MagicMock()
        mock_doc_class.return_value.__iter__ = MagicMock(return_value=iter([]))
        mock_doc_class.return_value.__len__ = MagicMock(return_value=1)
        
        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = {}
        mock_pipeline_class.return_value = mock_pipeline
        
        query_csv = temp_dir / "query.csv"
        query_csv.write_text("seg_id,text\nq1,Query\n", encoding="utf-8")
        source_csv = temp_dir / "source.csv"
        source_csv.write_text("seg_id,text\ns1,Source\n", encoding="utf-8")
        output_path = temp_dir / "output.csv"
        
        with patch.object(sys, 'argv', [
            'locisimiles',
            str(query_csv),
            str(source_csv),
            '-o', str(output_path),
            '-k', '20',
        ]):
            main()
        
        # Check pipeline.run was called with correct top_k
        mock_pipeline.run.assert_called_once()
        call_kwargs = mock_pipeline.run.call_args[1]
        assert call_kwargs['top_k'] == 20

    @patch("locisimiles.cli.ClassificationPipelineWithCandidategeneration")
    @patch("locisimiles.cli.Document")
    def test_cli_threshold_parameter(self, mock_doc_class, mock_pipeline_class, temp_dir):
        """Test that threshold parameter affects output filtering."""
        from locisimiles.cli import main
        from locisimiles.document import TextSegment
        import csv
        
        mock_query_doc = MagicMock()
        query_segment = TextSegment("Query", "q1", row_id=0)
        mock_query_doc.__iter__ = MagicMock(return_value=iter([query_segment]))
        mock_query_doc.__len__ = MagicMock(return_value=1)
        
        mock_source_doc = MagicMock()
        source_segment = TextSegment("Source", "s1", row_id=0)
        mock_source_doc.__iter__ = MagicMock(return_value=iter([source_segment]))
        mock_source_doc.__len__ = MagicMock(return_value=1)
        
        mock_doc_class.side_effect = [mock_query_doc, mock_source_doc]
        
        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = {
            "q1": [CandidateJudge(segment=source_segment, candidate_score=0.9, judgment_score=0.6)],  # judgment_score=0.6
        }
        mock_pipeline_class.return_value = mock_pipeline
        
        query_csv = temp_dir / "query.csv"
        query_csv.write_text("seg_id,text\nq1,Query\n", encoding="utf-8")
        source_csv = temp_dir / "source.csv"
        source_csv.write_text("seg_id,text\ns1,Source\n", encoding="utf-8")
        output_path = temp_dir / "output.csv"
        
        with patch.object(sys, 'argv', [
            'locisimiles',
            str(query_csv),
            str(source_csv),
            '-o', str(output_path),
            '-t', '0.7',  # threshold = 0.7, prob = 0.6 → above_threshold = False
        ]):
            main()
        
        # Read output and check above_threshold column
        with open(output_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        if rows:  # If there are data rows
            assert rows[0]['above_threshold'] == 'No'

    @patch("locisimiles.cli.ClassificationPipelineWithCandidategeneration")
    @patch("locisimiles.cli.Document")
    def test_cli_custom_models(self, mock_doc_class, mock_pipeline_class, temp_dir):
        """Test that custom model names are passed to pipeline."""
        from locisimiles.cli import main
        
        mock_doc_class.return_value = MagicMock()
        mock_doc_class.return_value.__iter__ = MagicMock(return_value=iter([]))
        mock_doc_class.return_value.__len__ = MagicMock(return_value=1)
        
        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = {}
        mock_pipeline_class.return_value = mock_pipeline
        
        query_csv = temp_dir / "query.csv"
        query_csv.write_text("seg_id,text\nq1,Query\n", encoding="utf-8")
        source_csv = temp_dir / "source.csv"
        source_csv.write_text("seg_id,text\ns1,Source\n", encoding="utf-8")
        output_path = temp_dir / "output.csv"
        
        with patch.object(sys, 'argv', [
            'locisimiles',
            str(query_csv),
            str(source_csv),
            '-o', str(output_path),
            '--classification-model', 'custom/classifier',
            '--embedding-model', 'custom/embedder',
        ]):
            main()
        
        # Check pipeline was initialized with custom models
        mock_pipeline_class.assert_called_once()
        call_kwargs = mock_pipeline_class.call_args[1]
        assert call_kwargs['classification_name'] == 'custom/classifier'
        assert call_kwargs['embedding_model_name'] == 'custom/embedder'


class TestCLIErrorHandling:
    """Tests for CLI error handling."""

    @patch("locisimiles.cli.ClassificationPipelineWithCandidategeneration")
    @patch("locisimiles.cli.Document")
    def test_cli_pipeline_error(self, mock_doc_class, mock_pipeline_class, temp_dir, capsys):
        """Test that pipeline errors are handled gracefully."""
        from locisimiles.cli import main
        
        mock_doc_class.return_value = MagicMock()
        mock_doc_class.return_value.__len__ = MagicMock(return_value=1)
        
        mock_pipeline_class.side_effect = Exception("Model loading failed")
        
        query_csv = temp_dir / "query.csv"
        query_csv.write_text("seg_id,text\nq1,Query\n", encoding="utf-8")
        source_csv = temp_dir / "source.csv"
        source_csv.write_text("seg_id,text\ns1,Source\n", encoding="utf-8")
        output_path = temp_dir / "output.csv"
        
        with patch.object(sys, 'argv', [
            'locisimiles',
            str(query_csv),
            str(source_csv),
            '-o', str(output_path),
        ]):
            result = main()
        
        # Should return error code
        assert result == 1
        captured = capsys.readouterr()
        assert "Error" in captured.err or "error" in captured.err.lower()
