"""
Unit tests for locisimiles.pipeline._types module.
Tests type definitions and utility functions.
"""
import pytest
from io import StringIO
import sys

from locisimiles.document import TextSegment
from locisimiles.pipeline._types import pretty_print, SimDict, FullDict


class TestTypeStructures:
    """Tests for type structure definitions."""

    def test_simdict_structure(self, sample_segments):
        """Test SimDict matches expected format: Dict[str, List[Tuple[TextSegment, float]]]."""
        sim_dict: SimDict = {
            "q1": [
                (sample_segments[0], 0.95),
                (sample_segments[1], 0.85),
            ],
            "q2": [
                (sample_segments[2], 0.75),
            ],
        }
        # Verify structure
        assert isinstance(sim_dict, dict)
        for qid, pairs in sim_dict.items():
            assert isinstance(qid, str)
            assert isinstance(pairs, list)
            for segment, score in pairs:
                assert isinstance(segment, TextSegment)
                assert isinstance(score, float)

    def test_fulldict_structure(self, sample_segments):
        """Test FullDict matches expected format: Dict[str, List[Tuple[TextSegment, float, float]]]."""
        full_dict: FullDict = {
            "q1": [
                (sample_segments[0], 0.95, 0.88),
                (sample_segments[1], 0.85, 0.45),
            ],
            "q2": [
                (sample_segments[2], 0.75, 0.32),
            ],
        }
        # Verify structure
        assert isinstance(full_dict, dict)
        for qid, pairs in full_dict.items():
            assert isinstance(qid, str)
            assert isinstance(pairs, list)
            for segment, sim, prob in pairs:
                assert isinstance(segment, TextSegment)
                assert isinstance(sim, (float, type(None)))
                assert isinstance(prob, float)

    def test_fulldict_with_none_similarity(self, sample_segments):
        """Test FullDict can have None for similarity (classification-only pipeline)."""
        full_dict: FullDict = {
            "q1": [
                (sample_segments[0], None, 0.88),  # None similarity
            ],
        }
        segment, sim, prob = full_dict["q1"][0]
        assert sim is None
        assert prob == 0.88


class TestPrettyPrint:
    """Tests for the pretty_print utility function."""

    def test_pretty_print_output(self, sample_segments):
        """Test pretty_print produces formatted output."""
        full_dict: FullDict = {
            "query_1": [
                (sample_segments[0], 0.95, 0.88),
                (sample_segments[1], 0.75, 0.45),
            ],
        }
        
        # Capture stdout
        captured_output = StringIO()
        sys.stdout = captured_output
        
        pretty_print(full_dict)
        
        sys.stdout = sys.__stdout__
        output = captured_output.getvalue()
        
        # Verify output contains expected elements
        assert "query_1" in output
        assert "seg1" in output
        assert "sim=" in output
        assert "P(pos)=" in output

    def test_pretty_print_multiple_queries(self, sample_segments):
        """Test pretty_print handles multiple queries."""
        full_dict: FullDict = {
            "q1": [(sample_segments[0], 0.9, 0.8)],
            "q2": [(sample_segments[1], 0.7, 0.6)],
            "q3": [(sample_segments[2], 0.5, 0.4)],
        }
        
        captured_output = StringIO()
        sys.stdout = captured_output
        
        pretty_print(full_dict)
        
        sys.stdout = sys.__stdout__
        output = captured_output.getvalue()
        
        assert "q1" in output
        assert "q2" in output
        assert "q3" in output

    def test_pretty_print_none_similarity(self, sample_segments):
        """Test pretty_print handles None similarity scores."""
        full_dict: FullDict = {
            "q1": [(sample_segments[0], None, 0.88)],
        }
        
        captured_output = StringIO()
        sys.stdout = captured_output
        
        pretty_print(full_dict)
        
        sys.stdout = sys.__stdout__
        output = captured_output.getvalue()
        
        # Should show "N/A" for None similarity
        assert "N/A" in output

    def test_pretty_print_empty_dict(self):
        """Test pretty_print handles empty FullDict."""
        full_dict: FullDict = {}
        
        captured_output = StringIO()
        sys.stdout = captured_output
        
        pretty_print(full_dict)
        
        sys.stdout = sys.__stdout__
        output = captured_output.getvalue()
        
        # Should produce no output (or just newlines)
        assert output.strip() == ""

    def test_pretty_print_empty_results(self):
        """Test pretty_print handles query with no candidates."""
        full_dict: FullDict = {
            "q1": [],
        }
        
        captured_output = StringIO()
        sys.stdout = captured_output
        
        pretty_print(full_dict)
        
        sys.stdout = sys.__stdout__
        output = captured_output.getvalue()
        
        assert "q1" in output

    def test_pretty_print_formatting(self, sample_segments):
        """Test pretty_print uses correct number formatting."""
        full_dict: FullDict = {
            "q1": [(sample_segments[0], 0.123456, 0.987654)],
        }
        
        captured_output = StringIO()
        sys.stdout = captured_output
        
        pretty_print(full_dict)
        
        sys.stdout = sys.__stdout__
        output = captured_output.getvalue()
        
        # Check formatting (3 decimal places for sim and prob)
        assert "+0.123" in output or "0.123" in output
        assert "0.988" in output  # Rounded


class TestScoreType:
    """Tests for ScoreT type."""

    def test_score_is_float(self):
        """Test that scores are floats."""
        from locisimiles.pipeline._types import ScoreT
        score: ScoreT = 0.95
        assert isinstance(score, float)

    def test_score_range(self):
        """Test scores can be in valid ranges."""
        from locisimiles.pipeline._types import ScoreT
        
        # Probability scores [0, 1]
        prob_score: ScoreT = 0.5
        assert 0.0 <= prob_score <= 1.0
        
        # Similarity scores (cosine) [-1, 1]
        sim_score: ScoreT = -0.3
        assert -1.0 <= sim_score <= 1.0
