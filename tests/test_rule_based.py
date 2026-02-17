"""
Unit tests for locisimiles.pipeline.rule_based module.
Tests RuleBasedPipeline class with its various filters and preprocessing steps.
"""
import pytest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from locisimiles.document import Document, TextSegment
from locisimiles.pipeline._types import CandidateJudge, CandidateJudgeOutput


class TestRuleBasedPipelineInit:
    """Tests for RuleBasedPipeline initialization."""

    def test_init_default_parameters(self):
        """Test initialization with default parameters."""
        from locisimiles.pipeline.rule_based import RuleBasedPipeline
        
        pipeline = RuleBasedPipeline()
        
        assert pipeline.min_shared_words == 2
        assert pipeline.min_complura == 4
        assert pipeline.max_distance == 3
        assert pipeline.similarity_threshold == 0.3
        assert pipeline.use_htrg is False
        assert pipeline.use_similarity is False
        assert len(pipeline.stopwords) > 0

    def test_init_custom_parameters(self):
        """Test initialization with custom parameters."""
        from locisimiles.pipeline.rule_based import RuleBasedPipeline
        
        custom_stopwords = {"et", "in", "cum"}
        pipeline = RuleBasedPipeline(
            min_shared_words=3,
            min_complura=5,
            max_distance=4,
            similarity_threshold=0.5,
            stopwords=custom_stopwords,
        )
        
        assert pipeline.min_shared_words == 3
        assert pipeline.min_complura == 5
        assert pipeline.max_distance == 4
        assert pipeline.similarity_threshold == 0.5
        assert pipeline.stopwords == custom_stopwords

    def test_init_htrg_without_torch_raises(self):
        """Test that enabling HTRG without torch raises ImportError."""
        from locisimiles.pipeline.rule_based import RuleBasedPipeline, TORCH_AVAILABLE
        
        if not TORCH_AVAILABLE:
            with pytest.raises(ImportError, match="HTRG filter requires torch"):
                RuleBasedPipeline(use_htrg=True)

    def test_init_similarity_without_spacy_raises(self):
        """Test that enabling similarity without spacy raises ImportError."""
        from locisimiles.pipeline.rule_based import RuleBasedPipeline, SPACY_AVAILABLE
        
        if not SPACY_AVAILABLE:
            with pytest.raises(ImportError, match="Similarity filter requires spacy"):
                RuleBasedPipeline(use_similarity=True)


class TestRuleBasedTokenization:
    """Tests for tokenization methods."""

    def test_tokenize_basic(self):
        """Test basic tokenization."""
        from locisimiles.pipeline.rule_based import RuleBasedPipeline
        
        pipeline = RuleBasedPipeline()
        tokens = pipeline._tokenize("Hello, world!")
        
        assert "Hello" in tokens
        assert "world" in tokens
        assert "," in tokens
        assert "!" in tokens

    def test_tokenize_latin_text(self):
        """Test tokenization of Latin text."""
        from locisimiles.pipeline.rule_based import RuleBasedPipeline
        
        pipeline = RuleBasedPipeline()
        tokens = pipeline._tokenize("Gallia est omnis divisa")
        
        assert tokens == ["Gallia", "est", "omnis", "divisa"]

    def test_simple_tokenize(self):
        """Test simple tokenizer for matching."""
        from locisimiles.pipeline.rule_based import RuleBasedPipeline
        
        pipeline = RuleBasedPipeline()
        tokens = pipeline._simple_tokenize("Hello, world! How are you?")
        
        assert "Hello" in tokens
        assert "world" in tokens
        assert "How" in tokens
        assert "are" in tokens
        assert "you" in tokens


class TestRuleBasedTransformToken:
    """Tests for prefix assimilation transformations."""

    def test_transform_adt_to_att(self):
        """Test prefix assimilation adt -> att."""
        from locisimiles.pipeline.rule_based import RuleBasedPipeline
        
        pipeline = RuleBasedPipeline()
        
        assert pipeline._transform_token("adtingo") == "attingo"
        assert pipeline._transform_token("Adtingo") == "Attingo"

    def test_transform_adp_to_app(self):
        """Test prefix assimilation adp -> app."""
        from locisimiles.pipeline.rule_based import RuleBasedPipeline
        
        pipeline = RuleBasedPipeline()
        
        assert pipeline._transform_token("adpono") == "appono"

    def test_transform_inm_to_imm(self):
        """Test prefix assimilation inm -> imm."""
        from locisimiles.pipeline.rule_based import RuleBasedPipeline
        
        pipeline = RuleBasedPipeline()
        
        assert pipeline._transform_token("inmortalis") == "immortalis"

    def test_transform_conm_to_comm(self):
        """Test prefix assimilation conm -> comm."""
        from locisimiles.pipeline.rule_based import RuleBasedPipeline
        
        pipeline = RuleBasedPipeline()
        
        assert pipeline._transform_token("conmitto") == "committo"

    def test_transform_no_change(self):
        """Test that non-matching tokens remain unchanged."""
        from locisimiles.pipeline.rule_based import RuleBasedPipeline
        
        pipeline = RuleBasedPipeline()
        
        assert pipeline._transform_token("Roma") == "Roma"
        assert pipeline._transform_token("Caesar") == "Caesar"

    def test_transform_ads_before_vowel(self):
        """Test ads prefix before vowel."""
        from locisimiles.pipeline.rule_based import RuleBasedPipeline
        
        pipeline = RuleBasedPipeline()
        
        # ads before vowel -> ass
        assert pipeline._transform_token("adserit") == "asserit"

    def test_transform_ads_before_consonant(self):
        """Test ads prefix before consonant."""
        from locisimiles.pipeline.rule_based import RuleBasedPipeline
        
        pipeline = RuleBasedPipeline()
        
        # ads before consonant -> as (removes d)
        assert pipeline._transform_token("adstringit") == "astringit"


class TestRuleBasedPhrasing:
    """Tests for prose and poetry phrasing functions."""

    def test_normalize_quotation_marks(self):
        """Test quotation mark normalization."""
        from locisimiles.pipeline.rule_based import RuleBasedPipeline
        
        pipeline = RuleBasedPipeline()
        # Use raw string with escaped fancy quotes
        text_list = [["id1", 'He said \u201chello\u201d and \u2018goodbye\u2019']]
        
        result = pipeline._normalize_quotation_marks(text_list)
        
        assert "'" in result[0][1]
        assert '\u201c' not in result[0][1]  # left double quote
        assert '\u201d' not in result[0][1]  # right double quote

    def test_remove_whitespace_connectors(self):
        """Test connector normalization (que, ve)."""
        from locisimiles.pipeline.rule_based import RuleBasedPipeline
        
        pipeline = RuleBasedPipeline()
        text_list = [["id1", "Roma que Carthago"]]
        
        result = pipeline._remove_whitespace_connectors(text_list)
        
        assert "Romaque" in result[0][1]

    def test_strip_whitespaces(self):
        """Test whitespace stripping."""
        from locisimiles.pipeline.rule_based import RuleBasedPipeline
        
        pipeline = RuleBasedPipeline()
        text_list = [["id1", "  Hello World  "]]
        
        result = pipeline._strip_whitespaces(text_list)
        
        assert result[0][1] == "Hello World"

    def test_cleanup_removes_empty(self):
        """Test cleanup removes empty entries."""
        from locisimiles.pipeline.rule_based import RuleBasedPipeline
        
        pipeline = RuleBasedPipeline()
        text_list = [
            ["id1", "Valid text"],
            ["id2", "  "],
            ["id3", "Another valid"],
        ]
        
        result = pipeline._cleanup(text_list)
        
        assert len(result) == 2
        assert result[0][0] == "id1"
        assert result[1][0] == "id3"

    def test_phrasing_prose(self):
        """Test prose phrasing pipeline."""
        from locisimiles.pipeline.rule_based import RuleBasedPipeline
        
        pipeline = RuleBasedPipeline()
        text_list = [["id1", '  \u201cHello\u201d Roma que Carthago  ']]
        
        result = pipeline._phrasing_prose(text_list)
        
        assert len(result) == 1
        # Should have normalized quotes and connectors
        assert "'" in result[0][1]
        assert "Romaque" in result[0][1]

    def test_phrasing_poetry_marks_verse_endings(self):
        """Test poetry phrasing adds verse ending markers."""
        from locisimiles.pipeline.rule_based import RuleBasedPipeline
        
        pipeline = RuleBasedPipeline()
        text_list = [["id1", "Arma virumque cano"]]
        
        result = pipeline._phrasing_poetry(text_list)
        
        assert result[0][1].endswith(" /")


class TestRuleBasedNormalization:
    """Tests for text normalization."""

    def test_normalize_for_matching_v_to_u(self):
        """Test v -> u normalization."""
        from locisimiles.pipeline.rule_based import RuleBasedPipeline
        
        pipeline = RuleBasedPipeline()
        text_list = [["virtus"]]
        
        result = pipeline._normalize_for_matching(text_list)
        
        assert "uirtus" in result[0][0].strip()

    def test_normalize_for_matching_j_to_i(self):
        """Test j -> i normalization."""
        from locisimiles.pipeline.rule_based import RuleBasedPipeline
        
        pipeline = RuleBasedPipeline()
        text_list = [["Julius"]]
        
        result = pipeline._normalize_for_matching(text_list)
        
        assert "iulius" in result[0][0].strip()

    def test_normalize_for_matching_lowercase(self):
        """Test lowercase normalization."""
        from locisimiles.pipeline.rule_based import RuleBasedPipeline
        
        pipeline = RuleBasedPipeline()
        text_list = [["ROMA"]]
        
        result = pipeline._normalize_for_matching(text_list)
        
        assert "roma" in result[0][0].strip()


class TestRuleBasedTextMatching:
    """Tests for text matching functionality."""

    def test_find_adjacent_sequence_basic(self):
        """Test finding adjacent sequence of indices."""
        from locisimiles.pipeline.rule_based import RuleBasedPipeline
        
        pipeline = RuleBasedPipeline()
        
        # Sequence of 4 adjacent
        indices = [1, 2, 3, 4, 7, 8]
        result = pipeline._find_adjacent_sequence(indices)
        
        assert result == [1, 2, 3, 4]

    def test_find_adjacent_sequence_longer(self):
        """Test finding longest adjacent sequence."""
        from locisimiles.pipeline.rule_based import RuleBasedPipeline
        
        pipeline = RuleBasedPipeline()
        
        # Two sequences, second is longer
        indices = [1, 2, 3, 4, 10, 11, 12, 13, 14]
        result = pipeline._find_adjacent_sequence(indices)
        
        assert result == [10, 11, 12, 13, 14]

    def test_find_adjacent_sequence_none(self):
        """Test no adjacent sequence found."""
        from locisimiles.pipeline.rule_based import RuleBasedPipeline
        
        pipeline = RuleBasedPipeline()
        
        # Only 3 adjacent (less than 4)
        indices = [1, 2, 3, 10, 20]
        result = pipeline._find_adjacent_sequence(indices)
        
        assert result == []

    def test_highlight_words(self):
        """Test word highlighting with ** markers."""
        from locisimiles.pipeline.rule_based import RuleBasedPipeline
        
        pipeline = RuleBasedPipeline()
        text = " Roma et Carthago "
        words = ["roma", "carthago"]
        
        result = pipeline._highlight_words(text, words)
        
        assert "**Roma**" in result
        assert "**Carthago**" in result
        assert "et" in result
        assert "**et**" not in result

    def test_min_distance_two_tokens(self):
        """Test minimum distance between two shared tokens."""
        from locisimiles.pipeline.rule_based import RuleBasedPipeline
        
        pipeline = RuleBasedPipeline()
        shared = [("**roma**", 2), ("**carthago**", 5)]
        
        result = pipeline._min_distance(shared)
        
        assert result == 3

    def test_min_distance_single_token(self):
        """Test minimum distance with single token returns infinity."""
        from locisimiles.pipeline.rule_based import RuleBasedPipeline
        
        pipeline = RuleBasedPipeline()
        shared = [("**roma**", 2)]
        
        result = pipeline._min_distance(shared)
        
        assert result == float('inf')


class TestRuleBasedFilters:
    """Tests for filtering functions."""

    def test_extract_substrings(self):
        """Test substring extraction between markers."""
        from locisimiles.pipeline.rule_based import RuleBasedPipeline
        
        pipeline = RuleBasedPipeline()
        text = "Before **roma** middle text **carthago** after"
        shared = ["roma", "carthago"]
        
        result = pipeline._extract_substrings(text, shared)
        
        assert len(result) == 1
        assert "middle text" in result[0]

    def test_compare_punctuation_equal(self):
        """Test punctuation comparison with equal counts."""
        from locisimiles.pipeline.rule_based import RuleBasedPipeline
        
        pipeline = RuleBasedPipeline()
        substr1 = [", hello,"]
        substr2 = [", world,"]
        
        result = pipeline._compare_punctuation(substr1, substr2, ",")
        
        assert all(result)

    def test_compare_punctuation_unequal(self):
        """Test punctuation comparison with unequal counts."""
        from locisimiles.pipeline.rule_based import RuleBasedPipeline
        
        pipeline = RuleBasedPipeline()
        substr1 = [", hello,,"]
        substr2 = [", world,"]
        
        result = pipeline._compare_punctuation(substr1, substr2, ",")
        
        assert not all(result)

    def test_combine_matches_no_duplicates(self):
        """Test combining matches removes duplicates."""
        from locisimiles.pipeline.rule_based import RuleBasedPipeline
        
        pipeline = RuleBasedPipeline()
        matches = [[1, "s1", "text", "t1", "text", "shared"]]
        complura = [
            [2, "s1", "text", "t1", "text", "shared"],  # duplicate
            [3, "s2", "text", "t2", "text", "shared"],  # new
        ]
        
        result = pipeline._combine_matches(matches, complura)
        
        assert len(result) == 2


class TestRuleBasedStopwords:
    """Tests for stopword handling."""

    def test_default_stopwords_loaded(self):
        """Test that default stopwords are loaded."""
        from locisimiles.pipeline.rule_based import RuleBasedPipeline
        
        pipeline = RuleBasedPipeline()
        
        # Common Latin stopwords should be present
        assert "et" in pipeline.stopwords
        assert "in" in pipeline.stopwords
        assert "non" in pipeline.stopwords

    def test_custom_stopwords(self):
        """Test custom stopwords override defaults."""
        from locisimiles.pipeline.rule_based import RuleBasedPipeline
        
        custom = {"custom", "words"}
        pipeline = RuleBasedPipeline(stopwords=custom)
        
        assert pipeline.stopwords == custom
        assert "et" not in pipeline.stopwords

    def test_load_stopwords_from_file(self):
        """Test loading stopwords from file."""
        from locisimiles.pipeline.rule_based import RuleBasedPipeline
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("word1\nword2\nword3\n")
            f.flush()
            
            pipeline = RuleBasedPipeline()
            initial_count = len(pipeline.stopwords)
            pipeline.load_stopwords(f.name)
            
            assert "word1" in pipeline.stopwords
            assert "word2" in pipeline.stopwords
            assert "word3" in pipeline.stopwords
            assert len(pipeline.stopwords) > initial_count
            
            Path(f.name).unlink()


class TestRuleBasedDocumentConversion:
    """Tests for document conversion."""

    def test_document_to_list(self, sample_document):
        """Test converting Document to internal list format."""
        from locisimiles.pipeline.rule_based import RuleBasedPipeline
        
        pipeline = RuleBasedPipeline()
        result = pipeline._document_to_list(sample_document)
        
        assert len(result) == len(sample_document)
        assert all(len(item) == 2 for item in result)
        assert all(isinstance(item[0], str) for item in result)
        assert all(isinstance(item[1], str) for item in result)

    def test_matches_to_judge_output(self, sample_document, sample_source_document):
        """Test converting matches to CandidateJudgeOutput format."""
        from locisimiles.pipeline.rule_based import RuleBasedPipeline
        
        pipeline = RuleBasedPipeline()
        
        # Create sample matches
        matches = [
            [1, "src1", "source text", "seg1", "target text", "shared; words"]
        ]
        
        result = pipeline._matches_to_judge_output(matches, sample_source_document, sample_document)
        
        assert isinstance(result, dict)
        assert "seg1" in result


class TestRuleBasedRun:
    """Integration tests for the run method."""

    def test_run_basic(self, sample_document, sample_source_document):
        """Test basic run with two documents."""
        from locisimiles.pipeline.rule_based import RuleBasedPipeline
        
        pipeline = RuleBasedPipeline(min_shared_words=1)
        result = pipeline.run(query=sample_document, source=sample_source_document)
        
        assert isinstance(result, dict)
        # Should have entries for each query segment
        for seg in sample_document:
            assert str(seg.id) in result

    def test_run_with_prose_genre(self, sample_document, sample_source_document):
        """Test run with prose genre specification."""
        from locisimiles.pipeline.rule_based import RuleBasedPipeline
        
        pipeline = RuleBasedPipeline()
        result = pipeline.run(
            query=sample_document,
            source=sample_source_document,
            query_genre="prose",
            source_genre="prose",
        )
        
        assert isinstance(result, dict)

    def test_run_with_poetry_genre(self, sample_document, sample_source_document):
        """Test run with poetry genre specification."""
        from locisimiles.pipeline.rule_based import RuleBasedPipeline
        
        pipeline = RuleBasedPipeline()
        result = pipeline.run(
            query=sample_document,
            source=sample_source_document,
            query_genre="poetry",
            source_genre="poetry",
        )
        
        assert isinstance(result, dict)

    def test_run_returns_judge_output_format(self, sample_document, sample_source_document):
        """Test that run returns proper CandidateJudgeOutput format."""
        from locisimiles.pipeline.rule_based import RuleBasedPipeline
        
        pipeline = RuleBasedPipeline(min_shared_words=1)
        result = pipeline.run(query=sample_document, source=sample_source_document)
        
        # All values should be lists of CandidateJudge objects
        for seg_id, matches in result.items():
            assert isinstance(matches, list)
            for match in matches:
                assert isinstance(match, CandidateJudge)
                assert isinstance(match.segment, TextSegment)


class TestRuleBasedPreprocess:
    """Tests for the preprocessing pipeline."""

    def test_preprocess_prose(self):
        """Test preprocessing for prose texts."""
        from locisimiles.pipeline.rule_based import RuleBasedPipeline
        
        pipeline = RuleBasedPipeline()
        text_list = [["id1", '  \u201cGallia\u201d est omnis divisa  ']]
        
        result = pipeline._preprocess(text_list, "prose")
        
        assert len(result) == 1
        # Should be cleaned up
        assert result[0][1].strip() != ""

    def test_preprocess_poetry(self):
        """Test preprocessing for poetry texts."""
        from locisimiles.pipeline.rule_based import RuleBasedPipeline
        
        pipeline = RuleBasedPipeline()
        text_list = [["id1", "Arma virumque cano"]]
        
        result = pipeline._preprocess(text_list, "poetry")
        
        assert len(result) == 1
        # Poetry should have verse ending marker
        assert result[0][1].endswith(" /")


class TestRuleBasedCompareTexts:
    """Tests for text comparison functionality."""

    def test_compare_texts_finds_matches(self):
        """Test that compare_texts finds matching passages."""
        from locisimiles.pipeline.rule_based import RuleBasedPipeline
        
        pipeline = RuleBasedPipeline(min_shared_words=2, max_distance=10)
        
        source = [["s1", "Roma magna urbs antiqua est"]]
        target = [["t1", "Carthago magna urbs nova est"]]
        
        # Preprocess
        source = pipeline._preprocess(source, "prose")
        target = pipeline._preprocess(target, "prose")
        
        matches, complura = pipeline._compare_texts(source, target)
        
        # Should find matches (magna, urbs are shared non-stopwords)
        assert isinstance(matches, list)
        assert isinstance(complura, list)


# ============== FIXTURES ==============

@pytest.fixture
def sample_document(temp_dir):
    """Create a sample Document for testing."""
    csv_path = temp_dir / "target.csv"
    csv_path.write_text(
        "seg_id,text\n"
        "seg1,This is the first segment with some words.\n"
        "seg2,This is the second segment with other words.\n"
        "seg3,This is the third segment with more words.\n",
        encoding="utf-8"
    )
    return Document(csv_path)


@pytest.fixture  
def sample_source_document(temp_dir):
    """Create a sample source Document for testing."""
    csv_path = temp_dir / "source.csv"
    csv_path.write_text(
        "seg_id,text\n"
        "src1,This is a source with first and words.\n"
        "src2,Another source with second segment.\n"
        "src3,Third source with different content.\n",
        encoding="utf-8"
    )
    return Document(csv_path)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)
