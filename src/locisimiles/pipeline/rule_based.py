# pipeline/rule_based.py
# -*- coding: utf-8 -*-
"""
Rule-based pipeline for intertextuality detection in Latin texts.

This pipeline implements a multi-stage rule-based approach:
1. Text preprocessing (assimilation, orthographic normalization)
2. Text matching (finding shared non-stopword tokens)
3. Distance criterion filter
4. Scissa filter (punctuation comparison)
5. HTRG filter (Part-of-Speech analysis) - optional
6. Similarity filter (word embedding similarity) - optional

Based on work by Michael Wittweiler, Franziska Schropp, and Marie Revellio.

This work was carried out as part of the project "Zitieren als narrative Strategie.
Eine digital-hermeneutische Untersuchung von Intertextualitätsphänomenen am Beispiel
des Briefcorpus des Kirchenlehrers Hieronymus." under the supervision of
Prof. Dr. Barbara Feichtinger and Dr. Marie Revellio, and was supported by the
German Research Foundation (DFG, Forschungsgemeinschaft) [382880410].
"""
from __future__ import annotations

import csv
import re
import string
import itertools
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple, Set, Union, Any, Optional, Sequence

import numpy as np

from locisimiles.document import Document, TextSegment
from locisimiles.pipeline._types import FullDict

# Optional heavy dependencies - loaded lazily
try:
    import torch
    from transformers import XLMRobertaForTokenClassification, XLMRobertaTokenizer
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import spacy
    from spacy.language import Language
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False


# ============== DEFAULT CONFIGURATION ==============

DEFAULT_CONFIG = {
    "min_shared_words": 2,           # Minimum non-stopword matches required
    "min_complura": 4,               # Minimum adjacent tokens for complura matches
    "max_distance": 3,               # Maximum distance between shared words
    "similarity_threshold": 0.3,     # Cosine similarity threshold for filtering
    "pos_model": "enelpol/evalatin2022-pos-open",
    "spacy_model": "la_core_web_lg",
}


# ============== STOPWORDS ==============

# Default Latin stopwords (common function words)
DEFAULT_STOPWORDS = {
    "et", "in", "non", "est", "ut", "cum", "ad", "que", "sed", "si",
    "quod", "enim", "nec", "per", "qui", "quae", "ex", "de", "ab",
    "aut", "atque", "ac", "an", "ante", "apud", "at", "autem", "circa",
    "contra", "cur", "dum", "ego", "ergo", "esse", "hic", "iam", "idem",
    "ideo", "igitur", "ille", "inter", "ipse", "is", "iste", "ita",
    "magis", "me", "mihi", "nam", "ne", "neque", "nihil", "nisi", "nos",
    "noster", "nunc", "ob", "post", "pro", "propter", "quam", "quando",
    "quia", "quid", "quidem", "quo", "quoque", "re", "sic", "sine",
    "sub", "sum", "super", "suus", "tam", "tamen", "te", "tibi", "tu",
    "tum", "tunc", "uel", "vel", "vero", "a", "e", "o", "i", "u",
}


class RuleBasedPipeline:
    """
    A rule-based pipeline for detecting intertextuality in Latin texts.
    
    This pipeline uses lexical matching combined with various filters to identify
    potential quotations, allusions, and textual reuse between Latin documents.
    
    Features:
        - Orthographic normalization (v→u, j→i, prefix assimilation)
        - Stopword filtering
        - Distance criterion (shared words must be close together)
        - Scissa filter (punctuation agreement)
        - HTRG filter (Part-of-Speech agreement) - optional
        - Similarity filter (embedding-based) - optional
    
    Example:
        ```python
        pipeline = RuleBasedPipeline()
        results = pipeline.run(query=query_doc, source=source_doc)
        ```
    """
    
    def __init__(
        self,
        *,
        min_shared_words: int = 2,
        min_complura: int = 4,
        max_distance: int = 3,
        similarity_threshold: float = 0.3,
        stopwords: Optional[Set[str]] = None,
        use_htrg: bool = False,
        use_similarity: bool = False,
        pos_model: str = "enelpol/evalatin2022-pos-open",
        spacy_model: str = "la_core_web_lg",
        device: Optional[str] = None,
    ):
        """
        Initialize the rule-based pipeline.
        
        Args:
            min_shared_words: Minimum number of shared non-stopwords required.
            min_complura: Minimum adjacent tokens for complura detection.
            max_distance: Maximum distance between shared words.
            similarity_threshold: Threshold for semantic similarity filter.
            stopwords: Set of stopwords to exclude. Uses defaults if None.
            use_htrg: Whether to apply HTRG (POS-based) filter. Requires torch.
            use_similarity: Whether to apply similarity filter. Requires spacy.
            pos_model: HuggingFace model name for POS tagging.
            spacy_model: spaCy model name for embeddings.
            device: Device for neural models ('cuda', 'cpu', or None for auto).
        """
        self.min_shared_words = min_shared_words
        self.min_complura = min_complura
        self.max_distance = max_distance
        self.similarity_threshold = similarity_threshold
        self.stopwords = stopwords if stopwords is not None else DEFAULT_STOPWORDS.copy()
        self.use_htrg = use_htrg
        self.use_similarity = use_similarity
        self.pos_model_name = pos_model
        self.spacy_model_name = spacy_model
        
        # Determine device
        if device is None:
            if TORCH_AVAILABLE and torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        # Lazy-loaded models
        self._pos_model = None
        self._pos_tokenizer = None
        self._spacy_nlp = None
        
        # Validate optional dependencies
        if self.use_htrg and not TORCH_AVAILABLE:
            raise ImportError("HTRG filter requires torch and transformers. "
                            "Install with: pip install torch transformers")
        if self.use_similarity and not SPACY_AVAILABLE:
            raise ImportError("Similarity filter requires spacy. "
                            "Install with: pip install spacy && python -m spacy download la_core_web_lg")
    
    # ============== PUBLIC API ==============
    
    def run(
        self,
        query: Document,
        source: Document,
        *,
        top_k: Optional[int] = None,
        query_genre: str = "prose",
        source_genre: str = "poetry",
        threshold: float = 0.5,
    ) -> FullDict:
        """
        Run the rule-based pipeline on query and source documents.
        
        Args:
            query: Query document (text being analyzed for intertextuality).
            source: Source document (potential origin of quotations).
            top_k: Maximum matches per query (None = no limit). For API compatibility.
            query_genre: Genre of query ('prose' or 'poetry').
            source_genre: Genre of source ('prose' or 'poetry').
            threshold: Not used (included for API compatibility).
        
        Returns:
            FullDict mapping query segment IDs to lists of
            (source_segment, similarity, probability) tuples.
        """
        # Convert documents to internal format
        source_list = self._document_to_list(source)
        query_list = self._document_to_list(query)
        
        # Preprocess texts
        source_processed = self._preprocess(source_list, source_genre)
        query_processed = self._preprocess(query_list, query_genre)
        
        # Run text matching
        matches, complura_matches = self._compare_texts(
            source_processed, query_processed
        )
        
        # Apply scissa filter
        matches = self._apply_scissa(matches)
        
        # Apply optional HTRG filter
        if self.use_htrg:
            matches = self._apply_htrg(matches)
        
        # Apply optional similarity filter
        if self.use_similarity:
            matches = self._apply_similarity(matches)
        
        # Combine matches and complura matches
        all_matches = self._combine_matches(matches, complura_matches)
        
        # Convert to FullDict format
        results = self._matches_to_fulldict(all_matches, source, query)
        
        # Apply top_k limit if specified
        if top_k is not None:
            for qid in results:
                results[qid] = results[qid][:top_k]
        
        return results
    
    def load_stopwords(self, filepath: Union[str, Path]) -> None:
        """
        Load stopwords from a file (one word per line).
        
        Args:
            filepath: Path to stopwords file.
        """
        filepath = Path(filepath)
        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
            for row in reader:
                if row:
                    self.stopwords.add(row[0].strip().lower())
    
    # ============== DOCUMENT CONVERSION ==============
    
    def _document_to_list(self, doc: Document) -> List[List[str]]:
        """Convert Document to internal list format [[id, text], ...]."""
        return [[str(seg.id), seg.text] for seg in doc]
    
    def _matches_to_fulldict(
        self,
        matches: List[List[Any]],
        source: Document,
        query: Document,
    ) -> FullDict:
        """Convert internal matches to FullDict format."""
        results: FullDict = {str(seg.id): [] for seg in query}
        source_segments = {str(seg.id): seg for seg in source}
        
        for match in matches:
            # Match format: [idx, source_id, source_text, query_id, query_text, shared, ...]
            if len(match) >= 4:
                source_id = str(match[1])
                query_id = str(match[3])
                
                if source_id in source_segments and query_id in results:
                    # Calculate a score based on number of shared words
                    shared_words = match[5].split(";") if len(match) > 5 else []
                    score = min(len(shared_words) / 5.0, 1.0)  # Normalize to [0, 1]
                    
                    results[query_id].append((
                        source_segments[source_id],
                        score,  # similarity score
                        1.0,    # probability (always positive for matches)
                    ))
        
        return results
    
    # ============== TEXT PREPROCESSING ==============
    
    def _preprocess(
        self,
        text_list: List[List[str]],
        genre: str,
    ) -> List[List[str]]:
        """Apply preprocessing based on genre."""
        # Assimilation (prefix transformations)
        text_list = self._assimilate(text_list)
        
        # Phrasing based on genre
        if genre == "poetry":
            text_list = self._phrasing_poetry(text_list)
        else:
            text_list = self._phrasing_prose(text_list)
        
        return text_list
    
    def _assimilate(self, text_list: List[List[str]]) -> List[List[str]]:
        """Apply assimilation rules to normalize prefixes."""
        for item in text_list:
            if len(item) > 1:
                tokens = self._tokenize(item[1])
                for i, token in enumerate(tokens):
                    transformed = self._transform_token(token)
                    if transformed != token:
                        item[1] = item[1].replace(token, transformed)
        return text_list
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words and punctuation."""
        return re.findall(r'\w+|[^\w\s]', text, re.UNICODE)
    
    def _transform_token(self, token: str) -> str:
        """Apply prefix assimilation rules."""
        prefix_map = {
            'adt': 'att', 'Adt': 'Att', 'adp': 'app', 'Adp': 'App',
            'adc': 'acc', 'Adc': 'Acc', 'adg': 'agg', 'Adg': 'Agg',
            'adf': 'aff', 'Adf': 'Aff', 'adl': 'all', 'Adl': 'All',
            'adr': 'arr', 'Adr': 'Arr', 'ads': 'ass', 'Ads': 'Ass',
            'adqu': 'acqu', 'Adqu': 'Acqu', 'inm': 'imm', 'Inm': 'Imm',
            'inl': 'ill', 'Inl': 'Ill', 'inr': 'irr', 'Inr': 'Irr',
            'inb': 'imb', 'Inb': 'Imb', 'conm': 'comm', 'Conm': 'Comm',
            'conl': 'coll', 'Conl': 'Coll', 'conr': 'corr', 'Conr': 'Corr',
            'conb': 'comb', 'Conb': 'Comb', 'conp': 'comp', 'Conp': 'Comp',
        }
        vowels = 'aeiou'
        
        for prefix, replacement in prefix_map.items():
            if token.lower().startswith(prefix):
                if prefix.lower() == 'ads' and len(token) > 3:
                    if token[3].lower() in vowels:
                        return token[0] + replacement[1:] + token[3:]
                    else:
                        return token[0] + token[2:]
                elif prefix.lower() != 'ads':
                    return token[0] + replacement[1:] + token[len(prefix):]
                break
        return token
    
    # ============== PHRASING ==============
    
    def _normalize_quotation_marks(self, text_list: List[List[str]]) -> List[List[str]]:
        """Normalize quotation marks to standard apostrophe."""
        # Match various quote characters: " " " „ ' ' (using Unicode escapes)
        quote_pattern = '[\u0022\u201c\u201d\u201e\u0027\u2018\u2019]'
        return [[item[0], re.sub(quote_pattern, "'", item[1])] for item in text_list if len(item) > 1]
    
    def _remove_whitespace_connectors(self, text_list: List[List[str]]) -> List[List[str]]:
        """Connect -que and -ve with preceding words."""
        return [[item[0], re.sub(r" (que|ve|ue)([ ,\.!?])", r"\1\2", item[1])] 
                for item in text_list if len(item) > 1]
    
    def _strip_whitespaces(self, text_list: List[List[str]]) -> List[List[str]]:
        """Remove leading/trailing whitespace."""
        return [[item[0], item[1].strip()] for item in text_list if len(item) > 1]
    
    def _cleanup(self, text_list: List[List[str]]) -> List[List[str]]:
        """Remove empty entries and strip whitespace."""
        return [[item[0], item[1].strip()] 
                for item in text_list 
                if len(item) > 1 and len(item[1].strip()) > 1]
    
    def _phrasing_prose(self, text_list: List[List[str]]) -> List[List[str]]:
        """Apply prose-specific phrasing rules."""
        text_list = self._normalize_quotation_marks(text_list)
        text_list = self._remove_whitespace_connectors(text_list)
        text_list = self._strip_whitespaces(text_list)
        text_list = self._cleanup(text_list)
        return text_list
    
    def _phrasing_poetry(self, text_list: List[List[str]]) -> List[List[str]]:
        """Apply poetry-specific phrasing rules."""
        # Mark verse endings
        text_list = [[item[0], item[1] + ' /'] for item in text_list if len(item) > 1]
        text_list = self._normalize_quotation_marks(text_list)
        text_list = self._remove_whitespace_connectors(text_list)
        text_list = self._strip_whitespaces(text_list)
        text_list = self._cleanup(text_list)
        return text_list
    
    # ============== TEXT MATCHING ==============
    
    def _compare_texts(
        self,
        source_list: List[List[str]],
        target_list: List[List[str]],
    ) -> Tuple[List[List[Any]], List[List[Any]]]:
        """Compare texts and find matches."""
        # Separate text and metadata
        source_texts = [[item[1]] for item in source_list if len(item) > 1]
        target_texts = [[item[1]] for item in target_list if len(item) > 1]
        source_meta = [item[0] for item in source_list if len(item) > 1]
        target_meta = [item[0] for item in target_list if len(item) > 1]
        
        # Normalize texts for comparison
        source_normalized = self._normalize_for_matching(source_texts)
        target_normalized = self._normalize_for_matching(target_texts)
        
        # Tokenize
        source_tokens = self._tokenize_texts(source_normalized)
        target_tokens = self._tokenize_texts(target_normalized)
        
        # Match texts
        matches, complura = self._text_matching(
            source_tokens, target_tokens,
            source_texts, target_texts,
            source_meta, target_meta,
        )
        
        # Apply distance criterion
        matches = self._apply_distance_criterion(matches)
        
        return matches, complura
    
    def _normalize_for_matching(self, text_list: List[List[str]]) -> List[List[str]]:
        """Normalize texts for matching (v→u, j→i, lowercase)."""
        trans = str.maketrans("vjVJ", "uiUI")
        result = []
        for sublist in text_list:
            normalized = []
            for text in sublist:
                text = text.translate(trans).lower().strip()
                text = ' ' + text + ' '
                # Remove typography artifacts
                text = re.sub(r'\xe2\x80\x9c|\xe2\x80\x9d|\x9c|\x9d|\xef\xbb\xbf', '', text)
                normalized.append(text)
            result.append(normalized)
        return result
    
    def _tokenize_texts(self, text_list: List[List[str]]) -> List[List[str]]:
        """Tokenize normalized texts."""
        result = []
        for sublist in text_list:
            for text in sublist:
                tokens = self._simple_tokenize(text)
                result.append(tokens)
        return result
    
    def _simple_tokenize(self, text: str) -> List[str]:
        """Simple tokenizer for matching."""
        separators = ' —?!-,.()[]:;\'/""\„'
        word = ''
        tokens = []
        for char in text + '.':
            if char not in separators:
                word += char
            elif word:
                tokens.append(word)
                word = ''
        return tokens
    
    def _text_matching(
        self,
        source_tokens: List[List[str]],
        target_tokens: List[List[str]],
        source_texts: List[List[str]],
        target_texts: List[List[str]],
        source_meta: List[str],
        target_meta: List[str],
    ) -> Tuple[List[List[Any]], List[List[Any]]]:
        """Find matching passages between source and target."""
        matches = []
        complura_matches = []
        count = 0
        
        for src_idx, src_tokens in enumerate(source_tokens):
            for tgt_idx, tgt_tokens in enumerate(target_tokens):
                matched_items = []
                complura_items = []
                
                for pos_src, token_src in enumerate(src_tokens):
                    for pos_tgt, token_tgt in enumerate(tgt_tokens):
                        if token_src == token_tgt and token_src:
                            complura_items.append((token_src, pos_src, pos_tgt))
                            if token_src not in matched_items:
                                if token_src.lower() not in self.stopwords:
                                    matched_items.append(token_src)
                
                # Check for complura matches (adjacent sequences)
                if len(complura_items) >= self.min_complura:
                    indices_src = [e[1] for e in complura_items]
                    indices_tgt = [e[2] for e in complura_items]
                    seq_src = self._find_adjacent_sequence(indices_src)
                    seq_tgt = self._find_adjacent_sequence(indices_tgt)
                    
                    if len(seq_src) >= self.min_complura and len(seq_tgt) >= self.min_complura:
                        shared = [src_tokens[i] for i in seq_src]
                        src_text = ' ' + source_texts[src_idx][0] + ' '
                        tgt_text = ' ' + target_texts[tgt_idx][0] + ' '
                        src_text = self._highlight_words(src_text, shared)
                        tgt_text = self._highlight_words(tgt_text, shared)
                        shared_str = '; '.join(shared)
                        
                        complura_matches.append([
                            len(complura_matches) + 1,
                            source_meta[src_idx], src_text,
                            target_meta[tgt_idx], tgt_text,
                            shared_str
                        ])
                
                # Regular matches
                if len(matched_items) >= self.min_shared_words:
                    count += 1
                    src_text = ' ' + source_texts[src_idx][0] + ' '
                    tgt_text = ' ' + target_texts[tgt_idx][0] + ' '
                    src_text = self._highlight_words(src_text, matched_items)
                    tgt_text = self._highlight_words(tgt_text, matched_items)
                    shared_str = '; '.join(matched_items)
                    
                    matches.append([
                        count,
                        source_meta[src_idx], src_text,
                        target_meta[tgt_idx], tgt_text,
                        shared_str
                    ])
        
        return matches, complura_matches
    
    def _find_adjacent_sequence(self, indices: List[int]) -> List[int]:
        """Find the longest sequence of adjacent indices."""
        if not indices:
            return []
        
        sorted_indices = sorted(set(indices))
        longest = []
        current = [sorted_indices[0]]
        
        for i in range(1, len(sorted_indices)):
            if sorted_indices[i] == sorted_indices[i-1] + 1:
                current.append(sorted_indices[i])
            else:
                if len(current) > len(longest):
                    longest = current
                current = [sorted_indices[i]]
        
        if len(current) > len(longest):
            longest = current
        
        return longest if len(longest) >= 4 else []
    
    def _highlight_words(self, text: str, words: List[str]) -> str:
        """Highlight matched words with **markers**."""
        separators = ' —?!-,.()[]:;\'/""\„'
        current_word = ''
        result = ''
        
        for char in text:
            if char not in separators:
                current_word += char
            else:
                if current_word.lower() in [w.lower() for w in words]:
                    result += '**' + current_word + '**'
                else:
                    result += current_word
                result += char
                current_word = ''
        
        if current_word:
            if current_word.lower() in [w.lower() for w in words]:
                result += '**' + current_word + '**'
            else:
                result += current_word
        
        return result.replace('****', '**').strip()
    
    # ============== FILTERS ==============
    
    def _apply_distance_criterion(self, matches: List[List[Any]]) -> List[List[Any]]:
        """Filter matches where shared words are too far apart."""
        filtered = []
        
        for match in matches:
            text_src = match[2]
            text_tgt = match[4]
            
            tokens_src = [t for t in self._simple_tokenize(text_src) 
                         if t.lower() not in self.stopwords]
            tokens_tgt = [t for t in self._simple_tokenize(text_tgt) 
                         if t.lower() not in self.stopwords]
            
            shared_src = [(t, i) for i, t in enumerate(tokens_src) if t.startswith("**")]
            shared_tgt = [(t, i) for i, t in enumerate(tokens_tgt) if t.startswith("**")]
            
            dist_src = self._min_distance(shared_src)
            dist_tgt = self._min_distance(shared_tgt)
            
            if dist_src <= self.max_distance and dist_tgt <= self.max_distance:
                filtered.append(match)
        
        return filtered
    
    def _min_distance(self, shared_tokens: List[Tuple[str, int]]) -> float:
        """Calculate minimum distance between shared token indices."""
        if len(shared_tokens) < 2:
            return float('inf')
        indices = [idx for _, idx in shared_tokens]
        distances = [abs(indices[i] - indices[j]) 
                    for i in range(len(indices)) 
                    for j in range(i + 1, len(indices))]
        return min(distances) if distances else float('inf')
    
    def _apply_scissa(self, matches: List[List[Any]]) -> List[List[Any]]:
        """Filter based on punctuation agreement between shared words."""
        filtered = []
        
        for match in matches:
            shared = match[5].split(";")
            shared = [s.strip() for s in shared]
            
            if len(shared) == 2:
                text_src = match[2]
                text_tgt = match[4]
                
                substr_src = self._extract_substrings(text_src, shared)
                substr_tgt = self._extract_substrings(text_tgt, shared)
                
                if substr_src and substr_tgt:
                    commas_match = self._compare_punctuation(substr_src, substr_tgt, ',')
                    semicolons_match = self._compare_punctuation(substr_src, substr_tgt, ';')
                    colons_match = self._compare_punctuation(substr_src, substr_tgt, ':')
                    
                    if all(commas_match) and all(semicolons_match) and all(colons_match):
                        filtered.append(match)
                else:
                    filtered.append(match)
            elif len(shared) >= 3:
                filtered.append(match)
        
        return filtered
    
    def _extract_substrings(self, text: str, shared: List[str]) -> List[str]:
        """Extract substrings between highlighted shared words."""
        collection = []
        text_lower = text.lower()
        shared = [s.lower().strip() for s in shared]
        
        marker1 = '**' + shared[0] + '**'
        marker2 = '**' + shared[1] + '**'
        
        if marker1 in text_lower and marker2 in text_lower:
            idx1 = text_lower.index(marker1)
            idx2 = text_lower.index(marker2)
            
            if idx1 < idx2:
                start = idx1 + len(marker1)
                end = idx2
            else:
                start = idx2 + len(marker2)
                end = idx1
            
            collection.append(text[start:end])
        
        return collection
    
    def _compare_punctuation(
        self,
        substr_src: List[str],
        substr_tgt: List[str],
        punct: str
    ) -> List[bool]:
        """Compare punctuation counts between substrings."""
        results = []
        for s1 in substr_src:
            for s2 in substr_tgt:
                results.append(s1.count(punct) == s2.count(punct))
        return results if results else [True]
    
    def _combine_matches(
        self,
        matches: List[List[Any]],
        complura: List[List[Any]]
    ) -> List[List[Any]]:
        """Combine regular matches with complura matches, avoiding duplicates."""
        combined = list(matches)
        existing_pairs = {(m[1], m[3]) for m in matches}
        
        for c in complura:
            if (c[1], c[3]) not in existing_pairs:
                combined.append(c)
                existing_pairs.add((c[1], c[3]))
        
        return combined
    
    # ============== OPTIONAL FILTERS ==============
    
    def _load_pos_model(self) -> None:
        """Load the POS tagging model (lazy loading)."""
        if self._pos_model is None and TORCH_AVAILABLE:
            self._pos_tokenizer = XLMRobertaTokenizer.from_pretrained(self.pos_model_name)
            self._pos_model = XLMRobertaForTokenClassification.from_pretrained(self.pos_model_name)
            if self.device == "cuda" and torch.cuda.is_available():
                self._pos_model = self._pos_model.to("cuda")
            self._pos_model.eval()
    
    def _load_spacy_model(self) -> None:
        """Load the spaCy model (lazy loading)."""
        if self._spacy_nlp is None and SPACY_AVAILABLE:
            self._spacy_nlp = spacy.load(self.spacy_model_name)
    
    def _apply_htrg(self, matches: List[List[Any]]) -> List[List[Any]]:
        """Apply HTRG (Part-of-Speech) filter."""
        if not TORCH_AVAILABLE:
            return matches
        
        self._load_pos_model()
        
        # Valid POS tag combinations
        valid_grammar = {
            ('NOUN', 'VERB'), ('VERB', 'NOUN'), ('NOUN', 'NOUN'),
            ('VERB', 'VERB'), ('PROPN', 'NOUN'), ('PROPN', 'VERB'),
            ('NOUN', 'PROPN'), ('VERB', 'PROPN'), ('PROPN', 'PROPN'),
        }
        
        filtered = []
        for match in matches:
            shared = [w.strip() for w in match[5].split(";")]
            text_src = match[2].replace('**', ' ')
            text_tgt = match[4].replace('**', ' ')
            
            tags_src = self._tag_text(text_src)
            tags_tgt = self._tag_text(text_tgt)
            
            # Get POS tags for shared words
            pos_src = self._get_pos_for_words(shared, tags_src)
            pos_tgt = self._get_pos_for_words(shared, tags_tgt)
            
            # Check if any valid combination exists
            if self._has_valid_grammar(pos_src, pos_tgt, valid_grammar):
                match.append(pos_src)
                match.append(pos_tgt)
                match.append('in')
                filtered.append(match)
        
        return filtered
    
    def _tag_text(self, text: str) -> List[Tuple[str, str]]:
        """Tag text with POS using the loaded model."""
        if self._pos_model is None:
            return []
        
        # Preprocess
        for punct in string.punctuation:
            text = text.replace(punct, f' {punct} ')
        text = " et " + text  # Dummy word for first token
        
        tokens = self._pos_tokenizer.tokenize(text)
        if len(tokens) > 512:
            return []
        
        input_ids = self._pos_tokenizer.convert_tokens_to_ids(tokens)
        inputs = torch.tensor([input_ids])
        
        if self.device == "cuda" and torch.cuda.is_available():
            inputs = inputs.to("cuda")
        
        with torch.no_grad():
            outputs = self._pos_model(inputs)
            predictions = torch.argmax(outputs.logits, dim=2).squeeze().tolist()
        
        id2label = {
            0: "", 1: "ADJ", 2: "ADP", 3: "ADV", 4: "AUX", 5: "CCONJ",
            6: "DET", 7: "INTJ", 8: "NOUN", 9: "NUM", 10: "PART",
            11: "PRON", 12: "PROPN", 13: "PUNCT", 14: "SCONJ", 15: "VERB",
            16: "X", 17: "O"
        }
        
        token_tags = [id2label.get(id, "") for id in predictions]
        grouped = self._group_subwords(tokens)
        
        result = []
        idx = 0
        for word in grouped:
            subtokens = self._pos_tokenizer.tokenize(word)
            if idx < len(token_tags):
                result.append((word.lower(), token_tags[idx]))
            idx += len(subtokens)
        
        return result
    
    def _group_subwords(self, tokens: List[str]) -> List[str]:
        """Group subword tokens back into words."""
        grouped = []
        current = ""
        for token in tokens:
            if token.startswith("▁"):
                if current:
                    grouped.append(current)
                current = token[1:]
            else:
                current += token
        if current:
            grouped.append(current)
        return grouped
    
    def _get_pos_for_words(
        self,
        words: List[str],
        word_tags: List[Tuple[str, str]]
    ) -> List[List[str]]:
        """Get POS tags for specific words."""
        tokens = [t[0] for t in word_tags]
        tags = [t[1] for t in word_tags]
        
        result = []
        for word in words:
            word_lower = word.lower()
            indices = [i for i, t in enumerate(tokens) if t == word_lower]
            word_tags_found = [tags[i] for i in indices if i < len(tags)]
            result.append(word_tags_found if word_tags_found else [""])
        
        return result
    
    def _has_valid_grammar(
        self,
        pos_src: List[List[str]],
        pos_tgt: List[List[str]],
        valid: Set[Tuple[str, str]]
    ) -> bool:
        """Check if any valid grammar combination exists."""
        combos_src = list(itertools.product(*pos_src)) if pos_src else []
        combos_tgt = list(itertools.product(*pos_tgt)) if pos_tgt else []
        
        shared_combos = set(combos_src).intersection(set(combos_tgt))
        
        for combo in shared_combos:
            pairs = set(combinations(combo, 2))
            if valid.intersection(pairs):
                return True
        
        return False
    
    def _apply_similarity(self, matches: List[List[Any]]) -> List[List[Any]]:
        """Apply semantic similarity filter."""
        if not SPACY_AVAILABLE:
            return matches
        
        self._load_spacy_model()
        
        filtered = []
        for match in matches:
            shared = [w.strip() for w in match[5].split(";")]
            
            if len(shared) == 2:
                doc1 = self._spacy_nlp(shared[0])
                doc2 = self._spacy_nlp(shared[1])
                
                v1 = doc1.vector
                v2 = doc2.vector
                
                if np.any(v1) and np.any(v2):
                    sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                    sim = round(float(sim), 2)
                    match.append(sim)
                    
                    if sim <= self.similarity_threshold:
                        filtered.append(match)
                else:
                    filtered.append(match)
            else:
                filtered.append(match)
        
        return filtered
