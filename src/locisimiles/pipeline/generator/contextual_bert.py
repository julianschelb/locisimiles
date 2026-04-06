"""Contextual Latin-BERT-style candidate generator.

This generator performs token-level contextual similarity between query and
source segments. Each segment is encoded with a transformer model and token
embeddings are pooled to word-level vectors via offset mapping. Segment scores
are computed as the best token-token cosine match and mapped to ``[0, 1]``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from locisimiles.document import Document, TextSegment
from locisimiles.pipeline._types import Candidate, CandidateGeneratorOutput
from locisimiles.pipeline.generator._base import CandidateGeneratorBase

DEFAULT_CONTEXTUAL_BERT_MODEL_NAME = "xlm-roberta-base"

# Compact stopword list to suppress high-frequency Latin function words.
LATIN_STOPWORDS = {
    "a",
    "ab",
    "ac",
    "ad",
    "at",
    "atque",
    "aut",
    "autem",
    "cum",
    "de",
    "dum",
    "e",
    "et",
    "ex",
    "in",
    "inter",
    "nec",
    "neque",
    "non",
    "per",
    "post",
    "pro",
    "quae",
    "quam",
    "qui",
    "quo",
    "sed",
    "si",
    "sive",
    "sub",
    "super",
    "ut",
}


class LatinBertContextualCandidateGenerator(CandidateGeneratorBase):
    """Generate candidates using contextual token similarity.

    Supports either a HuggingFace model identifier (``model_name``) or a local
    model directory/path (``model_path``). Exactly one source may be provided.

    Args:
        model_name: HuggingFace model identifier.
        model_path: Local path to a model directory.
        device: Torch device string.
        max_length: Maximum tokenizer length per segment.
        min_token_length: Minimum word length to keep during filtering.
        use_stopword_filter: Whether to remove common Latin stopwords.
    """

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_CONTEXTUAL_BERT_MODEL_NAME,
        model_path: str | Path | None = None,
        device: str | int | None = None,
        max_length: int = 256,
        min_token_length: int = 2,
        use_stopword_filter: bool = True,
    ):
        self.device = device if device is not None else "cpu"
        self.max_length = max(16, int(max_length))
        self.min_token_length = max(1, int(min_token_length))
        self.use_stopword_filter = bool(use_stopword_filter)

        model_ref = self._resolve_model_reference(model_name=model_name, model_path=model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_ref)
        self.model = AutoModel.from_pretrained(model_ref)
        self.model.to(self.device).eval()

        self._source_cache: dict[str, np.ndarray] | None = None
        self._source_cache_doc_id: int | None = None

    @staticmethod
    def _resolve_model_reference(*, model_name: str, model_path: str | Path | None) -> str:
        """Resolve a single model reference from name/path inputs."""
        if model_path is not None and model_name != DEFAULT_CONTEXTUAL_BERT_MODEL_NAME:
            raise ValueError("Provide either model_name or model_path, not both.")

        if model_path is not None:
            path_obj = Path(model_path)
            if not path_obj.exists():
                raise FileNotFoundError(f"Latin BERT model path does not exist: {path_obj}")
            return str(path_obj)

        return model_name

    @staticmethod
    def _word_spans(text: str) -> list[tuple[str, int, int]]:
        """Extract word spans using a Latin-friendly regex."""
        return [(m.group(0), m.start(), m.end()) for m in re.finditer(r"[A-Za-z]+", text)]

    def _keep_token(self, token: str, min_token_length: int, use_stopword_filter: bool) -> bool:
        """Return True when a token is eligible for similarity scoring."""
        lower = token.lower()
        if len(lower) < min_token_length:
            return False
        return not (use_stopword_filter and lower in LATIN_STOPWORDS)

    def _segment_word_embeddings(
        self,
        segment: TextSegment,
        *,
        max_length: int,
        min_token_length: int,
        use_stopword_filter: bool,
    ) -> np.ndarray:
        """Encode one segment and pool subword states to word-level vectors."""
        text = segment.text.strip()
        if not text:
            return np.empty((0, 0), dtype=np.float32)

        spans = [
            (tok, start, end)
            for tok, start, end in self._word_spans(text)
            if self._keep_token(tok, min_token_length, use_stopword_filter)
        ]
        if not spans:
            return np.empty((0, 0), dtype=np.float32)

        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            return_offsets_mapping=True,
        )

        offsets = encoded.pop("offset_mapping")[0].tolist()
        encoded = {key: value.to(self.device) for key, value in encoded.items()}

        with torch.no_grad():
            hidden = self.model(**encoded).last_hidden_state[0]

        vectors: list[np.ndarray] = []
        for _tok, start, end in spans:
            indices = [
                i
                for i, (sub_start, sub_end) in enumerate(offsets)
                if sub_end > sub_start and not (sub_end <= start or sub_start >= end)
            ]
            if not indices:
                continue

            pooled = hidden[indices].mean(dim=0)
            norm = torch.linalg.norm(pooled)
            if float(norm) <= 0.0:
                continue
            vectors.append((pooled / norm).detach().cpu().numpy().astype(np.float32))

        if not vectors:
            return np.empty((0, 0), dtype=np.float32)

        return np.vstack(vectors)

    def _build_source_cache(
        self,
        source_segments: Iterable[TextSegment],
        *,
        max_length: int,
        min_token_length: int,
        use_stopword_filter: bool,
    ) -> dict[str, np.ndarray]:
        """Precompute and cache source token embeddings for one run."""
        cache: dict[str, np.ndarray] = {}
        for segment in source_segments:
            cache[str(segment.id)] = self._segment_word_embeddings(
                segment,
                max_length=max_length,
                min_token_length=min_token_length,
                use_stopword_filter=use_stopword_filter,
            )
        return cache

    @staticmethod
    def _segment_similarity(query_tokens: np.ndarray, source_tokens: np.ndarray) -> float:
        """Best token-token cosine mapped from ``[-1, 1]`` to ``[0, 1]``."""
        if query_tokens.size == 0 or source_tokens.size == 0:
            return 0.0

        cosine_matrix = query_tokens @ source_tokens.T
        max_cosine = float(np.max(cosine_matrix))
        return max(0.0, min(1.0, (max_cosine + 1.0) / 2.0))

    def generate(
        self,
        *,
        query: Document,
        source: Document,
        top_k: int = 100,
        max_length: int | None = None,
        min_token_length: int | None = None,
        use_stopword_filter: bool | None = None,
        **kwargs: Any,
    ) -> CandidateGeneratorOutput:
        """Generate top-k candidates using contextual token similarity."""
        eff_top_k = max(1, int(top_k))
        eff_max_length = self.max_length if max_length is None else max(16, int(max_length))
        eff_min_token_len = (
            self.min_token_length if min_token_length is None else max(1, int(min_token_length))
        )
        eff_stopword_filter = (
            self.use_stopword_filter if use_stopword_filter is None else bool(use_stopword_filter)
        )

        source_segments = list(source.segments.values())
        if self._source_cache is None or self._source_cache_doc_id != id(source):
            self._source_cache = self._build_source_cache(
                source_segments,
                max_length=eff_max_length,
                min_token_length=eff_min_token_len,
                use_stopword_filter=eff_stopword_filter,
            )
            self._source_cache_doc_id = id(source)

        results: CandidateGeneratorOutput = {}
        for query_segment in query.segments.values():
            query_vectors = self._segment_word_embeddings(
                query_segment,
                max_length=eff_max_length,
                min_token_length=eff_min_token_len,
                use_stopword_filter=eff_stopword_filter,
            )

            scored: list[Candidate] = []
            for source_segment in source_segments:
                source_vectors = self._source_cache.get(str(source_segment.id))
                if source_vectors is None:
                    score = 0.0
                else:
                    score = self._segment_similarity(query_vectors, source_vectors)
                scored.append(Candidate(segment=source_segment, score=score))

            scored.sort(key=lambda c: c.score, reverse=True)
            results[str(query_segment.id)] = scored[:eff_top_k]

        return results
