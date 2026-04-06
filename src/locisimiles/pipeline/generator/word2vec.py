"""Word2Vec-based candidate generator inspired by Burns et al. (2021)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable, Optional, Protocol, Sequence

from locisimiles.document import Document, TextSegment
from locisimiles.pipeline._types import Candidate, CandidateGeneratorOutput
from locisimiles.pipeline.generator._base import CandidateGeneratorBase

DEFAULT_WORD2VEC_MODEL_PATH = (
    Path(__file__).resolve().parents[4] / "models" / "latin_w2v_bamman_lemma300_100_1.model"
)


class _KeyedVectorsLike(Protocol):
    """Protocol for the subset of KeyedVectors used by this generator."""

    def similarity(self, w1: str, w2: str) -> float:
        ...

    def __contains__(self, key: str) -> bool:
        ...


class _Word2VecLike(Protocol):
    """Protocol for the subset of gensim Word2Vec used by this generator."""

    @property
    def wv(self) -> _KeyedVectorsLike:
        ...


def _load_word2vec_model(model_path: Path) -> _Word2VecLike:
    """Load a Word2Vec model with a lazy gensim import."""
    try:
        from gensim.models import Word2Vec
    except ImportError as exc:
        raise ImportError(
            "Word2Vec support requires gensim. Install it with: pip install 'locisimiles[word2vec]'"
        ) from exc

    return Word2Vec.load(str(model_path))


class Word2VecCandidateGenerator(CandidateGeneratorBase):
    """Generate candidates with pair-aware bigram similarity.

    The generator compares query/source bigrams and scores each source segment
    by its best matching bigram pair. Similarities are mapped from cosine
    ``[-1, 1]`` to ``[0, 1]`` for consistency with existing threshold UX.

    Args:
        model_path: Path to a local gensim ``.model`` file.
        interval: Maximum gap between two tokens inside a bigram.
            ``0`` means contiguous bigrams.
        order_free: If ``True``, sort each bigram token pair so order is ignored.
    """

    def __init__(
        self,
        *,
        model_path: str | Path = DEFAULT_WORD2VEC_MODEL_PATH,
        interval: int = 0,
        order_free: bool = False,
    ):
        self.model_path = Path(model_path)
        self.interval = max(0, int(interval))
        self.order_free = bool(order_free)

        if not self.model_path.exists():
            raise FileNotFoundError(
                "Word2Vec model not found at "
                f"{self.model_path}. Provide a valid path via 'model_path'."
            )

        self.model = _load_word2vec_model(self.model_path)

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Apply lightweight Latin-oriented normalization for token matching."""
        text = text.lower().replace("j", "i").replace("v", "u")
        text = re.sub(r"\d+", " ", text)
        text = re.sub(r"[^a-z\s]", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    def _tokenize(self, text: str) -> list[str]:
        normalized = self._normalize_text(text)
        if not normalized:
            return []
        return [token for token in normalized.split(" ") if token]

    def _build_bigrams(self, tokens: Sequence[str], interval: int) -> list[tuple[str, str]]:
        """Build interval-constrained bigrams from tokens."""
        if len(tokens) < 2:
            return []

        bigrams: list[tuple[str, str]] = []
        for i in range(len(tokens) - 1):
            max_j = min(len(tokens), i + interval + 2)
            for j in range(i + 1, max_j):
                left, right = tokens[i], tokens[j]
                if self.order_free and right < left:
                    left, right = right, left
                bigrams.append((left, right))
        return bigrams

    def _word_similarity(self, left: str, right: str) -> Optional[float]:
        """Return normalized cosine similarity in [0, 1], or None for OOV."""
        vectors = self.model.wv
        if left not in vectors or right not in vectors:
            return None
        cosine = float(vectors.similarity(left, right))
        return max(0.0, min(1.0, (cosine + 1.0) / 2.0))

    def _bigram_pair_score(
        self,
        query_bigram: tuple[str, str],
        source_bigram: tuple[str, str],
    ) -> Optional[float]:
        """Score a query/source bigram pair with pair-aware alignment."""
        q1, q2 = query_bigram
        s1, s2 = source_bigram

        q1s1 = self._word_similarity(q1, s1)
        q1s2 = self._word_similarity(q1, s2)
        q2s1 = self._word_similarity(q2, s1)
        q2s2 = self._word_similarity(q2, s2)

        direct = None
        if q1s1 is not None and q2s2 is not None:
            direct = (q1s1 + q2s2) / 2.0

        crossed = None
        if q1s2 is not None and q2s1 is not None:
            crossed = (q1s2 + q2s1) / 2.0

        if direct is None and crossed is None:
            return None
        if direct is None:
            return crossed
        if crossed is None:
            return direct
        return max(direct, crossed)

    def _segment_score(
        self,
        query_bigrams: Sequence[tuple[str, str]],
        source_bigrams: Sequence[tuple[str, str]],
    ) -> float:
        """Return the best available bigram-pair score for one segment pair."""
        best = 0.0
        for q_bigram in query_bigrams:
            for s_bigram in source_bigrams:
                pair_score = self._bigram_pair_score(q_bigram, s_bigram)
                if pair_score is not None and pair_score > best:
                    best = pair_score
        return best

    def _precompute_source_bigrams(
        self,
        source_segments: Iterable[TextSegment],
        interval: int,
    ) -> dict[str, list[tuple[str, str]]]:
        """Precompute source bigrams for reuse across all query segments."""
        precomputed: dict[str, list[tuple[str, str]]] = {}
        for segment in source_segments:
            tokens = self._tokenize(segment.text)
            precomputed[str(segment.id)] = self._build_bigrams(tokens, interval)
        return precomputed

    def generate(
        self,
        *,
        query: Document,
        source: Document,
        top_k: int = 100,
        interval: int | None = None,
        order_free: bool | None = None,
        **kwargs: Any,
    ) -> CandidateGeneratorOutput:
        """Generate top-k Word2Vec candidates for each query segment."""
        if order_free is not None:
            self.order_free = bool(order_free)
        eff_interval = self.interval if interval is None else max(0, int(interval))
        eff_top_k = max(1, int(top_k))

        source_segments = list(source.segments.values())
        source_bigrams = self._precompute_source_bigrams(source_segments, interval=eff_interval)

        results: CandidateGeneratorOutput = {}
        for query_segment in query.segments.values():
            query_tokens = self._tokenize(query_segment.text)
            query_bigrams = self._build_bigrams(query_tokens, eff_interval)

            scored: list[Candidate] = []
            if query_bigrams:
                for src_segment in source_segments:
                    score = self._segment_score(query_bigrams, source_bigrams[str(src_segment.id)])
                    scored.append(Candidate(segment=src_segment, score=score))

            scored.sort(key=lambda item: item.score, reverse=True)
            results[query_segment.id] = scored[:eff_top_k]

        return results
