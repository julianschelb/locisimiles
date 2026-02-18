# pipeline/rule_based.py
"""
Rule-based pipeline for lexical intertextuality detection.

Provides ``RuleBasedPipeline`` which identifies textual reuse
between Latin documents through lexical matching combined with
linguistic filters.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Set, Union

from locisimiles.pipeline.generator.rule_based import (
    SPACY_AVAILABLE,  # noqa: F401
    TORCH_AVAILABLE,  # noqa: F401
    RuleBasedCandidateGenerator,
)
from locisimiles.pipeline.judge.identity import IdentityJudge
from locisimiles.pipeline.pipeline import Pipeline


class RuleBasedPipeline(Pipeline):
    """Rule-based pipeline for lexical intertextuality detection.

    Identifies potential quotations, allusions, and textual reuse between
    Latin documents using a multi-stage rule-based approach.

    Pipeline steps:

    1. **Text preprocessing** - Normalise prefix assimilations
       (e.g. *adt-* → *att-*, *inm-* → *imm-*), quotation marks,
       and whitespace connectors.  Apply genre-specific phrasing
       rules for prose or poetry.
    2. **Text matching** - Tokenise both documents and find shared
       non-stopword tokens between every query-source segment pair.
    3. **Distance criterion** - Discard matches where shared words
       are too far apart (controlled by ``max_distance``).
    4. **Scissa filter** - Compare punctuation patterns around shared
       words to strengthen evidence of deliberate textual reuse.
    5. **HTRG filter** *(optional)* - Part-of-speech analysis using a
       HuggingFace token-classification model.  Requires ``torch``.
    6. **Similarity filter** *(optional)* - Word-embedding similarity
       check using spaCy vectors.  Requires ``spacy``.

    Args:
        min_shared_words: Minimum number of shared non-stopwords required.
        min_complura: Minimum adjacent tokens for complura detection.
        max_distance: Maximum distance between shared words.
        similarity_threshold: Threshold for semantic similarity filter.
        stopwords: Set of stopwords to exclude.  Uses defaults if ``None``.
        use_htrg: Whether to apply HTRG (POS-based) filter.  Requires torch.
        use_similarity: Whether to apply similarity filter.  Requires spacy.
        pos_model: HuggingFace model name for POS tagging.
        spacy_model: spaCy model name for embeddings.
        device: Device for neural models (``"cuda"``, ``"cpu"``, or ``None``
            for auto).

    Example:
        ```python
        from locisimiles.pipeline import RuleBasedPipeline
        from locisimiles.document import Document

        # Load documents
        query = Document("query.csv")
        source = Document("source.csv")

        # Define pipeline
        pipeline = RuleBasedPipeline()

        # Run pipeline
        results = pipeline.run(query=query, source=source)
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
        super().__init__(
            generator=RuleBasedCandidateGenerator(
                min_shared_words=min_shared_words,
                min_complura=min_complura,
                max_distance=max_distance,
                similarity_threshold=similarity_threshold,
                stopwords=stopwords,
                use_htrg=use_htrg,
                use_similarity=use_similarity,
                pos_model=pos_model,
                spacy_model=spacy_model,
                device=device,
            ),
            judge=IdentityJudge(),
        )

    # ------ Backward-compatible property accessors ------

    @property
    def min_shared_words(self) -> int:
        """Minimum number of shared non-stopwords required."""
        return self.generator.min_shared_words

    @property
    def min_complura(self) -> int:
        """Minimum adjacent tokens for complura detection."""
        return self.generator.min_complura

    @property
    def max_distance(self) -> int:
        """Maximum distance between shared words."""
        return self.generator.max_distance

    @property
    def similarity_threshold(self) -> float:
        """Threshold for semantic similarity filter."""
        return self.generator.similarity_threshold

    @property
    def stopwords(self) -> Set[str]:
        """Set of stopwords to exclude."""
        return self.generator.stopwords

    @property
    def use_htrg(self) -> bool:
        """Whether HTRG (POS-based) filter is enabled."""
        return self.generator.use_htrg

    @property
    def use_similarity(self) -> bool:
        """Whether similarity filter is enabled."""
        return self.generator.use_similarity

    @property
    def device(self) -> str:
        """Device for neural models."""
        return self.generator.device

    def load_stopwords(self, filepath: Union[str, Path]) -> None:
        """Load stopwords from a file (one word per line).

        Args:
            filepath: Path to stopwords file.
        """
        self.generator.load_stopwords(filepath)
