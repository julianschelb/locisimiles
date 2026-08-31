"""Vocabulary-coverage diagnostics for embedding-based generators.

A checkpoint whose vocabulary does not match the text it is applied to fails
quietly: lookups miss, scores degrade, and nothing raises. Reporting coverage
at construction time turns that silent failure into a visible one.
"""

from __future__ import annotations

import warnings
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Protocol, Sequence

__all__ = ["CoverageReport", "SupportsContains", "vocab_coverage", "warn_on_low_coverage"]


class SupportsContains(Protocol):
    """Anything that answers ``token in vocabulary`` for string tokens.

    Deliberately narrower than ``Container[str]``: gensim's ``KeyedVectors``
    declares ``__contains__(self, key: str)``, which is not compatible with
    ``Container``'s ``__contains__(self, object)``.
    """

    def __contains__(self, key: str, /) -> bool: ...


#: Coverage below which :func:`warn_on_low_coverage` emits a warning.
DEFAULT_MIN_TOKEN_COVERAGE = 0.70


@dataclass(frozen=True)
class CoverageReport:
    """Coverage of a token stream against a vocabulary.

    Attributes:
        n_tokens: Number of running tokens inspected.
        n_tokens_in_vocab: Running tokens found in the vocabulary.
        n_types: Number of distinct tokens inspected.
        n_types_in_vocab: Distinct tokens found in the vocabulary.
        top_oov: Most frequent out-of-vocabulary tokens with their counts.
    """

    n_tokens: int
    n_tokens_in_vocab: int
    n_types: int
    n_types_in_vocab: int
    top_oov: tuple[tuple[str, int], ...]

    @property
    def token_coverage(self) -> float:
        """Share of running tokens present in the vocabulary."""
        return self.n_tokens_in_vocab / self.n_tokens if self.n_tokens else 0.0

    @property
    def type_coverage(self) -> float:
        """Share of distinct tokens present in the vocabulary."""
        return self.n_types_in_vocab / self.n_types if self.n_types else 0.0

    def summary(self) -> str:
        """One-line human-readable summary."""
        oov = ", ".join(f"{w} ({c})" for w, c in self.top_oov[:5])
        return (
            f"tokens {self.n_tokens_in_vocab:,}/{self.n_tokens:,} ({self.token_coverage:.1%}), "
            f"types {self.n_types_in_vocab:,}/{self.n_types:,} ({self.type_coverage:.1%})"
            + (f"; frequent OOV: {oov}" if oov else "")
        )


def vocab_coverage(
    tokens: Iterable[str], vocabulary: SupportsContains, *, top_n: int = 15
) -> CoverageReport:
    """Measure how much of ``tokens`` the ``vocabulary`` covers.

    Args:
        tokens: Running tokens, in the same form used for lookup.
        vocabulary: Any container supporting ``in`` (e.g. gensim ``KeyedVectors``).
        top_n: How many frequent out-of-vocabulary tokens to report.
    """
    counts: Counter[str] = Counter()
    oov: Counter[str] = Counter()
    for token in tokens:
        counts[token] += 1
        if token not in vocabulary:
            oov[token] += 1

    n_tokens = sum(counts.values())
    return CoverageReport(
        n_tokens=n_tokens,
        n_tokens_in_vocab=n_tokens - sum(oov.values()),
        n_types=len(counts),
        n_types_in_vocab=len(counts) - len(oov),
        top_oov=tuple(oov.most_common(top_n)),
    )


def warn_on_low_coverage(
    tokens: Sequence[str],
    vocabulary: SupportsContains,
    *,
    label: str,
    min_token_coverage: float = DEFAULT_MIN_TOKEN_COVERAGE,
) -> CoverageReport:
    """Compute coverage and warn when it falls below ``min_token_coverage``.

    Low coverage usually means the text and the checkpoint disagree about
    normalization (orthography, lemmatization, or casing) rather than that the
    checkpoint is simply small.
    """
    report = vocab_coverage(tokens, vocabulary)
    if report.n_tokens and report.token_coverage < min_token_coverage:
        warnings.warn(
            f"{label}: only {report.token_coverage:.1%} of tokens are in the vocabulary "
            f"({report.summary()}). Check that tokenization, lemmatization and "
            "orthography match how the checkpoint was trained.",
            UserWarning,
            stacklevel=3,
        )
    return report
