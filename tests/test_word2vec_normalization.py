"""Tests for Latin orthography normalization in the Word2Vec generator.

The reference checkpoints are trained on u/i-normalized lemmas, whereas CLTK
emits v/j spellings and homonym indices, so tokens must be mapped before
lookup or they silently miss the vocabulary.
"""

from __future__ import annotations

import pytest

from locisimiles.pipeline.generator.word2vec import Word2VecCandidateGenerator as W2V


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("volo", "uolo"),        # v -> u
        ("Jam", "iam"),          # j -> i, lowercased
        ("civilis", "ciuilis"),
        ("venus2", "uenus"),     # homonym index stripped
        ("cum2", "cum"),
        ("uirumque", "uirumque"),  # already normalized
        ("arma,", "arma"),       # punctuation removed
    ],
)
def test_normalizes_tokens_to_the_vector_vocabulary(raw, expected):
    assert W2V._normalize_token(raw) == expected


def test_sense_digits_are_stripped_not_split():
    """'venus2' must become one token, not 'venus' plus a stray fragment."""
    assert W2V._normalize_token("venus2") == "uenus"
    assert "2" not in W2V._normalize_token("liber4")
