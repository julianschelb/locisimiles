# training/lexical/features.py
"""Shared feature engineering for the lexical (LogReg/GBDT) classifier baseline.

Used identically at training time (:class:`LexicalClassifierTrainer`) and at
inference time (:class:`LexicalClassifierJudge`) so the two never drift apart.
Ports the per-pair similarity features used to evaluate the benchmark: TF-IDF
cosine over lemma unigrams, lemma unigrams+bigrams, and character 3-4-grams,
plus lemma Jaccard overlap, raw-token overlap, and length features.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from locisimiles.pipeline.generator._latin_text import NgramAnalyzer, preprocess

DEFAULT_WORD_MAX_FEATURES = 50_000
DEFAULT_CHAR_MAX_FEATURES = 100_000

FEATURE_NAMES: list[str] = [
    "tfidf_cos_lemma_1",
    "tfidf_cos_lemma_12",
    "tfidf_cos_char_34",
    "jaccard_lemma",
    "overlap_count",
    "overlap_count_norm",
    "len_q",
    "len_c",
    "len_absdiff",
    "len_ratio",
]


# =============================================================================
# Vectorizer fitting
# =============================================================================


def fit_vectorizers(
    texts: list[str],
    *,
    lemmatize: bool,
    lowercase: bool,
    word_max_features: int = DEFAULT_WORD_MAX_FEATURES,
    char_max_features: int = DEFAULT_CHAR_MAX_FEATURES,
) -> dict[str, Any]:
    """Fit the three TF-IDF vectorizers on the training texts (query + corpus pooled)."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    token_lists = [preprocess(text, lemmatize=lemmatize, lowercase=lowercase) for text in texts]

    lemma_1 = TfidfVectorizer(
        analyzer=NgramAnalyzer((1, 1)),
        max_features=word_max_features,
        min_df=2,
        sublinear_tf=True,
        norm="l2",
    )
    lemma_1.fit(token_lists)

    lemma_12 = TfidfVectorizer(
        analyzer=NgramAnalyzer((1, 2)),
        max_features=word_max_features,
        min_df=2,
        sublinear_tf=True,
        norm="l2",
    )
    lemma_12.fit(token_lists)

    char_34 = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 4),
        lowercase=True,
        max_features=char_max_features,
        min_df=2,
        sublinear_tf=True,
        norm="l2",
    )
    char_34.fit(texts)

    return {
        "tfidf_cos_lemma_1": lemma_1,
        "tfidf_cos_lemma_12": lemma_12,
        "tfidf_cos_char_34": char_34,
    }


# =============================================================================
# Feature matrix
# =============================================================================


def _row_cosine(matrix_a: Any, matrix_b: Any) -> np.ndarray:
    """Element-wise cosine similarity between rows of two L2-normalized sparse matrices."""
    product = matrix_a.multiply(matrix_b).sum(axis=1)
    return np.asarray(product).ravel()


def build_feature_matrix(
    query_texts: list[str],
    corpus_texts: list[str],
    vectorizers: dict[str, Any],
    *,
    lemmatize: bool,
    lowercase: bool,
) -> np.ndarray:
    """Build the per-pair feature matrix, in the fixed order given by ``FEATURE_NAMES``."""
    q_tokens = [preprocess(text, lemmatize=lemmatize, lowercase=lowercase) for text in query_texts]
    c_tokens = [preprocess(text, lemmatize=lemmatize, lowercase=lowercase) for text in corpus_texts]

    q_lemma_1 = vectorizers["tfidf_cos_lemma_1"].transform(q_tokens)
    c_lemma_1 = vectorizers["tfidf_cos_lemma_1"].transform(c_tokens)
    q_lemma_12 = vectorizers["tfidf_cos_lemma_12"].transform(q_tokens)
    c_lemma_12 = vectorizers["tfidf_cos_lemma_12"].transform(c_tokens)
    q_char = vectorizers["tfidf_cos_char_34"].transform(query_texts)
    c_char = vectorizers["tfidf_cos_char_34"].transform(corpus_texts)

    tfidf_cos_lemma_1 = _row_cosine(q_lemma_1, c_lemma_1)
    tfidf_cos_lemma_12 = _row_cosine(q_lemma_12, c_lemma_12)
    tfidf_cos_char_34 = _row_cosine(q_char, c_char)

    n = len(query_texts)
    jaccard_lemma = np.zeros(n, dtype=np.float64)
    overlap_count = np.zeros(n, dtype=np.float64)
    overlap_count_norm = np.zeros(n, dtype=np.float64)
    for i, (a, b) in enumerate(zip(q_tokens, c_tokens)):
        sa, sb = set(a), set(b)
        union = sa | sb
        jaccard_lemma[i] = (len(sa & sb) / len(union)) if union else 0.0
        ca, cb = Counter(a), Counter(b)
        overlap = sum((ca & cb).values())
        overlap_count[i] = overlap
        denom = min(len(a), len(b)) or 1
        overlap_count_norm[i] = overlap / denom

    len_q = np.array([len(s) for s in query_texts], dtype=np.float64)
    len_c = np.array([len(s) for s in corpus_texts], dtype=np.float64)
    len_absdiff = np.abs(len_q - len_c)
    len_ratio = np.minimum(len_q, len_c) / np.maximum(np.maximum(len_q, len_c), 1.0)

    columns = {
        "tfidf_cos_lemma_1": tfidf_cos_lemma_1,
        "tfidf_cos_lemma_12": tfidf_cos_lemma_12,
        "tfidf_cos_char_34": tfidf_cos_char_34,
        "jaccard_lemma": jaccard_lemma,
        "overlap_count": overlap_count,
        "overlap_count_norm": overlap_count_norm,
        "len_q": len_q,
        "len_c": len_c,
        "len_absdiff": len_absdiff,
        "len_ratio": len_ratio,
    }
    return np.column_stack([columns[name] for name in FEATURE_NAMES])
