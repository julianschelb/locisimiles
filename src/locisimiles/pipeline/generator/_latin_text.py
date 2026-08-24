# pipeline/generator/_latin_text.py
"""Shared CLTK-based tokenization/lemmatization for the lexical baselines.

Used by :class:`TfidfCandidateGenerator` and :class:`BM25CandidateGenerator`
(and, downstream, the lexical classifier judge) so all lexical baselines
share one tokenization path, matching the reference experiments faithfully.
"""

from __future__ import annotations

import re
from typing import Any

_WORD_RE = re.compile(r"[^\W\d_]+", flags=re.UNICODE)

_tokenizer_cache: dict[str, Any] = {}
_lemmatizer_cache: dict[str, Any] = {}


# =============================================================================
# Tokenization helpers
# =============================================================================


def _load_latin_word_tokenizer() -> Any:
    """Load CLTK's Latin word tokenizer, trying known import paths."""
    if "tokenizer" in _tokenizer_cache:
        return _tokenizer_cache["tokenizer"]

    errors: list[str] = []
    module_missing = True
    for module_name in ("cltk.tokenizers.lat.lat", "cltk.tokenizers.lat.word"):
        try:
            module = __import__(module_name, fromlist=["LatinWordTokenizer"])
            module_missing = False
            tokenizer_cls = module.LatinWordTokenizer
            tokenizer = tokenizer_cls()
            _tokenizer_cache["tokenizer"] = tokenizer
            return tokenizer
        except ModuleNotFoundError as exc:
            errors.append(f"{module_name}: {exc!r}")
        except Exception as exc:  # pragma: no cover - depends on optional install
            module_missing = False
            errors.append(f"{module_name}: {exc!r}")

    if module_missing:
        raise ImportError(
            "This generator requires CLTK's Latin word tokenizer. Install it with: "
            "pip install 'locisimiles[lexical]'. Tried: " + "; ".join(errors)
        )
    raise ImportError(
        "CLTK is installed, but its Latin corpus data (sentence tokenizer model) "
        "could not be loaded. Fetch it once with:\n"
        '  python -c "from cltk.data.fetch import FetchCorpus; '
        "FetchCorpus(language='lat').import_corpus('lat_models_cltk')\"\n"
        "Tried: " + "; ".join(errors)
    )


def _load_latin_lemmatizer() -> Any | None:
    """Load CLTK's Latin backoff lemmatizer, or None if unavailable."""
    if "lemmatizer" in _lemmatizer_cache:
        return _lemmatizer_cache["lemmatizer"]

    try:
        from cltk.lemmatize.lat import LatinBackoffLemmatizer

        lemmatizer = LatinBackoffLemmatizer()
    except Exception:  # pragma: no cover - depends on optional install
        lemmatizer = None

    _lemmatizer_cache["lemmatizer"] = lemmatizer
    return lemmatizer


def tokenize(text: str, *, lowercase: bool = True) -> list[str]:
    """Tokenize Latin text with CLTK and keep alphabetic word tokens."""
    tokenizer = _load_latin_word_tokenizer()
    tokens = tokenizer.tokenize(text or "")
    # CLTK keeps punctuation as separate tokens; drop anything non-alphabetic.
    tokens = [tok for tok in tokens if _WORD_RE.fullmatch(tok)]
    return [tok.lower() for tok in tokens] if lowercase else tokens


def lemmatize_tokens(tokens: list[str]) -> list[str]:
    """Return lemmas for the given tokens, or the tokens unchanged if no lemmatizer."""
    lemmatizer = _load_latin_lemmatizer()
    if lemmatizer is None or not tokens:
        return list(tokens)
    pairs = lemmatizer.lemmatize(list(tokens))
    return [lemma for _surface, lemma in pairs]


def preprocess(text: str, *, lemmatize: bool, lowercase: bool) -> list[str]:
    """Tokenize and optionally lemmatize one segment of Latin text."""
    tokens = tokenize(text, lowercase=lowercase)
    if lemmatize:
        tokens = lemmatize_tokens(tokens)
    return tokens


# =============================================================================
# N-gram expansion
# =============================================================================


class NgramAnalyzer:
    """Picklable analyzer expanding pre-tokenized text to n-grams.

    Passed as ``TfidfVectorizer(analyzer=...)``. Must be picklable (a plain
    class, not a closure) so vectorizers using it can be persisted via
    ``joblib``/``pickle`` — e.g. by :class:`~locisimiles.training.lexical.LexicalClassifierTrainer`.
    """

    def __init__(self, ngram_range: tuple[int, int]):
        self.ngram_range = ngram_range

    def __call__(self, tokens: list[str]) -> list[str]:
        min_n, max_n = self.ngram_range
        features: list[str] = []
        for n in range(min_n, max_n + 1):
            if n <= 0 or len(tokens) < n:
                continue
            if n == 1:
                features.extend(tokens)
            else:
                features.extend(" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1))
        return features
