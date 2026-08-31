"""Subword tokenization for Latin BERT (Bamman and Burns, 2020).

Latin BERT was trained with a ``tensor2tensor`` ``SubwordTextEncoder``, whose
vocabulary marks word endings with a trailing underscore (``et_``, ``que_``)
and has no ``##`` continuation prefix. The public HuggingFace conversions of
the model ship that vocabulary as ``vocab.txt`` but no tokenizer
configuration, so ``AutoTokenizer.from_pretrained`` silently builds a
**WordPiece** tokenizer over it. WordPiece cannot segment such a vocabulary and
falls back to ``[UNK]`` for roughly 60% of Latin words, which quietly destroys
most of the input while still producing plausible-looking scores.

This module reimplements the encoder in pure Python so that no TensorFlow
dependency is required, and reproduces the id layout used by the reference
implementation (``gen_berts.py`` in the Latin BERT repository): the five
special tokens occupy ids 0-4 and every subtoken id is shifted by ``+5``.

Reference: https://github.com/dbamman/latin-bert
"""

from __future__ import annotations

import re
from pathlib import Path

__all__ = ["SubwordTextEncoder", "LatinBertSubwordTokenizer", "unk_rate"]

PAD_ID, UNK_ID, CLS_ID, SEP_ID, MASK_ID = 0, 1, 2, 3, 4
SPECIAL_TOKENS = ("[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]")
NUM_SPECIAL_TOKENS = len(SPECIAL_TOKENS)

_ESCAPE_RE = re.compile(r"\\([0-9]+);")
_WORD_RE = re.compile(r"[A-Za-z]+")

# Probe sentence used to detect a tokenizer that cannot segment Latin.
PROBE_SENTENCE = "obstipui steteruntque comae et uox faucibus haesit"


def _escape_token(token: str, alphabet: frozenset[str]) -> str:
    """Escape a token exactly as ``tensor2tensor`` does before segmentation.

    Backslashes and underscores are escaped, characters outside the vocabulary
    alphabet become ``\\<codepoint>;``, and a terminal underscore marks the end
    of the word.
    """
    token = token.replace("\\", "\\\\").replace("_", "\\u")
    chars = [c if (c in alphabet and c != "\n") else f"\\{ord(c)};" for c in token]
    return "".join(chars) + "_"


class SubwordTextEncoder:
    """Greedy longest-match subword encoder over a ``tensor2tensor`` vocabulary.

    Args:
        subtokens: Subtoken strings in vocabulary order; the index is the id.
    """

    def __init__(self, subtokens: list[str]):
        if not subtokens:
            raise ValueError("Subword vocabulary is empty.")
        self._subtokens = list(subtokens)
        self._subtoken_to_id = {s: i for i, s in enumerate(self._subtokens)}
        self._max_subtoken_len = max(len(s) for s in self._subtokens)
        self._alphabet = frozenset(c for s in self._subtokens for c in s)

    # ---------- Construction ----------

    @classmethod
    def from_file(cls, path: str | Path) -> SubwordTextEncoder:
        """Load a ``*.subword.encoder`` vocabulary file.

        Lines may be wrapped in single or double quotes, as written by
        ``tensor2tensor``.
        """
        vocab_path = Path(path)
        if not vocab_path.exists():
            raise FileNotFoundError(f"Subword vocabulary not found: {vocab_path}")

        subtokens: list[str] = []
        with vocab_path.open(encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if (stripped.startswith("'") and stripped.endswith("'")) or (
                    stripped.startswith('"') and stripped.endswith('"')
                ):
                    stripped = stripped[1:-1]
                subtokens.append(stripped)
        return cls(subtokens)

    # ---------- Properties ----------

    @property
    def vocab_size(self) -> int:
        """Number of subtokens in the vocabulary."""
        return len(self._subtokens)

    def __contains__(self, subtoken: str) -> bool:
        return subtoken in self._subtoken_to_id

    # ---------- Encoding ----------

    def subtokens_for_word(self, word: str) -> list[str]:
        """Segment one word into subtoken strings via greedy longest match."""
        escaped = _escape_token(word, self._alphabet)
        pieces: list[str] = []
        start, length = 0, len(escaped)
        while start < length:
            for end in range(min(length, start + self._max_subtoken_len), start, -1):
                candidate = escaped[start:end]
                if candidate in self._subtoken_to_id:
                    pieces.append(candidate)
                    start = end
                    break
            else:
                # Unreachable for well-formed vocabularies, which always contain
                # every single character of their own alphabet.
                raise ValueError(
                    f"No subtoken of {escaped[start:]!r} is present in the vocabulary."
                )
        return pieces

    def encode_word(self, word: str) -> list[int]:
        """Return raw (unshifted) subtoken ids for one word."""
        return [self._subtoken_to_id[s] for s in self.subtokens_for_word(word)]

    def decode_subtokens(self, subtokens: list[str]) -> str:
        """Join subtokens back into text (inverse of the escaping above)."""
        joined = "".join(subtokens).replace("_", " ").replace("\\u", "_").replace("\\\\", "\\")
        return _ESCAPE_RE.sub(lambda m: chr(int(m.group(1))), joined).strip()


class LatinBertSubwordTokenizer:
    """Latin BERT tokenizer with the id layout of the reference implementation.

    Words are lowercased before segmentation, matching ``convert_to_toks`` in
    ``gen_berts.py``, and every subtoken id is shifted by ``+5`` to leave room
    for ``[PAD] [UNK] [CLS] [SEP] [MASK]``.

    Args:
        encoder: Loaded subword encoder.
        lowercase: Whether to lowercase text before segmentation.
    """

    def __init__(self, encoder: SubwordTextEncoder, *, lowercase: bool = True):
        self.encoder = encoder
        self.lowercase = bool(lowercase)

    @classmethod
    def from_vocab_file(
        cls, path: str | Path, *, lowercase: bool = True
    ) -> LatinBertSubwordTokenizer:
        """Build a tokenizer from a ``latin.subword.encoder`` file."""
        return cls(SubwordTextEncoder.from_file(path), lowercase=lowercase)

    # ---------- Properties ----------

    @property
    def vocab_size(self) -> int:
        """Vocabulary size including the five special tokens."""
        return self.encoder.vocab_size + NUM_SPECIAL_TOKENS

    # ---------- Encoding ----------

    def words(self, text: str) -> list[str]:
        """Extract alphabetic words, lowercased when configured."""
        source = text or ""
        if self.lowercase:
            source = source.lower()
        return _WORD_RE.findall(source)

    def encode_words(self, text: str) -> list[list[int]]:
        """Return one list of subtoken ids per word.

        Grouping by word is what allows word-level pooling without an offset
        mapping, which this vocabulary cannot provide.
        """
        return [
            [i + NUM_SPECIAL_TOKENS for i in self.encoder.encode_word(w)] for w in self.words(text)
        ]

    def encode_pair(
        self, text_a: str, text_b: str, *, max_length: int = 512
    ) -> tuple[list[int], list[int]]:
        """Encode a segment pair as ``[CLS] a [SEP] b [SEP]`` with token type ids.

        The longer side is truncated first, as HuggingFace's ``longest_first``
        strategy does.
        """
        a = [i for word in self.encode_words(text_a) for i in word]
        b = [i for word in self.encode_words(text_b) for i in word]
        budget = max(2, int(max_length) - 3)
        while len(a) + len(b) > budget:
            (a if len(a) >= len(b) else b).pop()
        ids = [CLS_ID] + a + [SEP_ID] + b + [SEP_ID]
        token_type_ids = [0] * (len(a) + 2) + [1] * (len(b) + 1)
        return ids, token_type_ids

    def encode_segment(
        self, text: str, *, max_length: int = 256
    ) -> tuple[list[int], list[tuple[int, int]]]:
        """Encode one segment as ``[CLS] … [SEP]``.

        Returns the ids and, for each word that fits, the ``[start, end)`` span
        of its subtokens within those ids.
        """
        ids = [CLS_ID]
        spans: list[tuple[int, int]] = []
        for word_ids in self.encode_words(text):
            if len(ids) + len(word_ids) + 1 > max_length:
                break
            spans.append((len(ids), len(ids) + len(word_ids)))
            ids.extend(word_ids)
        ids.append(SEP_ID)
        return ids, spans


def unk_rate(tokenizer, text: str = PROBE_SENTENCE) -> float:
    """Fraction of ``text``'s words that a HuggingFace tokenizer maps to ``[UNK]``.

    Used to detect a tokenizer that cannot segment the target language, such as
    a WordPiece tokenizer built from a ``tensor2tensor`` vocabulary.
    """
    unk_id = getattr(tokenizer, "unk_token_id", None)
    words = _WORD_RE.findall(text)
    if unk_id is None or not words:
        return 0.0

    unknown = 0
    for word in words:
        try:
            ids = tokenizer(word, add_special_tokens=False)["input_ids"]
        except Exception:  # pragma: no cover - defensive, tokenizer-specific
            return 0.0
        if not ids or any(i == unk_id for i in ids):
            unknown += 1
    return unknown / len(words)
