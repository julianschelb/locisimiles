"""Tests for the Latin BERT subword tokenizer and the tokenizer guard.

The reference segmentations below were produced with the real
``tensor2tensor`` ``SubwordTextEncoder`` over the published Latin BERT
vocabulary, so they pin this pure-Python reimplementation to the original.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from locisimiles.tokenization.latin_bert import (
    CLS_ID,
    NUM_SPECIAL_TOKENS,
    SEP_ID,
    UNK_ID,
    LatinBertSubwordTokenizer,
    SubwordTextEncoder,
    unk_rate,
)

FIXTURE = Path(__file__).parent / "fixtures" / "latin_mini.subword.encoder"

# Ground truth from tensor2tensor over models/subword_tokenizer_latin/latin.subword.encoder
REFERENCE_SEGMENTATIONS = {
    "obstipui": ["obsti", "pui", "_"],
    "steteruntque": ["steter", "unt", "que_"],
    "comae": ["coma", "e_"],
    "et": ["et_"],
    "uox": ["uo", "x_"],
    "faucibus": ["faucibus_"],
    "haesit": ["haesit", "_"],
    "virumque": ["viru", "mque_"],
}


@pytest.fixture(scope="module")
def encoder() -> SubwordTextEncoder:
    return SubwordTextEncoder.from_file(FIXTURE)


@pytest.fixture(scope="module")
def tokenizer(encoder: SubwordTextEncoder) -> LatinBertSubwordTokenizer:
    return LatinBertSubwordTokenizer(encoder)


@pytest.mark.parametrize("word,expected", sorted(REFERENCE_SEGMENTATIONS.items()))
def test_matches_tensor2tensor_segmentation(encoder, word, expected):
    """Segmentation reproduces the original tensor2tensor encoder."""
    assert encoder.subtokens_for_word(word) == expected


def test_no_unknown_tokens_on_latin(tokenizer):
    """Latin words segment fully; nothing falls back to [UNK]."""
    ids, spans = tokenizer.encode_segment("obstipui steteruntque comae et uox faucibus haesit")
    assert ids[0] == CLS_ID and ids[-1] == SEP_ID
    assert UNK_ID not in ids
    assert len(spans) == 7


def test_ids_are_shifted_past_the_special_tokens(tokenizer, encoder):
    """Every subtoken id leaves room for [PAD] [UNK] [CLS] [SEP] [MASK]."""
    (word_ids,) = tokenizer.encode_words("et")
    assert word_ids == [encoder.encode_word("et")[0] + NUM_SPECIAL_TOKENS]
    assert min(word_ids) >= NUM_SPECIAL_TOKENS


def test_word_spans_cover_ids_contiguously(tokenizer):
    """Each word maps to a contiguous, non-overlapping id range."""
    ids, spans = tokenizer.encode_segment("obstipui et faucibus")
    assert [end - start for start, end in spans] == [3, 1, 1]
    assert all(spans[i][1] == spans[i + 1][0] for i in range(len(spans) - 1))
    assert spans[-1][1] == len(ids) - 1  # everything before the final [SEP]


def test_lowercases_like_the_reference_implementation(tokenizer):
    """gen_berts.convert_to_toks lowercases before tokenizing."""
    assert tokenizer.words("Obstipui ET") == ["obstipui", "et"]


def test_max_length_truncates_whole_words(tokenizer):
    """Truncation never splits a word across the limit."""
    ids, spans = tokenizer.encode_segment("obstipui steteruntque comae", max_length=6)
    assert len(ids) <= 6
    assert all(end <= len(ids) - 1 for _start, end in spans)


def test_pair_encoding_builds_token_type_ids(tokenizer):
    """Pair encoding yields [CLS] a [SEP] b [SEP] with matching segment ids."""
    ids, token_type_ids = tokenizer.encode_pair("et faucibus", "comae haesit")
    assert len(ids) == len(token_type_ids)
    assert ids[0] == CLS_ID and ids[-1] == SEP_ID
    assert set(token_type_ids) == {0, 1}
    assert token_type_ids[0] == 0 and token_type_ids[-1] == 1


class _WordPieceOverSubwordVocab:
    """Stand-in for AutoTokenizer over a tensor2tensor vocabulary.

    WordPiece cannot segment subtokens that end in '_', so nearly every Latin
    word becomes [UNK] -- the failure this guard exists to catch.
    """

    unk_token_id = 100

    def __call__(self, text, add_special_tokens=False):
        known = {"et", "haesit"}
        return {"input_ids": [7 if text.lower() in known else self.unk_token_id]}


class _HealthyTokenizer:
    unk_token_id = 100

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [7, 8]}


def test_unk_rate_detects_a_tokenizer_that_cannot_segment_latin():
    """The broken pairing is measurable, not silent."""
    assert unk_rate(_WordPieceOverSubwordVocab()) > 0.5
    assert unk_rate(_HealthyTokenizer()) == 0.0


def test_generator_rejects_a_tokenizer_that_cannot_segment_latin(monkeypatch):
    """Constructing the generator with a broken tokenizer raises, not warns."""
    from locisimiles.pipeline.generator import contextual_bert as module

    monkeypatch.setattr(
        module.AutoTokenizer, "from_pretrained",
        classmethod(lambda cls, *a, **k: _WordPieceOverSubwordVocab()),
    )
    with pytest.raises(ValueError, match=r"\[UNK\]"):
        module.LatinBertContextualCandidateGenerator(model_name="some/latin-bert")


class _ModelLoadReached(RuntimeError):
    """Sentinel proving construction proceeded past the tokenizer guard."""


def test_guard_can_be_disabled(monkeypatch):
    """check_tokenizer=False lets a user proceed knowingly."""
    from locisimiles.pipeline.generator import contextual_bert as module

    monkeypatch.setattr(
        module.AutoTokenizer, "from_pretrained",
        classmethod(lambda cls, *a, **k: _WordPieceOverSubwordVocab()),
    )
    monkeypatch.setattr(
        module.AutoModel, "from_pretrained",
        classmethod(lambda cls, *a, **k: (_ for _ in ()).throw(_ModelLoadReached())),
    )
    # Reaching the model load means the guard did not fire.
    with pytest.raises(_ModelLoadReached):
        module.LatinBertContextualCandidateGenerator(
            model_name="some/latin-bert", check_tokenizer=False
        )
