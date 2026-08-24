# training/classification/tokenizer_utils.py
"""Tokenizer helpers shared between the classification trainer and judge.

The pair-truncation strategy here mirrors
:meth:`~locisimiles.pipeline.judge.classification.ClassificationJudge._truncate_pair`
exactly, so tokenization at training time matches tokenization at inference
time.
"""

from __future__ import annotations

from typing import Any, Tuple


def count_special_tokens_added(tokenizer: Any) -> int:
    """Count special tokens added by the tokenizer for pair encoding."""
    return tokenizer.num_special_tokens_to_add(pair=True)


def truncate_pair(
    tokenizer: Any,
    sentence1: str,
    sentence2: str,
    max_len: int = 512,
) -> Tuple[str, str]:
    """Truncate a text pair to fit within ``max_len`` including specials."""
    num_special = count_special_tokens_added(tokenizer)
    max_tokens = max(0, max_len - num_special)
    half = max_tokens // 2

    tokens1 = tokenizer.tokenize(sentence1)[:half]
    tokens2 = tokenizer.tokenize(sentence2)[:half]

    return (
        tokenizer.convert_tokens_to_string(tokens1),
        tokenizer.convert_tokens_to_string(tokens2),
    )


def add_roberta_separators(tokenizer: Any) -> None:
    """Force a RoBERTa-style ``<s> A </s></s> B </s>`` post-processor for pair encoding.

    Some Latin RoBERTa checkpoints (e.g. PhilBerta, LaBerta) ship a
    single-sequence post-processor by default, which mishandles paired
    sequences during pair classification. This rebuilds the fast
    tokenizer's post-processor to the standard RoBERTa pair template.
    """
    if not hasattr(tokenizer, "backend_tokenizer"):
        raise ValueError("add_roberta_separators requires a fast (Rust-backed) tokenizer")

    from tokenizers.processors import TemplateProcessing

    cls_token, sep_token = tokenizer.cls_token, tokenizer.sep_token
    cls_id, sep_id = tokenizer.cls_token_id, tokenizer.sep_token_id

    tokenizer.backend_tokenizer.post_processor = TemplateProcessing(
        single=f"{cls_token} $A {sep_token}",
        pair=f"{cls_token} $A {sep_token} {sep_token} $B {sep_token}",
        special_tokens=[(cls_token, cls_id), (sep_token, sep_id)],
    )
