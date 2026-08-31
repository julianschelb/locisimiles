"""Tests for the vocabulary-coverage diagnostic."""

from __future__ import annotations

import warnings

import pytest

from locisimiles.diagnostics import vocab_coverage, warn_on_low_coverage


def test_reports_token_and_type_coverage_separately():
    """A frequent OOV word hurts token coverage more than type coverage."""
    tokens = ["uolo"] * 8 + ["uel", "iam"]
    report = vocab_coverage(tokens, {"uel", "iam"})
    assert report.token_coverage == pytest.approx(0.2)
    assert report.type_coverage == pytest.approx(2 / 3)
    assert report.top_oov[0] == ("uolo", 8)


def test_full_coverage_produces_no_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        report = warn_on_low_coverage(["uel", "iam"], {"uel", "iam"}, label="test")
    assert report.token_coverage == 1.0


def test_low_coverage_warns_and_names_frequent_misses():
    """This is the signal that exposed the v/u orthography mismatch."""
    tokens = ["volo"] * 9 + ["uel"]
    with pytest.warns(UserWarning, match="volo"):
        report = warn_on_low_coverage(tokens, {"uel"}, label="Word2Vec generator")
    assert report.token_coverage == pytest.approx(0.1)


def test_empty_input_is_safe():
    report = vocab_coverage([], {"uel"})
    assert report.token_coverage == 0.0 and report.n_tokens == 0
