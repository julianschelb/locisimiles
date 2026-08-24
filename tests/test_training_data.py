"""Tests for TrainingData: iteration, concatenation, and negative sampling."""

from __future__ import annotations

import pytest

from locisimiles.document import Document
from locisimiles.ground_truth import GroundTruth
from locisimiles.training.data import TrainingData


@pytest.fixture
def query_doc_5(temp_dir) -> Document:
    path = temp_dir / "query5.csv"
    path.write_text(
        "seg_id,text\n"
        "q1,Arma virumque cano.\n"
        "q2,Italiam fato profugus.\n"
        "q3,Litora multum iactatus.\n",
        encoding="utf-8",
    )
    return Document(path)


@pytest.fixture
def source_doc_5(temp_dir) -> Document:
    path = temp_dir / "source5.csv"
    path.write_text(
        "seg_id,text\n"
        "s1,Arma virumque cano qui primus.\n"
        "s2,Fato profugus Italiam venit.\n"
        "s3,Multum terris iactatus et alto.\n"
        "s4,Completely unrelated text here.\n"
        "s5,Another unrelated segment entirely.\n",
        encoding="utf-8",
    )
    return Document(path)


@pytest.fixture
def positives() -> GroundTruth:
    return GroundTruth(
        [
            {"query_id": "q1", "source_id": "s1", "label": "cit"},
            {"query_id": "q2", "source_id": "s2", "label": "cf"},
        ]
    )


class TestTrainingDataIteration:
    def test_len_matches_ground_truth(self, query_doc_5, source_doc_5, positives):
        data = TrainingData(query_doc_5, source_doc_5, positives)
        assert len(data) == 2

    def test_iterates_resolved_text_triples(self, query_doc_5, source_doc_5, positives):
        data = TrainingData(query_doc_5, source_doc_5, positives)
        rows = list(data)
        assert rows[0] == ("Arma virumque cano.", "Arma virumque cano qui primus.", "cit")
        assert rows[1] == ("Italiam fato profugus.", "Fato profugus Italiam venit.", "cf")


class TestTrainingDataAdd:
    def test_add_concatenates_ground_truth(self, query_doc_5, source_doc_5, positives):
        a = TrainingData(query_doc_5, source_doc_5, positives)
        b = TrainingData(
            query_doc_5,
            source_doc_5,
            GroundTruth([{"query_id": "q3", "source_id": "s3", "label": "cf"}]),
        )
        combined = a + b
        assert len(combined) == 3
        assert len(a) == 2  # original untouched


class TestTrainingDataSampleRandomPairs:
    def test_default_method_random_pairs(self, query_doc_5, source_doc_5, positives):
        data = TrainingData(query_doc_5, source_doc_5, positives)
        result = data.sample_random_pairs(n_per_query=1, seed=1)
        assert len(result) == len(positives) + len(query_doc_5) * 1
        new_pairs = list(result)[len(positives) :]
        assert all(label == "no_match" for *_pair, label in new_pairs)

    def test_never_duplicates_a_known_positive(self, query_doc_5, source_doc_5, positives):
        data = TrainingData(query_doc_5, source_doc_5, positives)
        result = data.sample_random_pairs(n_per_query=5, seed=7)
        known = {(e.query_id, e.source_id) for e in positives}
        sampled = list(result.ground_truth)[len(positives) :]
        for entry in sampled:
            assert (entry.query_id, entry.source_id) not in known

    def test_not_restricted_to_positive_queries(self, query_doc_5, source_doc_5):
        """Unlike the other three methods, pairs aren't conditioned per query."""
        data = TrainingData(query_doc_5, source_doc_5, GroundTruth())
        result = data.sample_random_pairs(n_per_query=10, seed=3)
        sampled_query_ids = {e.query_id for e in result.ground_truth}
        # With enough draws, more than one query segment should appear.
        assert len(sampled_query_ids) > 1

    def test_reproducible_for_fixed_seed(self, query_doc_5, source_doc_5, positives):
        data = TrainingData(query_doc_5, source_doc_5, positives)
        first = data.sample_random_pairs(n_per_query=3, seed=99)
        second = data.sample_random_pairs(n_per_query=3, seed=99)
        first_pairs = [(e.query_id, e.source_id) for e in first.ground_truth]
        second_pairs = [(e.query_id, e.source_id) for e in second.ground_truth]
        assert first_pairs == second_pairs


class TestTrainingDataSampleRandomNegatives:
    def test_returns_new_training_data_not_mutated(self, query_doc_5, source_doc_5, positives):
        data = TrainingData(query_doc_5, source_doc_5, positives)
        result = data.sample_random_negatives(n_per_query=2, seed=42)
        assert result is not data
        assert len(data) == 2  # original untouched
        assert len(result) > len(data)

    def test_every_query_segment_gets_negatives(self, query_doc_5, source_doc_5, positives):
        data = TrainingData(query_doc_5, source_doc_5, positives)
        result = data.sample_random_negatives(n_per_query=2, seed=42)
        sampled = list(result.ground_truth)[len(positives) :]
        sampled_by_query: dict = {}
        for entry in sampled:
            sampled_by_query.setdefault(entry.query_id, []).append(entry)
        assert set(sampled_by_query) == {"q1", "q2", "q3"}
        assert all(len(v) == 2 for v in sampled_by_query.values())

    def test_never_samples_a_known_positive(self, query_doc_5, source_doc_5, positives):
        data = TrainingData(query_doc_5, source_doc_5, positives)
        result = data.sample_random_negatives(n_per_query=4, seed=1)
        known = {(e.query_id, e.source_id) for e in positives}
        sampled = list(result.ground_truth)[len(positives) :]
        for entry in sampled:
            assert (entry.query_id, entry.source_id) not in known
            assert entry.label == "no_match"

    def test_reproducible_for_fixed_seed(self, query_doc_5, source_doc_5, positives):
        data = TrainingData(query_doc_5, source_doc_5, positives)
        first = data.sample_random_negatives(n_per_query=2, seed=5)
        second = data.sample_random_negatives(n_per_query=2, seed=5)
        first_pairs = [(e.query_id, e.source_id) for e in first.ground_truth]
        second_pairs = [(e.query_id, e.source_id) for e in second.ground_truth]
        assert first_pairs == second_pairs

    def test_chains_from_constructor(self, query_doc_5, source_doc_5, positives):
        data = TrainingData(query_doc_5, source_doc_5, positives).sample_random_negatives(
            n_per_query=1, seed=1
        )
        assert isinstance(data, TrainingData)
        assert len(data) == len(positives) + len(query_doc_5)
