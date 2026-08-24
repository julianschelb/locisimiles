"""Tests for classification threshold tuning and application."""

from __future__ import annotations

from locisimiles.document import TextSegment
from locisimiles.pipeline._types import CandidateJudge
from locisimiles.training.classification.threshold import (
    ThresholdSet,
    apply_thresholds,
    apply_thresholds_to_judgments,
    tune_threshold,
)

# id_to_label for a 3-class no_match/cit/cf classifier.
ID_TO_LABEL = {0: "no_match", 1: "cit", 2: "cf"}

# Clean-separated synthetic probabilities: rows 0-2 are clearly "cit", rows
# 3-5 clearly "cf", rows 6-8 clearly "no_match".
PROBABILITIES = [
    [0.05, 0.90, 0.05],
    [0.10, 0.85, 0.05],
    [0.02, 0.93, 0.05],
    [0.05, 0.05, 0.90],
    [0.10, 0.05, 0.85],
    [0.03, 0.02, 0.95],
    [0.90, 0.05, 0.05],
    [0.85, 0.10, 0.05],
    [0.92, 0.03, 0.05],
]
GOLD_LABELS = ["cit", "cit", "cit", "cf", "cf", "cf", "no_match", "no_match", "no_match"]


class TestThresholdSetJson:
    def test_round_trip(self, temp_dir):
        thresholds = ThresholdSet(thresholds={"cit": 0.6, "cf": 0.7}, method="max_f1")
        path = thresholds.to_json(temp_dir / "threshold.json")
        assert path.exists()

        loaded = ThresholdSet.from_json(path)
        assert loaded.thresholds == thresholds.thresholds
        assert loaded.method == "max_f1"
        assert loaded.tie_break == "max_probability"


class TestTuneThreshold:
    def test_max_f1_finds_separating_thresholds(self):
        result = tune_threshold(
            probabilities=PROBABILITIES,
            gold_labels=GOLD_LABELS,
            id_to_label=ID_TO_LABEL,
            method="max_f1",
        )
        assert set(result.thresholds) == {"cit", "cf"}
        # Any threshold strictly above the highest negative-row probability (0.10 for
        # "cit", 0.05 for "cf") and at/below the lowest positive-row probability (0.85)
        # perfectly separates the classes here; the sweep returns the lowest such value.
        assert 0.10 < result.thresholds["cit"] <= 0.85
        assert 0.05 < result.thresholds["cf"] <= 0.85

    def test_youden_method_runs(self):
        result = tune_threshold(
            probabilities=PROBABILITIES,
            gold_labels=GOLD_LABELS,
            id_to_label=ID_TO_LABEL,
            method="youden",
        )
        assert set(result.thresholds) == {"cit", "cf"}

    def test_unknown_method_raises(self):
        import pytest

        with pytest.raises(ValueError, match="Unknown threshold-tuning method"):
            tune_threshold(
                probabilities=PROBABILITIES,
                gold_labels=GOLD_LABELS,
                id_to_label=ID_TO_LABEL,
                method="bogus",
            )

    def test_negative_label_excluded_from_thresholds(self):
        result = tune_threshold(
            probabilities=PROBABILITIES,
            gold_labels=GOLD_LABELS,
            id_to_label=ID_TO_LABEL,
            negative_label="no_match",
        )
        assert "no_match" not in result.thresholds


class TestApplyThresholds:
    def setup_method(self):
        self.thresholds = ThresholdSet(thresholds={"cit": 0.5, "cf": 0.5})

    def test_clear_positive_wins(self):
        label, class_id = apply_thresholds(
            {"no_match": 0.05, "cit": 0.90, "cf": 0.05}, self.thresholds
        )
        assert label == "cit"
        assert class_id is None

    def test_falls_back_to_negative_when_nothing_clears(self):
        label, _ = apply_thresholds({"no_match": 0.80, "cit": 0.15, "cf": 0.05}, self.thresholds)
        assert label == "no_match"

    def test_tie_break_picks_higher_probability(self):
        # Both cit and cf clear their 0.5 threshold; cf has the higher probability.
        label, _ = apply_thresholds({"no_match": 0.05, "cit": 0.55, "cf": 0.60}, self.thresholds)
        assert label == "cf"

    def test_missing_class_in_probabilities_treated_as_zero(self):
        label, _ = apply_thresholds({"no_match": 0.4, "cit": 0.6}, self.thresholds)
        assert label == "cit"


class TestApplyThresholdsToJudgments:
    def test_rewrites_label_and_score_from_thresholds(self):
        thresholds = ThresholdSet(thresholds={"cit": 0.5, "cf": 0.5})
        segment = TextSegment("text", "s1")
        judgments = {
            "q1": [
                CandidateJudge(
                    segment=segment,
                    candidate_score=0.9,
                    judgment_score=0.05,  # argmax judge picked "no_match" originally
                    predicted_class_id=0,
                    predicted_label="no_match",
                    class_probabilities={"no_match": 0.30, "cit": 0.05, "cf": 0.65},
                )
            ]
        }
        result = apply_thresholds_to_judgments(judgments, thresholds, negative_label="no_match")
        rewritten = result["q1"][0]
        assert rewritten.predicted_label == "cf"
        assert rewritten.judgment_score == 0.65
        # class_probabilities are preserved unchanged.
        assert rewritten.class_probabilities == judgments["q1"][0].class_probabilities

    def test_passes_through_judgments_without_class_probabilities(self):
        thresholds = ThresholdSet(thresholds={"cit": 0.5})
        segment = TextSegment("text", "s1")
        original = CandidateJudge(segment=segment, candidate_score=0.9, judgment_score=0.8)
        judgments = {"q1": [original]}

        result = apply_thresholds_to_judgments(judgments, thresholds)
        assert result["q1"][0] is original

    def test_does_not_mutate_input(self):
        thresholds = ThresholdSet(thresholds={"cit": 0.5})
        segment = TextSegment("text", "s1")
        original = CandidateJudge(
            segment=segment,
            candidate_score=0.9,
            judgment_score=0.2,
            class_probabilities={"no_match": 0.8, "cit": 0.2},
        )
        judgments = {"q1": [original]}

        apply_thresholds_to_judgments(judgments, thresholds)
        assert judgments["q1"][0] is original
        assert original.judgment_score == 0.2  # unchanged
