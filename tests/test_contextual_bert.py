"""Tests for contextual Latin BERT generator and pipelines."""

from __future__ import annotations

import re
from unittest.mock import MagicMock, patch

import torch

from locisimiles.pipeline._types import Candidate


class _DummyTokenizer:
    def __init__(self):
        self._vocab: dict[str, int] = {}
        self._next_id = 10

    @classmethod
    def from_pretrained(cls, _model_ref):
        return cls()

    def _token_id(self, token: str) -> int:
        key = token.lower()
        if key not in self._vocab:
            self._vocab[key] = self._next_id
            self._next_id += 1
        return self._vocab[key]

    def __call__(
        self,
        text: str,
        *,
        return_tensors: str,
        truncation: bool,
        max_length: int,
        return_offsets_mapping: bool,
    ):
        words = [(m.group(0), m.start(), m.end()) for m in re.finditer(r"[A-Za-z]+", text)]
        offsets = [(0, 0)]
        token_ids = [0]
        for token, start, end in words:
            offsets.append((start, end))
            token_ids.append(self._token_id(token))
        offsets.append((0, 0))
        token_ids.append(0)

        input_ids = torch.tensor([token_ids], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        offset_mapping = torch.tensor([offsets], dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "offset_mapping": offset_mapping,
        }


class _DummyModel:
    @classmethod
    def from_pretrained(cls, _model_ref):
        return cls()

    def to(self, _device):
        return self

    def eval(self):
        return self

    def __call__(self, **kwargs):
        input_ids = kwargs["input_ids"]
        hidden = torch.nn.functional.one_hot(input_ids % 16, num_classes=16).float()
        return MagicMock(last_hidden_state=hidden)


class TestContextualGenerator:
    @patch("locisimiles.pipeline.generator.contextual_bert.AutoModel", _DummyModel)
    @patch("locisimiles.pipeline.generator.contextual_bert.AutoTokenizer", _DummyTokenizer)
    def test_generate_returns_ranked_candidates(self, query_document, source_document):
        from locisimiles.pipeline.generator.contextual_bert import (
            LatinBertContextualCandidateGenerator,
        )

        generator = LatinBertContextualCandidateGenerator(model_name="dummy/model")
        results = generator.generate(query=query_document, source=source_document, top_k=3)

        assert set(results.keys()) == {seg.id for seg in query_document}
        for cands in results.values():
            assert len(cands) <= 3
            assert all(isinstance(c, Candidate) for c in cands)
            scores = [c.score for c in cands]
            assert scores == sorted(scores, reverse=True)

    @patch("locisimiles.pipeline.generator.contextual_bert.AutoModel", _DummyModel)
    @patch("locisimiles.pipeline.generator.contextual_bert.AutoTokenizer", _DummyTokenizer)
    def test_missing_local_path_raises(self, temp_dir):
        from locisimiles.pipeline.generator.contextual_bert import (
            LatinBertContextualCandidateGenerator,
        )

        missing_path = temp_dir / "does-not-exist"
        try:
            LatinBertContextualCandidateGenerator(model_path=missing_path)
            raise AssertionError("Expected FileNotFoundError")
        except FileNotFoundError:
            pass


class TestContextualPipelines:
    @patch("locisimiles.pipeline.generator.contextual_bert.AutoModel", _DummyModel)
    @patch("locisimiles.pipeline.generator.contextual_bert.AutoTokenizer", _DummyTokenizer)
    def test_retrieval_pipeline_composition(self):
        from locisimiles.pipeline.contextual_retrieval import LatinBertRetrievalPipeline
        from locisimiles.pipeline.judge.threshold import ThresholdJudge

        pipeline = LatinBertRetrievalPipeline(model_name="dummy/model", top_k=7)
        assert isinstance(pipeline.judge, ThresholdJudge)
        assert pipeline.judge.top_k == 7

    @patch("locisimiles.pipeline.generator.contextual_bert.AutoModel", _DummyModel)
    @patch("locisimiles.pipeline.generator.contextual_bert.AutoTokenizer", _DummyTokenizer)
    @patch("locisimiles.pipeline.contextual_two_stage.ClassificationJudge")
    def test_two_stage_pipeline_composition(self, mock_judge):
        from locisimiles.pipeline.contextual_two_stage import LatinBertTwoStagePipeline

        mock_judge.return_value = MagicMock()
        pipeline = LatinBertTwoStagePipeline(model_name="dummy/model")
        assert mock_judge.called
        assert pipeline.judge is mock_judge.return_value
