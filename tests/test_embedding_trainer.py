"""Tests for EmbeddingTrainer.

Trains a real (not mocked) tiny SentenceTransformer end to end, using a
small public HF test checkpoint, since this is the only way to verify the
trainer and
:class:`~locisimiles.pipeline.generator.embedding.EmbeddingCandidateGenerator`
stay consistent with each other (matching the ``prompts``/``prompt_name``
convention).
"""

from __future__ import annotations

import json

import pytest
import torch

from locisimiles.document import Document
from locisimiles.ground_truth import GroundTruth
from locisimiles.training.data import TrainingData

TINY_MODEL = "hf-internal-testing/tiny-random-bert"


@pytest.fixture(autouse=True)
def _force_cpu_device(monkeypatch):
    """Force CPU-only training for this module.

    HuggingFace's ``Trainer``/``accelerate`` auto-detect and dispatch to
    Apple's MPS backend independent of ``EmbeddingTrainerConfig.device``, and
    this machine's torch/MPS build hits a placeholder-storage bug under that
    path. Disabling MPS visibility keeps these tests deterministic and
    CPU-only regardless of the host machine.
    """
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)


@pytest.fixture
def query_doc_2(temp_dir) -> Document:
    path = temp_dir / "query2.csv"
    path.write_text(
        "seg_id,text\nq1,Arma virumque cano.\nq2,Italiam fato profugus.\n", encoding="utf-8"
    )
    return Document(path)


@pytest.fixture
def source_doc_4(temp_dir) -> Document:
    path = temp_dir / "source4.csv"
    path.write_text(
        "seg_id,text\n"
        "s1,Arma virumque cano qui primus.\n"
        "s2,Completely unrelated cooking text.\n"
        "s3,Fato profugus Italiam venit.\n"
        "s4,Another unrelated politics segment.\n",
        encoding="utf-8",
    )
    return Document(path)


@pytest.fixture
def train_data(query_doc_2, source_doc_4) -> TrainingData:
    gt = GroundTruth(
        [
            {"query_id": "q1", "source_id": "s1", "label": "match"},
            {"query_id": "q1", "source_id": "s2", "label": "no_match"},
            {"query_id": "q2", "source_id": "s3", "label": "match"},
            {"query_id": "q2", "source_id": "s4", "label": "no_match"},
        ]
    )
    return TrainingData(query_doc_2, source_doc_4, gt)


@pytest.fixture
def tiny_config(temp_dir):
    from locisimiles.training.embedding import EmbeddingTrainerConfig

    return EmbeddingTrainerConfig(
        output_dir=temp_dir,
        model_name=TINY_MODEL,
        epochs=1,
        batch_size=2,
    )


class TestEmbeddingTrainerFitSave:
    def test_fit_and_save_roundtrip(self, tiny_config, train_data):
        from locisimiles.training.embedding import EmbeddingTrainer

        trainer = EmbeddingTrainer(tiny_config)
        model = trainer.fit(data=train_data)
        assert model is not None

        out_path = trainer.save()
        assert out_path.exists()
        assert out_path.name == tiny_config.output_filename
        assert (out_path / "config_sentence_transformers.json").exists()

    def test_prompts_baked_into_saved_config(self, tiny_config, train_data):
        from locisimiles.training.embedding import EmbeddingTrainer

        trainer = EmbeddingTrainer(tiny_config)
        trainer.fit(data=train_data)
        out_path = trainer.save()

        saved = json.loads((out_path / "config_sentence_transformers.json").read_text())
        assert saved["prompts"] == {"query": "query: ", "match": "passage: "}

    def test_custom_prompts_are_used(self, temp_dir, train_data):
        from locisimiles.training.embedding import EmbeddingTrainer, EmbeddingTrainerConfig

        cfg = EmbeddingTrainerConfig(
            output_dir=temp_dir,
            model_name=TINY_MODEL,
            epochs=1,
            batch_size=2,
            prompts={"query": "Q: ", "match": "P: "},
        )
        trainer = EmbeddingTrainer(cfg)
        trainer.fit(data=train_data)
        out_path = trainer.save()

        saved = json.loads((out_path / "config_sentence_transformers.json").read_text())
        assert saved["prompts"] == {"query": "Q: ", "match": "P: "}

    def test_saved_model_directly_loadable_by_embedding_generator(
        self, tiny_config, train_data, query_doc_2, source_doc_4
    ):
        from locisimiles.pipeline.generator.embedding import EmbeddingCandidateGenerator
        from locisimiles.training.embedding import EmbeddingTrainer

        trainer = EmbeddingTrainer(tiny_config)
        trainer.fit(data=train_data)
        out_path = trainer.save()

        generator = EmbeddingCandidateGenerator(embedding_model_name=str(out_path))
        results = generator.generate(query=query_doc_2, source=source_doc_4, top_k=2)

        assert set(results.keys()) == {seg.id for seg in query_doc_2}
        for candidates in results.values():
            assert len(candidates) == 2
            for candidate in candidates:
                assert -1.0 <= candidate.score <= 1.0

    def test_empty_training_data_raises(self, tiny_config, query_doc_2, source_doc_4):
        from locisimiles.training.embedding import EmbeddingTrainer

        empty_data = TrainingData(query_doc_2, source_doc_4, GroundTruth())
        trainer = EmbeddingTrainer(tiny_config)
        with pytest.raises(ValueError, match="No training rows found"):
            trainer.fit(data=empty_data)

    def test_save_before_fit_raises(self, tiny_config):
        from locisimiles.training.embedding import EmbeddingTrainer

        trainer = EmbeddingTrainer(tiny_config)
        with pytest.raises(ValueError, match="No trained model available"):
            trainer.save()


class TestEmbeddingTrainerLossVariants:
    def test_contrastive_loss_runs(self, temp_dir, train_data):
        from locisimiles.training.embedding import EmbeddingTrainer, EmbeddingTrainerConfig

        cfg = EmbeddingTrainerConfig(
            output_dir=temp_dir,
            model_name=TINY_MODEL,
            epochs=1,
            batch_size=2,
            loss_type="contrastive",
        )
        trainer = EmbeddingTrainer(cfg)
        trainer.fit(data=train_data)
        assert trainer.model is not None

    def test_unknown_loss_type_raises(self, temp_dir, train_data):
        from locisimiles.training.embedding import EmbeddingTrainer, EmbeddingTrainerConfig

        cfg = EmbeddingTrainerConfig(
            output_dir=temp_dir,
            model_name=TINY_MODEL,
            loss_type="bogus",  # type: ignore[arg-type]
        )
        trainer = EmbeddingTrainer(cfg)
        with pytest.raises(ValueError, match="Unknown loss_type"):
            trainer.fit(data=train_data)


class TestEmbeddingTrainerEvalData:
    def test_eval_data_runs_binary_evaluator_without_error(
        self, temp_dir, query_doc_2, source_doc_4, train_data
    ):
        from locisimiles.training.embedding import EmbeddingTrainer, EmbeddingTrainerConfig

        eval_gt = GroundTruth(
            [
                {"query_id": "q1", "source_id": "s1", "label": "match"},
                {"query_id": "q2", "source_id": "s4", "label": "no_match"},
            ]
        )
        eval_data = TrainingData(query_doc_2, source_doc_4, eval_gt)

        cfg = EmbeddingTrainerConfig(
            output_dir=temp_dir, model_name=TINY_MODEL, epochs=1, batch_size=2
        )
        trainer = EmbeddingTrainer(cfg)
        model = trainer.fit(data=train_data, eval_data=eval_data)
        assert model is not None


class TestEmbeddingTrainerLoadArtifacts:
    def test_load_artifacts_round_trip(self, tiny_config, train_data):
        from locisimiles.training.embedding import EmbeddingTrainer

        trainer = EmbeddingTrainer(tiny_config)
        trainer.fit(data=train_data)
        out_path = trainer.save()

        reloaded = EmbeddingTrainer(tiny_config)
        model = reloaded.load_artifacts(out_path)
        assert model is not None
        embedding = model.encode(["Arma virumque cano."], prompt_name="query")
        assert embedding.shape[0] == 1
