# pipeline/pipeline.py
"""Generic pipeline composer: generator + judge."""
from __future__ import annotations

from typing import Any

from locisimiles.document import Document
from locisimiles.pipeline._types import CandidateGeneratorOutput, CandidateJudgeOutput
from locisimiles.pipeline.generator._base import CandidateGeneratorBase
from locisimiles.pipeline.judge._base import JudgeBase


class Pipeline:
    """Compose a candidate generator and a judge into a full pipeline.

    This is the recommended way to build custom pipelines.  Any
    :class:`~locisimiles.pipeline.generator.CandidateGeneratorBase`
    can be paired with any
    :class:`~locisimiles.pipeline.judge.JudgeBase`.

    Args:
        generator: Candidate-generation component.
        judge: Scoring / classification component.

    Example:
        ```python
        from locisimiles.pipeline import Pipeline
        from locisimiles.pipeline.generator import EmbeddingCandidateGenerator
        from locisimiles.pipeline.judge import ClassificationJudge

        pipeline = Pipeline(
            generator=EmbeddingCandidateGenerator(device="cpu"),
            judge=ClassificationJudge(device="cpu"),
        )
        results = pipeline.run(query=query_doc, source=source_doc, top_k=10)
        ```

    Equivalent legacy pipelines composed with the new API:

        ```python
        # ClassificationPipelineWithCandidategeneration
        Pipeline(EmbeddingCandidateGenerator(), ClassificationJudge())

        # ClassificationPipeline (exhaustive)
        Pipeline(ExhaustiveCandidateGenerator(), ClassificationJudge())

        # RetrievalPipeline
        Pipeline(EmbeddingCandidateGenerator(), ThresholdJudge())

        # RuleBasedPipeline
        Pipeline(RuleBasedCandidateGenerator(), IdentityJudge())
        ```
    """

    def __init__(
        self,
        generator: CandidateGeneratorBase,
        judge: JudgeBase,
    ):
        self.generator = generator
        self.judge = judge

        # Cache last intermediate results
        self._last_candidates: CandidateGeneratorOutput | None = None
        self._last_judgments: CandidateJudgeOutput | None = None

    def generate_candidates(
        self,
        *,
        query: Document,
        source: Document,
        **kwargs: Any,
    ) -> CandidateGeneratorOutput:
        """Run only the candidate-generation stage.

        Args:
            query: Query document.
            source: Source document.
            **kwargs: Forwarded to the generator's ``generate()`` method.

        Returns:
            ``CandidateGeneratorOutput`` mapping query IDs → ``Candidate`` lists.
        """
        self._last_candidates = self.generator.generate(
            query=query, source=source, **kwargs
        )
        return self._last_candidates

    def judge_candidates(
        self,
        *,
        query: Document,
        candidates: CandidateGeneratorOutput,
        **kwargs: Any,
    ) -> CandidateJudgeOutput:
        """Run only the judgment stage on pre-generated candidates.

        Args:
            query: Query document.
            candidates: Output from a candidate generator.
            **kwargs: Forwarded to the judge's ``judge()`` method.

        Returns:
            ``CandidateJudgeOutput`` mapping query IDs → ``CandidateJudge`` lists.
        """
        self._last_judgments = self.judge.judge(
            query=query, candidates=candidates, **kwargs
        )
        return self._last_judgments

    def run(
        self,
        *,
        query: Document,
        source: Document,
        **kwargs: Any,
    ) -> CandidateJudgeOutput:
        """Run both stages: generate candidates then judge them.

        All *kwargs* are forwarded to **both** the generator and the judge;
        each component ignores keys it does not recognise.

        Args:
            query: Query document.
            source: Source document.
            **kwargs: Forwarded to both stages.

        Returns:
            ``CandidateJudgeOutput`` with judgment scores.
        """
        candidates = self.generate_candidates(query=query, source=source, **kwargs)
        return self.judge_candidates(query=query, candidates=candidates, **kwargs)
