# pipeline/pipeline.py
"""Generic pipeline composer: generator + judge."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Union

from locisimiles.document import Document
from locisimiles.pipeline._types import (
    CandidateGeneratorOutput,
    CandidateJudgeOutput,
    results_to_csv,
    results_to_json,
)
from locisimiles.pipeline.generator._base import CandidateGeneratorBase
from locisimiles.pipeline.judge._base import JudgeBase


class Pipeline:
    """Compose a candidate generator and a judge into a full pipeline.

    This is the recommended way to build custom pipelines.  Any
    ``CandidateGeneratorBase`` can be paired with any ``JudgeBase``.

    Args:
        generator: Candidate-generation component.
        judge: Scoring / classification component.

    Example:
        ```python
        from locisimiles.pipeline import Pipeline
        from locisimiles.pipeline.generator import EmbeddingCandidateGenerator
        from locisimiles.pipeline.judge import ClassificationJudge
        from locisimiles.document import Document

        # Load documents
        query = Document("query.csv")
        source = Document("source.csv")

        # Build a custom pipeline
        pipeline = Pipeline(
            generator=EmbeddingCandidateGenerator(device="cpu"),
            judge=ClassificationJudge(device="cpu"),
        )

        # Run pipeline
        results = pipeline.run(query=query, source=source, top_k=10)

        # Save results
        pipeline.to_csv("results.csv")
        pipeline.to_json("results.json")
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

    # ---------- Result I/O ----------

    def to_csv(
        self,
        path: Union[str, Path],
        results: CandidateJudgeOutput | None = None,
    ) -> None:
        """Save pipeline results to a CSV file.

        If *results* is ``None``, the results from the last ``run()`` call
        are used.

        Columns: ``query_id``, ``source_id``, ``source_text``,
        ``candidate_score``, ``judgment_score``.

        Args:
            path: Destination file path (e.g. ``"results.csv"``).
            results: Explicit results to save.  Defaults to the last
                ``run()`` output.

        Raises:
            ValueError: If no results are available.

        Example:
            ```python
            results = pipeline.run(query=query_doc, source=source_doc)
            pipeline.to_csv("results.csv")
            ```
        """
        data = results if results is not None else self._last_judgments
        if data is None:
            raise ValueError(
                "No results to save. Call run() first or pass results explicitly."
            )
        results_to_csv(data, path)

    def to_json(
        self,
        path: Union[str, Path],
        results: CandidateJudgeOutput | None = None,
        *,
        indent: int = 2,
    ) -> None:
        """Save pipeline results to a JSON file.

        If *results* is ``None``, the results from the last ``run()`` call
        are used.

        Produces a JSON object keyed by query segment ID, where each value
        is a list of match objects with ``source_id``, ``source_text``,
        ``candidate_score``, and ``judgment_score``.

        Args:
            path: Destination file path (e.g. ``"results.json"``).
            results: Explicit results to save.  Defaults to the last
                ``run()`` output.
            indent: JSON indentation level (default ``2``).

        Raises:
            ValueError: If no results are available.

        Example:
            ```python
            results = pipeline.run(query=query_doc, source=source_doc)
            pipeline.to_json("results.json")
            ```
        """
        data = results if results is not None else self._last_judgments
        if data is None:
            raise ValueError(
                "No results to save. Call run() first or pass results explicitly."
            )
        results_to_json(data, path, indent=indent)
