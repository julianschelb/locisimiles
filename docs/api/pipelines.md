# Pipelines Module

Processing pipelines for intertextual detection.

## RetrievalPipeline

Find similar passages using semantic embeddings.

::: locisimiles.pipeline.retrieval.RetrievalPipeline
    options:
      heading_level: 3

## ClassificationPipeline

Classify text pairs using transformer models (exhaustive comparison).

::: locisimiles.pipeline.classification.ClassificationPipeline
    options:
      heading_level: 3

## ClassificationPipelineWithCandidategeneration

Two-stage pipeline: retrieval for candidate generation, then classification.

::: locisimiles.pipeline.two_stage.ClassificationPipelineWithCandidategeneration
    options:
      heading_level: 3

## Type Definitions

Data classes for pipeline results.

::: locisimiles.pipeline._types
    options:
      heading_level: 3
      show_category_heading: false
