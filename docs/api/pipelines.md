# Pipelines Module

Processing pipelines for intertextual detection.

## Pipeline (Composer)

Compose any generator and judge into a full pipeline.

::: locisimiles.pipeline.pipeline.Pipeline
    options:
      heading_level: 3

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

## RuleBasedPipeline

Rule-based pipeline using lexical matching and linguistic filters.

::: locisimiles.pipeline.rule_based.RuleBasedPipeline
    options:
      heading_level: 3

## Type Definitions

Data classes for pipeline results.

::: locisimiles.pipeline._types
    options:
      heading_level: 3
      show_category_heading: false
