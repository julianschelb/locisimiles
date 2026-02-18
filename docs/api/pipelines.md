# Pipelines

Ready-to-use pipelines for detecting intertextual parallels in Latin literature.

Each pipeline loads its own models and exposes a single `run()` method that
accepts two `Document` objects and returns scored results.

## ClassificationPipelineWithCandidategeneration

::: locisimiles.pipeline.two_stage.TwoStagePipeline
    options:
      heading_level: 3
      show_root_heading: false

## ClassificationPipeline

::: locisimiles.pipeline.classification.ExhaustiveClassificationPipeline
    options:
      heading_level: 3
      show_root_heading: false

## RetrievalPipeline

::: locisimiles.pipeline.retrieval.RetrievalPipeline
    options:
      heading_level: 3

## RuleBasedPipeline

::: locisimiles.pipeline.rule_based.RuleBasedPipeline
    options:
      heading_level: 3
