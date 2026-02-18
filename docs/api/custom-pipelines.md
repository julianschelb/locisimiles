# Custom Pipelines

Build your own pipeline by combining any generator with any judge.

## Pipeline

::: locisimiles.pipeline.pipeline.Pipeline
    options:
      heading_level: 3

## Type Definitions

Data classes and type aliases used across all pipelines.

::: locisimiles.pipeline._types
    options:
      heading_level: 3
      show_category_heading: false
      members:
        - Candidate
        - CandidateJudge
        - CandidateGeneratorOutput
        - CandidateJudgeOutput
        - CandidateJudgeInput
        - pretty_print
        - results_to_csv
        - results_to_json
