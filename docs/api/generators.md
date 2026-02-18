# Generators

Candidate generators narrow the search space by selecting source segments
that are most likely to be relevant for each query segment.

All generators inherit from `CandidateGeneratorBase` and implement a
`generate()` method returning `CandidateGeneratorOutput`.

## CandidateGeneratorBase

::: locisimiles.pipeline.generator._base.CandidateGeneratorBase
    options:
      heading_level: 3

## EmbeddingCandidateGenerator

Generate candidates using semantic embedding similarity with sentence transformers and ChromaDB.

::: locisimiles.pipeline.generator.embedding.EmbeddingCandidateGenerator
    options:
      heading_level: 3

## ExhaustiveCandidateGenerator

Return all source segments as candidates (no filtering).

::: locisimiles.pipeline.generator.exhaustive.ExhaustiveCandidateGenerator
    options:
      heading_level: 3

## RuleBasedCandidateGenerator

Generate candidates using lexical matching and linguistic filters.

::: locisimiles.pipeline.generator.rule_based.RuleBasedCandidateGenerator
    options:
      heading_level: 3
