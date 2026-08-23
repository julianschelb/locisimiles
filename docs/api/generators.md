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

## Word2VecCandidateGenerator

Burns-style bigram similarity retrieval using a local gensim Word2Vec model.

::: locisimiles.pipeline.generator.word2vec.Word2VecCandidateGenerator
    options:
      heading_level: 3

## LatinBertContextualCandidateGenerator

Gong-style contextual token similarity retrieval using a BERT model.

::: locisimiles.pipeline.generator.contextual_bert.LatinBertContextualCandidateGenerator
    options:
      heading_level: 3

## TfidfCandidateGenerator

TF-IDF cosine similarity retrieval over (optionally lemmatized) Latin text.

::: locisimiles.pipeline.generator.tfidf.TfidfCandidateGenerator
    options:
      heading_level: 3

## BM25CandidateGenerator

Okapi BM25 retrieval over (optionally lemmatized) Latin text — the
benchmark's best single retriever.

::: locisimiles.pipeline.generator.bm25.BM25CandidateGenerator
    options:
      heading_level: 3
