# Pipelines

Ready-to-use pipelines for detecting intertextual parallels in Latin literature.

Each pipeline loads its own models and exposes a single `run()` method that
accepts two `Document` objects and returns scored results.

## TwoStagePipeline

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

## Word2VecRetrievalPipeline

Burns-style Word2Vec bigram retrieval + threshold judge.

::: locisimiles.pipeline.word2vec.Word2VecRetrievalPipeline
    options:
      heading_level: 3
      show_root_heading: false

## LatinBertRetrievalPipeline

Gong-style contextual Latin BERT token retrieval + threshold judge.

::: locisimiles.pipeline.contextual_retrieval.LatinBertRetrievalPipeline
    options:
      heading_level: 3
      show_root_heading: false

## LatinBertTwoStagePipeline

Contextual Latin BERT retrieval + classification.

::: locisimiles.pipeline.contextual_two_stage.LatinBertTwoStagePipeline
    options:
      heading_level: 3
      show_root_heading: false

## TfidfRetrievalPipeline

TF-IDF lexical retrieval + threshold judge.

::: locisimiles.pipeline.tfidf.TfidfRetrievalPipeline
    options:
      heading_level: 3
      show_root_heading: false

## BM25RetrievalPipeline

Okapi BM25 lexical retrieval + threshold judge — the benchmark's best retriever.

::: locisimiles.pipeline.bm25.BM25RetrievalPipeline
    options:
      heading_level: 3
      show_root_heading: false

## BM25TwoStagePipeline

BM25 retrieval + classification — the benchmark's "best combined" configuration.

::: locisimiles.pipeline.bm25_two_stage.BM25TwoStagePipeline
    options:
      heading_level: 3
      show_root_heading: false

## BM25LexicalTwoStagePipeline

BM25 retrieval + trained LogReg/GBDT lexical classifier — the benchmark's
"best non-neural" configuration. No neural model required end to end.

::: locisimiles.pipeline.bm25_lexical_two_stage.BM25LexicalTwoStagePipeline
    options:
      heading_level: 3
      show_root_heading: false
