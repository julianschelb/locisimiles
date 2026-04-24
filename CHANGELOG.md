# CHANGELOG

<!-- version list -->

## v1.7.0 (2026-04-24)

### Bug Fixes

- Add blank line for better readability in TestDocumentMethods
  ([`70a3844`](https://github.com/julianschelb/locisimiles/commit/70a38442a375f1d72fd13af12b99edc2aa339a31))

- Improve readability of Document class test cases by formatting long lines
  ([#13](https://github.com/julianschelb/locisimiles/pull/13),
  [`07aac26`](https://github.com/julianschelb/locisimiles/commit/07aac2660b66c7e54f1003b8073c3758ca3959fe))

### Features

- Add head() method to Document class for segment inspection and corresponding tests
  ([#13](https://github.com/julianschelb/locisimiles/pull/13),
  [`07aac26`](https://github.com/julianschelb/locisimiles/commit/07aac2660b66c7e54f1003b8073c3758ca3959fe))

- Enhance Document class with DataFrame conversion methods and serialization support
  ([#13](https://github.com/julianschelb/locisimiles/pull/13),
  [`07aac26`](https://github.com/julianschelb/locisimiles/commit/07aac2660b66c7e54f1003b8073c3758ca3959fe))


## v1.6.0 (2026-04-24)

### Bug Fixes

- Streamline CSV formatting in test_document.py for cleaner input
  ([#14](https://github.com/julianschelb/locisimiles/pull/14),
  [`625e882`](https://github.com/julianschelb/locisimiles/commit/625e882560f92353f969e174259a50a7f13cb433))

### Features

- Implement basic text cleanup methods in Document class
  ([#14](https://github.com/julianschelb/locisimiles/pull/14),
  [`625e882`](https://github.com/julianschelb/locisimiles/commit/625e882560f92353f969e174259a50a7f13cb433))


## v1.5.0 (2026-04-06)

### Features

- Add contextual Latin BERT pipelines for retrieval and two-stage classification
  ([#12](https://github.com/julianschelb/locisimiles/pull/12),
  [`1b84f3b`](https://github.com/julianschelb/locisimiles/commit/1b84f3b066f8e3ff3e7bdc61a0ef536febf718e5))

- Add Latin BERT contextual retrieval pipeline (Gong-style token-level similarity)
  ([#12](https://github.com/julianschelb/locisimiles/pull/12),
  [`1b84f3b`](https://github.com/julianschelb/locisimiles/commit/1b84f3b066f8e3ff3e7bdc61a0ef536febf718e5))

- Add Word2Vec example notebook and update documentation
  ([#12](https://github.com/julianschelb/locisimiles/pull/12),
  [`1b84f3b`](https://github.com/julianschelb/locisimiles/commit/1b84f3b066f8e3ff3e7bdc61a0ef536febf718e5))

- Add Word2Vec generic smoketest example and model training
  ([#12](https://github.com/julianschelb/locisimiles/pull/12),
  [`1b84f3b`](https://github.com/julianschelb/locisimiles/commit/1b84f3b066f8e3ff3e7bdc61a0ef536febf718e5))

- Add Word2Vec retrieval pipeline and candidate generator
  ([#12](https://github.com/julianschelb/locisimiles/pull/12),
  [`1b84f3b`](https://github.com/julianschelb/locisimiles/commit/1b84f3b066f8e3ff3e7bdc61a0ef536febf718e5))

- Enhance pipeline type annotations and improve scoring logic in candidate generators
  ([#12](https://github.com/julianschelb/locisimiles/pull/12),
  [`1b84f3b`](https://github.com/julianschelb/locisimiles/commit/1b84f3b066f8e3ff3e7bdc61a0ef536febf718e5))

- Reorganize imports and enhance Word2Vec and contextual BERT parameter configuration in GUI
  ([#12](https://github.com/julianschelb/locisimiles/pull/12),
  [`1b84f3b`](https://github.com/julianschelb/locisimiles/commit/1b84f3b066f8e3ff3e7bdc61a0ef536febf718e5))


## v1.4.0 (2026-03-15)

### Bug Fixes

- Correct casing of ClassificationPipelineWithCandidateGeneration and update related references
  ([#11](https://github.com/julianschelb/locisimiles/pull/11),
  [`381afcc`](https://github.com/julianschelb/locisimiles/commit/381afccbb8b0fce032de9e42c8d41e897cb10586))

### Features

- Add multi-pipeline model options and update default models/threshold to paper best
  baselinesFeature/additional models ([#11](https://github.com/julianschelb/locisimiles/pull/11),
  [`381afcc`](https://github.com/julianschelb/locisimiles/commit/381afccbb8b0fce032de9e42c8d41e897cb10586))

- Update classification threshold to 0.85 in CLI and GUI components
  ([#11](https://github.com/julianschelb/locisimiles/pull/11),
  [`381afcc`](https://github.com/julianschelb/locisimiles/commit/381afccbb8b0fce032de9e42c8d41e897cb10586))

- Update default models in documentation and examples to xlm-roberta and multilingual-e5
  ([#11](https://github.com/julianschelb/locisimiles/pull/11),
  [`381afcc`](https://github.com/julianschelb/locisimiles/commit/381afccbb8b0fce032de9e42c8d41e897cb10586))


## v1.3.0 (2026-02-27)

### Features

- Update .gitignore and pyproject.toml for Python 3.14 support
  ([#10](https://github.com/julianschelb/locisimiles/pull/10),
  [`bdb7b9e`](https://github.com/julianschelb/locisimiles/commit/bdb7b9ee0ad9110d5b1f02ff8c32b968f8bd9047))


## v1.2.0 (2026-02-25)

### Features

- Add example notebook and data files for Document class features demonstration
  ([#9](https://github.com/julianschelb/locisimiles/pull/9),
  [`939921b`](https://github.com/julianschelb/locisimiles/commit/939921b37cbdffa76b0d9138db546434df567aed))

- Add export methods for Document class to save segments as plain text and CSV
  ([#9](https://github.com/julianschelb/locisimiles/pull/9),
  [`939921b`](https://github.com/julianschelb/locisimiles/commit/939921b37cbdffa76b0d9138db546434df567aed))

- Add statistics method to Document class with corresponding tests
  ([#9](https://github.com/julianschelb/locisimiles/pull/9),
  [`939921b`](https://github.com/julianschelb/locisimiles/commit/939921b37cbdffa76b0d9138db546434df567aed))

- Enhance GUI with support for multiple pipeline configurations and visibility toggling
  ([#8](https://github.com/julianschelb/locisimiles/pull/8),
  [`72d2363`](https://github.com/julianschelb/locisimiles/commit/72d2363512f5757f7ec86edc0fb9afb7f4628127))

- Implement sentencization feature in Document class with comprehensive tests
  ([#9](https://github.com/julianschelb/locisimiles/pull/9),
  [`939921b`](https://github.com/julianschelb/locisimiles/commit/939921b37cbdffa76b0d9138db546434df567aed))


## v1.1.0 (2026-02-18)

### Features

- Add built-in example datasets and loaders for quick experimentation
  ([#7](https://github.com/julianschelb/locisimiles/pull/7),
  [`6809635`](https://github.com/julianschelb/locisimiles/commit/6809635d5d0c3cfacee962ce57a515ed191e11a6))


## v1.0.1 (2026-02-18)

### Bug Fixes

- Enhance release workflow to handle semantic versioning and artifact publishing
  ([#6](https://github.com/julianschelb/locisimiles/pull/6),
  [`ea3461e`](https://github.com/julianschelb/locisimiles/commit/ea3461eedbae664a3d52467fe8b0dd8614ff2ca6))

- Enhance release workflow to handle semantic versioning and artif…
  ([#6](https://github.com/julianschelb/locisimiles/pull/6),
  [`ea3461e`](https://github.com/julianschelb/locisimiles/commit/ea3461eedbae664a3d52467fe8b0dd8614ff2ca6))

### Chores

- Add caching for pip in CI workflow to improve build performance
  ([#6](https://github.com/julianschelb/locisimiles/pull/6),
  [`ea3461e`](https://github.com/julianschelb/locisimiles/commit/ea3461eedbae664a3d52467fe8b0dd8614ff2ca6))

- Add CODEOWNERS file for change approval process
  ([`722c76c`](https://github.com/julianschelb/locisimiles/commit/722c76cccc6d6902b71dfb4ff887554abf7d3614))


## v1.0.0 (2026-02-18)

- Initial Release
