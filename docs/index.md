# LociSimiles

**A Python package for extracting intertextualities in Latin literature using pre-trained language models.**

LociSimiles enables researchers to detect textual reuse, quotations, and allusions between Latin texts, from verbatim citations to subtle paraphrases and thematic echoes.

## Quick Start

### Installation

```bash
pip install locisimiles
```

Or install with development dependencies:

```bash
pip install "locisimiles[dev]"
```

### Basic Usage

```python
from locisimiles import Document, RetrievalPipeline

# Load your documents
source = Document.from_csv("source_texts.csv")
target = Document.from_csv("target_texts.csv")

# Create retrieval pipeline
pipeline = RetrievalPipeline()

# Find similar passages
results = pipeline.retrieve(source, target, top_k=5)
```

## Documentation

- [Getting Started](getting-started.md) - Installation and first steps
- [Examples](examples.md) - Working examples and tutorials
- [CLI Reference](cli.md) - Command-line interface documentation
- [API Reference](api/index.md) - Complete API documentation
- [Development](development.md) - Contributing and development setup

## Authors

- **Julian Schelb** - University of Konstanz
- **Michael Wittweiler** - University of Zurich

## Citation

If you use LociSimiles in your research, please cite our paper:

```bibtex
@article{schelb2026locisimiles,
  title={Loci Similes: A Benchmark for Extracting Intertextualities in Latin Literature},
  author={Schelb, Julian and Wittweiler, Michael and Revellio, Marie and Feichtinger, Barbara and Spitz, Andreas},
  journal={arXiv preprint arXiv:2601.07533},
  year={2026}
}
```

## License

This project is licensed under the MIT License.
