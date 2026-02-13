# Development Guide

This guide covers setting up a development environment and contributing to LociSimiles.

## Prerequisites

- Python 3.10 or higher
- Git
- pip package manager

## Setup

### Clone the Repository

```bash
git clone https://github.com/julianschelb/locisimiles.git
cd locisimiles
```

### Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Install Development Dependencies

```bash
pip install -e ".[dev]"
```

This installs the package in editable mode along with development tools:

- **pytest** - Testing framework
- **pytest-cov** - Coverage reporting
- **poethepoet** - Task runner
- **mkdocs** - Documentation
- **mkdocs-material** - Documentation theme

## Running Tests

### Using Poe (Recommended)

```bash
# Run all tests
poe test

# Run tests with coverage report
poe test-cov
```

### Using Pytest Directly

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_document.py

# Run specific test
pytest tests/test_document.py::TestDocument::test_from_csv

# Run with coverage
pytest --cov=locisimiles --cov-report=term-missing

# Generate HTML coverage report
pytest --cov=locisimiles --cov-report=html
```

## Documentation

### Serve Locally

```bash
poe docs
```

This starts a local server at `http://127.0.0.1:8000` with live reload.

### Build Static Site

```bash
poe docs-build
```

Output is written to the `site/` directory.

### Deploy to GitHub Pages

```bash
poe docs-deploy
```

This builds and pushes to the `gh-pages` branch.

## Project Structure

```
locisimiles/
├── src/
│   └── locisimiles/
│       ├── __init__.py          # Package exports
│       ├── cli.py               # Command-line interface
│       ├── document.py          # Document and TextSegment classes
│       ├── evaluator.py         # Evaluation metrics
│       └── pipeline/
│           ├── __init__.py      # Pipeline exports
│           ├── _types.py        # Type definitions
│           ├── classification.py # Classification pipeline
│           ├── retrieval.py     # Retrieval pipeline
│           └── two_stage.py     # Combined pipeline
├── tests/
│   ├── conftest.py              # Shared test fixtures
│   ├── test_document.py         # Document tests
│   ├── test_evaluator.py        # Evaluator tests
│   ├── test_cli.py              # CLI tests
│   ├── test_types.py            # Type definition tests
│   ├── test_retrieval.py        # Retrieval pipeline tests
│   ├── test_classification.py   # Classification pipeline tests
│   └── test_two_stage.py        # Two-stage pipeline tests
├── docs/                        # Documentation source
├── examples/                    # Example scripts and notebooks
├── pyproject.toml               # Project configuration
└── mkdocs.yml                   # Documentation configuration
```

## Code Style

### General Guidelines

- Follow PEP 8 style guidelines
- Use type hints for function signatures
- Write docstrings for public APIs
- Keep functions focused and small

### Example

```python
def compute_similarity(
    text_a: str,
    text_b: str,
    model: Optional[str] = None
) -> float:
    """Compute semantic similarity between two texts.
    
    Args:
        text_a: First text string.
        text_b: Second text string.
        model: Optional model name. Defaults to MiniLM.
    
    Returns:
        Similarity score between 0 and 1.
    
    Raises:
        ValueError: If either text is empty.
    """
    if not text_a or not text_b:
        raise ValueError("Texts cannot be empty")
    ...
```

## Testing Guidelines

### Writing Tests

- Place tests in the `tests/` directory
- Name test files `test_<module>.py`
- Name test classes `Test<ClassName>`
- Name test methods `test_<behavior>`

### Using Fixtures

Shared fixtures are defined in `conftest.py`:

```python
def test_document_loading(sample_csv_file):
    """Test loading document from CSV."""
    doc = Document.from_csv(sample_csv_file)
    assert len(doc) > 0
```

### Mocking External Dependencies

Use unittest.mock for ML models:

```python
from unittest.mock import MagicMock, patch

def test_retrieval_with_mock(mock_embedder):
    """Test retrieval with mocked embedding model."""
    pipeline = RetrievalPipeline()
    pipeline.model = mock_embedder
    # Test without actual model loading
```

## Contributing

### Workflow

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make changes and add tests
4. Ensure tests pass: `poe test`
5. Commit changes: `git commit -m "Add my feature"`
6. Push to fork: `git push origin feature/my-feature`
7. Open a Pull Request

### Pull Request Guidelines

- Include tests for new functionality
- Update documentation as needed
- Keep commits focused and atomic
- Write clear commit messages

## Available Poe Tasks

| Task | Command | Description |
|------|---------|-------------|
| `test` | `poe test` | Run all tests |
| `test-cov` | `poe test-cov` | Run tests with coverage |
| `docs` | `poe docs` | Serve documentation locally |
| `docs-build` | `poe docs-build` | Build documentation |
| `docs-deploy` | `poe docs-deploy` | Deploy to GitHub Pages |
