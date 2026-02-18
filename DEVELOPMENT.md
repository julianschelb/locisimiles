# Development Guide

This document covers setup and workflows for contributing to LociSimiles.

## Prerequisites

- Python 3.10 - 3.13
- pip or poetry

## Installation

### Install with Development Dependencies

```bash
# Clone the repository
git clone https://github.com/your-username/locisimiles.git
cd locisimiles

# Install in editable mode with dev dependencies
pip install -e ".[dev]"
```

### Optional: Install GUI Dependencies

```bash
pip install -e ".[gui]"
```

### Install All Optional Dependencies

```bash
pip install -e ".[dev,gui]"
```

## Running Tests

### Using Task Runner (Recommended)

```bash
# Run all tests
poe test

# Run tests with coverage report
poe test-cov
```

## Linting & Formatting

The project uses [Ruff](https://docs.astral.sh/ruff/) for linting and formatting.

```bash
# Check for lint errors
poe lint

# Auto-format code
poe format

# Check formatting without modifying files
poe format-check
```

## Type Checking

[Mypy](https://mypy.readthedocs.io/) is configured for gradual type checking:

```bash
poe typecheck
```

## Pre-commit Hooks

[Pre-commit](https://pre-commit.com/) runs Ruff, mypy, and file-hygiene checks
automatically before every `git commit`.

### Setup (once)

```bash
pre-commit install
```

### Running Manually

```bash
# Run all hooks on every tracked file
pre-commit run --all-files

# Run a specific hook
pre-commit run ruff --all-files
pre-commit run ruff-format --all-files
pre-commit run mypy --all-files

# Run hooks only on staged files (same as a real commit)
pre-commit run
```

### Updating Hook Versions

```bash
pre-commit autoupdate
```

## Building Documentation

```bash
# Live-reload dev server
poe docs

# One-off build (strict mode)
poe docs-build
```

## Continuous Integration

A GitHub Actions workflow (`.github/workflows/ci.yml`) runs on every push and
pull request to `main`. It executes four parallel jobs:

| Job | Description |
|---|---|
| **Lint & Format** | Ruff check + format check |
| **Type Check** | Mypy on Python 3.12 |
| **Test** | Pytest on Python 3.10 – 3.13 |
| **Docs Build** | `mkdocs build --strict` |

## Quick Reference

| Task | Command |
|---|---|
| Run tests | `poe test` |
| Lint | `poe lint` |
| Format | `poe format` |
| Type check | `poe typecheck` |
| All pre-commit hooks | `pre-commit run --all-files` |
| Serve docs | `poe docs` |
| Build docs | `poe docs-build` |
