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

After all CI jobs pass on `main`, a **Deploy Docs** job publishes the
documentation to GitHub Pages.

## Semantic Versioning & Releases

The project uses [Conventional Commits](https://www.conventionalcommits.org/)
and [python-semantic-release](https://python-semantic-release.readthedocs.io/)
for fully automated version bumps and changelog generation.

### Commit Message Format

Every commit message must follow the Conventional Commits format:

```
<type>(<optional scope>): <description>

[optional body]

[optional footer(s)]
```

**Common types and their effect on versioning:**

| Prefix | Example | Version Bump |
|---|---|---|
| `fix:` | `fix: handle empty document` | Patch (0.3.6 → 0.3.7) |
| `feat:` | `feat: add XML export` | Minor (0.3.6 → 0.4.0) |
| `feat!:` or `BREAKING CHANGE:` | `feat!: rename Pipeline API` | Major (0.3.6 → 1.0.0) |
| `docs:` | `docs: update getting started` | No release |
| `chore:` | `chore: update deps` | No release |
| `test:` | `test: add evaluator tests` | No release |
| `refactor:` | `refactor: simplify judge` | No release |
| `ci:` | `ci: add mypy step` | No release |

A pre-commit hook validates your commit message automatically. To install:

```bash
pre-commit install --hook-type commit-msg
```

### How Releases Work

1. Merge a PR into `main`
2. CI runs all checks (lint, typecheck, test, docs)
3. If CI passes, the **Release** workflow runs automatically
4. `semantic-release` analyses commits since the last tag
5. If there are `feat:` or `fix:` commits, it:
   - Bumps the version in `pyproject.toml` and `__init__.py`
   - Generates/updates `CHANGELOG.md`
   - Creates a git tag (`v0.4.0`)
   - Creates a GitHub Release with the changelog

### Manual Version Check

```bash
# Preview what the next version would be (dry run)
semantic-release version --print
```

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
| Preview next version | `semantic-release version --print` |
# Test semantic versioning
