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
