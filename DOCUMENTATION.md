# Documentation Guide

This project uses [MkDocs](https://www.mkdocs.org/) with the [Material theme](https://squidfunk.github.io/mkdocs-material/) and [mkdocstrings](https://mkdocstrings.github.io/) for auto-generated API docs.

## Quick Commands

```bash
poe docs          # Serve locally at http://127.0.0.1:8000
poe docs-build    # Build static site to site/
poe docs-deploy   # Deploy to GitHub Pages
```

## Updating Documentation

### Manual Pages

Edit markdown files in the `docs/` folder:

| File | Content |
|------|---------|
| `docs/index.md` | Homepage |
| `docs/getting-started.md` | Installation & usage guide |
| `docs/cli.md` | CLI reference |
| `docs/development.md` | Development guide |

### API Documentation (Auto-generated)

API docs in `docs/api/` are generated from source code docstrings. To update them:

1. Add/update docstrings in your Python code (Google style)
2. Rebuild the docs

Example docstring format:

```python
def my_function(param: str) -> int:
    """Short description.
    
    Args:
        param: Description of parameter.
    
    Returns:
        Description of return value.
    """
```

### Adding New Pages

1. Create a new `.md` file in `docs/`
2. Add it to `nav` in `mkdocs.yml`

## Publishing

```bash
poe docs-deploy
```

This builds the site and pushes to the `gh-pages` branch. GitHub Pages serves it automatically.

## Configuration

- `mkdocs.yml` - Site configuration, navigation, theme settings
- `pyproject.toml` - Dependencies in `[project.optional-dependencies] dev`
