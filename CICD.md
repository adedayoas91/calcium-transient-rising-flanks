# CI/CD Pipeline

This project uses GitHub Actions for continuous integration and deployment (CI/CD).

## Workflows

### 1. Test Workflow (`.github/workflows/test.yml`)

**Triggers**: Push to `main` or `devel`, Pull Requests to `main` or `devel`

**Steps**:
- Sets up Python 3.13
- Installs dependencies via `uv`
- Lints code with `ruff` (fails on issues)
- Type checks with `mypy` (warnings only, continues on error)
- Runs full test suite with `unittest`

**Environment Variables**:
- `PYTHONPATH=src` - Makes src module importable
- `MPLCONFIGDIR=/tmp/matplotlib-cache` - Matplotlib config directory
- `XDG_CACHE_HOME=/tmp/font-cache` - Font cache directory

### 2. Build Workflow (`.github/workflows/build.yml`)

**Triggers**: Push to `main` or `devel`, Pull Requests to `main` or `devel`

**Steps**:
- Sets up Python 3.13
- Installs `uv` package manager
- Builds package with `uv build`
- Uploads build artifacts (wheel and sdist)

**Artifacts**: `dist/` directory with built packages

### 3. Release Workflow (`.github/workflows/release.yml`)

**Triggers**: Push of version tags (e.g., `v1.0.0`)

**Steps**:
- Builds the package
- Creates GitHub Release with build artifacts
- Generates release notes automatically

**Usage**:
```bash
git tag -a v1.0.0 -m "Release version 1.0.0"
git push origin v1.0.0
```

### 4. Documentation Workflow (`.github/workflows/docs.yml`)

**Triggers**: Push to `main` or `devel`, Pull Requests to `main` or `devel`

**Steps**:
- Verifies README.md exists
- Checks for CLAUDE.md (optional)
- Validates markdown structure

## Branch Strategy

- **`main`** - Production-ready code, protected branch
  - All changes via Pull Requests
  - Must pass all CI checks
  - Releases via tags

- **`devel`** - Development branch
  - Integration point for features
  - Must pass all CI checks
  - Changes via Pull Requests recommended

## Local Development

### Running Tests Locally

```bash
PYTHONPATH=src MPLCONFIGDIR=/tmp/matplotlib-cache \
  python -m unittest discover -s tests -v
```

### Running Linting

```bash
uv run ruff check src/ tests/ examples/
```

### Running Type Checks

```bash
uv run mypy src/ --ignore-missing-imports
```

### Building Package

```bash
uv build
```

## Requirements

- Python 3.13 (as specified in `.python-version`)
- `uv` package manager
- Dependencies in `pyproject.toml`

## GitHub Secrets

No secrets are currently required. If adding PyPI publishing, configure:
- `PYPI_API_TOKEN` - PyPI API token for publishing

## Future Enhancements

- Code coverage reporting
- Automated PyPI publishing
- Docker image building
- Deployment automation
- Performance benchmarking

## References

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [uv Documentation](https://docs.astral.sh/uv/)
- [Ruff Documentation](https://docs.astral.sh/ruff/)
- [mypy Documentation](https://mypy.readthedocs.io/)
