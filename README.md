# CodeSaviour 2.0

CodeSaviour is a small FastAPI-based service that fixes code snippets using OpenRouter models, with built-in analyzers and utilities for applying unified diffs. It includes CI validation and unit tests.

## Features
- FastAPI endpoint for code fixing with OpenRouter
- Analyzer integration with path overrides via environment variables
- Unified diff applier with multi-hunk conflict handling
- CI workflow to validate patches (`git apply --3way --check`) and upload logs
- Unit tests with `pytest`

## Getting Started

### Requirements
- Python `3.11`
- Recommended: virtual environment (`python -m venv .venv`)

### Installation
```
python -m venv .venv
. .venv/Scripts/Activate.ps1   # Windows PowerShell
pip install -r server/requirements.txt
```

### Configuration
Create a `.env` file in the repository root (or `server/.env`). Environment variables supported:

```
# OpenRouter configuration
OPENROUTER_API_KEY=
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_MODEL=tngtech/deepseek-r1t-chimera:free
OPENROUTER_MODELS=tngtech/deepseek-r1t-chimera:free,tngtech/deepseek-r1t2-chimera:free,deepseek/deepseek-chat-v3.1:free

# Site
SITE_URL=http://127.0.0.1:8000/
SITE_TITLE=CodeSaviour

# Analyzer path overrides (optional)
FLAKE8_PATH=
ESLINT_PATH=
CPP_CHECK_PATH=
CLANG_TIDY_PATH=
PMD_PATH=
```

The app loads `.env` automatically at import time without overwriting already-set env vars.

### Running the API
```
python server/start_with_key.py
```
- The start script will run Uvicorn on `http://127.0.0.1:8001`.
- Ensure `OPENROUTER_API_KEY` is set in your environment or `.env`.

### Tests
Run unit tests:
```
pytest -q
```
All tests are under `server/tests/`.

## CI: Patch Validation & Tests
The repository includes `.github/workflows/patch-validate.yml` which:
- Validates patches in `patches/` via `git apply --3way --check`
- Uploads `patch-logs` as an artifact for troubleshooting
- Sets up Python and runs `pytest -q`

## Contributing
- Use feature branches and open PRs
- Keep changes minimal and focused
- Add or update tests where appropriate

## License
This project currently has no explicit license. If you plan to distribute, consider adding one (e.g., MIT).