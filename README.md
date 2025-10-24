# CodeSaviour 2.0

CodeSaviour is a small developer tool that analyzes and fixes code snippets. It includes a Cloudflare Worker API (OpenRouter-based Deep Scan) and a lightweight frontend dashboard. A local FastAPI server (optional) is provided for testing analyzers and diff utilities.

## Features
- Deep Scan via OpenRouter through a Cloudflare Worker endpoint
- Frontend dashboard for quick code analysis and reporting
- Optional FastAPI server with analyzers and unified diff apply utilities
- Unit tests (`pytest`) for parsers and diff application

## Quick Start

### 1) Cloudflare Worker (Production API)
- Install Wrangler: https://developers.cloudflare.com/workers/wrangler/install-and-update/
- Set secrets:
  - `wrangler secret put OPENROUTER_API_KEY`
  - Optionally: `wrangler secret put OPENROUTER_MODEL` (e.g. `qwen/qwen-2.5-coder-32b-instruct:free`)
- Deploy:
  - `wrangler deploy`

The Worker exposes `/api/analyze` and `/api/status`. The model can be overridden via the secret `OPENROUTER_MODEL`.

### 2) Frontend (Dashboard)
The HTML/JS/CSS files (`index.html`, `dashboard.html`, `landing.js`, `dashboard.js`, `styles.css`) are static and can be hosted on any static host.
- Cloudflare Pages or any static server works.
- Update your frontend to point to your deployed Worker URL if needed.

### 3) Local FastAPI Server (Optional)
If you want to test the Python analyzers and diff utilities locally:
```
python -m venv .venv
. .venv/Scripts/Activate.ps1   # Windows PowerShell
pip install -r server/requirements.txt
python server/start_with_key.py
```
- Ensure `FIREWORKS_API_KEY` (and optionally `OPENROUTER_API_KEY`) are set in your environment or a local `.env`.

## Configuration
Use `.env.example` as a reference and create a local `.env` if running the Python server. For production, use Wrangler secrets. Do not commit real keys.

## Tests
Run unit tests (optional for Python utilities):
```
pytest -q
```

## Security & Hygiene
- Secrets are managed via Wrangler; `.env` files are ignored by `.gitignore`.
- Temporary and Wrangler state directories are ignored.
- No API keys are committed to the repository.

## Contributing
- Keep changes minimal and focused
- Add or update tests where appropriate
- Open a PR with a clear summary of changes