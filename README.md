<div align="center">
  <img src="server/public/logo.png" alt="CodeSaviour" height="110" />
  <h1>CodeSaviour 2.0</h1>
  <p><strong>AI-powered code analysis and instant fixes</strong></p>
  
  <a href="https://5972a490.codesaviour.pages.dev/" target="_blank">
    <img src="https://img.shields.io/badge/Live%20Demo-Open-00C853?style=for-the-badge&logo=firefoxbrowser" alt="Live Demo" />
  </a>
  <br />
  <img src="https://img.shields.io/badge/Cloudflare-Pages-F38020?style=flat-square&logo=cloudflare" alt="Cloudflare Pages" />
  <img src="https://img.shields.io/badge/Cloudflare-Workers-F38020?style=flat-square&logo=cloudflare" alt="Cloudflare Workers" />
  <img src="https://img.shields.io/badge/OpenRouter-API-1A1A1A?style=flat-square" alt="OpenRouter" />
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/Frontend-HTML%2FCSS%2FJS-F7DF1E?style=flat-square&logo=javascript&logoColor=black" alt="Frontend" />
</div>

---

> 🚀 <strong>Try it now:</strong> Experience Deep Scan and instant analysis in your browser.
>
> 🎯 <a href="https://5972a490.codesaviour.pages.dev/" target="_blank"><strong>Open the Live Demo →</strong></a>

---

## Highlights
- 🔎 Smart Deep Scan via Cloudflare Worker (OpenRouter)
- 🧭 Clean dashboard for quick, readable reports
- 🧰 Optional FastAPI utilities (analyzers + unified diff apply)
- ✅ Tests for parsers and diff application (`pytest`)
- 🔐 Secrets handled via Wrangler; no API keys in repo

## Quick Start

### 1) Cloudflare Worker (Production API)
- Install Wrangler:
  - https://developers.cloudflare.com/workers/wrangler/install-and-update/
- Set secrets:
  - `wrangler secret put OPENROUTER_API_KEY`
  - Optional: `wrangler secret put OPENROUTER_MODEL` (e.g. `qwen/qwen-2.5-coder-32b-instruct:free`)
- Deploy:
  - `wrangler deploy`

Endpoints provided:
- `GET /api/status` — health + model info
- `POST /api/analyze` — returns structured JSON report

### 2) Frontend (Dashboard)
Static site files: `index.html`, `dashboard.html`, `landing.js`, `dashboard.js`, `styles.css`.
- Host anywhere (Cloudflare Pages recommended)
- Point the frontend to your Worker URL if needed
- Or just use the Live Demo:
  - `https://5972a490.codesaviour.pages.dev/`

### 3) Local FastAPI Server (Optional)
For analyzer/diff utility testing locally:
```
python -m venv .venv
. .venv/Scripts/Activate.ps1   # Windows PowerShell
pip install -r server/requirements.txt
python server/start_with_key.py
```
- Ensure `FIREWORKS_API_KEY` (and optionally `OPENROUTER_API_KEY`) are set via local `.env`.

## Configuration & Secrets
Use `.env.example` for local development only. For production, use Wrangler secrets.
- `.env` and `server/.env` are ignored by `.gitignore`
- Do not commit real API keys

Common variables:
```
# OpenRouter (Worker + optional local usage)
OPENROUTER_API_KEY=
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_MODEL=qwen/qwen-2.5-coder-32b-instruct:free

# Fireworks (optional local server)
FIREWORKS_API_KEY=
FIREWORKS_BASE_URL=https://api.fireworks.ai/inference/v1
FIREWORKS_MODEL=accounts/fireworks/models/qwen2p5-coder-32b-instruct
```

## Tests
Run Python tests for utilities:
```
pytest -q
```

## Project Structure
- `server/` — optional FastAPI utilities, analyzers, tests
- `dashboard.html`, `dashboard.js` — Deep Scan UI
- `server/worker.js` — Cloudflare Worker
- `wrangler.toml` — Worker config

## Contributing
- Keep changes minimal and focused
- Add/update tests where reasonable
- Open a PR with a clear summary

---

<div align="center">
  <a href="https://5972a490.codesaviour.pages.dev/" target="_blank">
    <img src="https://img.shields.io/badge/Try%20CodeSaviour%20Now-Open%20Demo-0A84FF?style=for-the-badge" alt="Try CodeSaviour" />
  </a>
  <p>✨ Clean code, fast fixes, fewer bugs.</p>
</div>