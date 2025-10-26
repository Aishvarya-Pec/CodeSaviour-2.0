<div align="center">
  <img src="server/public/logo.png" alt="CodeSaviour" height="110" />
  <h1>CodeSaviour 2.0</h1>
  <p><strong>AI-powered code analysis and instant fixes</strong></p>

  <a href="https://codesaviour2.vercel.app/" target="_blank">
    <img src="https://img.shields.io/badge/Live%20Demo-Open-00C853?style=for-the-badge&logo=vercel" alt="Live Demo" />
  </a>
  <br />
  <img src="https://img.shields.io/badge/Vercel-Serverless%20API-000000?style=flat-square&logo=vercel" alt="Vercel" />
  <img src="https://img.shields.io/badge/Fireworks%20AI-Integration-1A1A1A?style=flat-square" alt="Fireworks AI" />
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/Frontend-HTML%2FCSS%2FJS-F7DF1E?style=flat-square&logo=javascript&logoColor=black" alt="Frontend" />
</div>

---

> 🚀 <strong>Try it now:</strong> Run Fix Code and Deep Scan in your browser.
>
> 🎯 <a href="https://codesaviour2.vercel.app/" target="_blank"><strong>Open the Live Demo →</strong></a>

---

## Highlights
- 🔎 Deep Scan and instant fixes via same-origin Vercel Serverless Functions
- 🤖 First-class Fireworks AI integration (configurable models)
- 🧭 Clean dashboard with readable analysis reports
- ✅ Python utilities and tests for analyzers/diff application (`pytest`)
- 🔐 Secure headers and CSP via `vercel.json`

## Tech Stack
- Frontend: HTML5, CSS3 (Inter/Orbitron), vanilla JavaScript
- Backend: Vercel Serverless Functions (`api/fix.js`, `api/analyze.js`)
- AI: Fireworks AI (primary) with optional OpenRouter fallback
- Tooling: Node.js, Vercel CLI, custom cross-platform build script (`scripts/build.js`)
- Python utilities: FastAPI helpers, analyzers, tests (`server/`)
- Security/Perf: Content-Security-Policy, same-origin API calls, static asset optimization

## Quick Start (Local)
1. Install dependencies:
   ```bash
   npm install
   ```
2. Start local dev servers:
   ```bash
   # Static preview (frontend)
   npx serve -l 8015
   # Vercel dev (API + frontend)
   npx vercel dev --listen localhost:8016
   ```
3. Open `http://localhost:8016/` (same-origin API) or `http://localhost:8015/` (static only).

## Environment Variables
Add these in a local `.env` (for dev) and in Vercel → Project Settings → Environment Variables.

```
# Fireworks AI (required)
FIREWORKS_API_KEY=
FIREWORKS_BASE_URL=https://api.fireworks.ai/inference/v1
FIREWORKS_MODEL=accounts/fireworks/models/qwen2p5-coder-32b-instruct

# Optional OpenRouter
OPENROUTER_API_KEY=
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_MODEL=qwen/qwen-2.5-coder-32b-instruct:free
```

## Deploy to Production (Vercel)
1. Install Vercel CLI and login:
   ```bash
   npm i -g vercel
   vercel login
   ```
2. Set environment variables in Vercel Project Settings.
3. Deploy:
   ```bash
   npx vercel deploy --prod --yes
   ```
- Build configuration and CSP headers are defined in `vercel.json`.
- The frontend uses `window.location.origin` to call the API functions: `POST /api/fix`, `POST /api/analyze`.

## Tests (Python Utilities)
Run unit tests for parsers/diff utilities:
```bash
pytest -q
```

## Project Structure
- `api/` — Vercel Serverless API (`fix.js`, `analyze.js`)
- `index.html`, `dashboard.html`, `landing.js`, `dashboard.js`, `styles.css` — frontend
- `scripts/build.js` — cross-platform static build helper
- `server/` — optional Python analyzers + tests
- `vercel.json` — global headers, CSP, build config

## Contributing
- Keep changes minimal and focused
- Add/update tests where reasonable
- Open a PR with a clear summary

---

<div align="center">
  <a href="https://codesaviour2.vercel.app/" target="_blank">
    <img src="https://img.shields.io/badge/Try%20CodeSaviour%20Now-Open%20Demo-0A84FF?style=for-the-badge" alt="Try CodeSaviour" />
  </a>
  <p>✨ Clean code, fast fixes, fewer bugs.</p>
</div>