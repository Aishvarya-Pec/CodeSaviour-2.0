import os
import asyncio
import hashlib
from typing import Optional, List, Tuple

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from openai import OpenAI
import httpx

def _load_env_files():
    """Load .env files into os.environ without overriding existing variables.
    Looks for .env in project root and server folder.
    Supports simple KEY=VALUE lines; ignores comments and blank lines.
    """
    candidates = []
    try:
        # Prefer explicit env var if provided
        env_path = os.environ.get("ENV_PATH")
        if env_path:
            candidates.append(env_path)
    except Exception:
        pass
    try:
        here = os.path.dirname(__file__)
        root = os.path.abspath(os.path.join(here, os.pardir))
        candidates.extend([
            os.path.join(root, ".env"),
            os.path.join(here, ".env"),
        ])
    except Exception:
        pass
    for p in candidates:
        try:
            if not os.path.isfile(p):
                continue
            with open(p, "r", encoding="utf-8") as f:
                for raw in f:
                    line = raw.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip('"').strip("'")
                    if k and (k not in os.environ):
                        os.environ[k] = v
        except Exception:
            continue

# Load .env before reading configuration
_load_env_files()

FIREWORKS_BASE_URL = os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1")
FIREWORKS_API_KEY = os.environ.get("FIREWORKS_API_KEY", "")
FIREWORKS_MODEL = os.environ.get("FIREWORKS_MODEL", "accounts/fireworks/models/qwen2p5-coder-32b-instruct")
# Ordered fallback list of models; can be overridden via FIREWORKS_MODELS env (comma-separated)
DEFAULT_MODELS = (
    "accounts/fireworks/models/qwen2p5-coder-32b-instruct,"
    "accounts/fireworks/models/llama-v3p1-70b-instruct,"
    "accounts/fireworks/models/llama-v3p1-8b-instruct"
)
FIREWORKS_MODELS = [
    m.strip()
    for m in os.environ.get("FIREWORKS_MODELS", "".join(DEFAULT_MODELS)).split(",")
    if m.strip()
]
# OpenRouter configuration for analysis
OPENROUTER_BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "qwen/qwen-2.5-coder-32b-instruct:free")
SITE_URL = os.environ.get("SITE_URL", "http://127.0.0.1:8000/")
SITE_TITLE = os.environ.get("SITE_TITLE", "CodeSaviour")

app = FastAPI(title="CodeSaviour Fix API (Fireworks AI)", version="1.0")
_origins_env = os.environ.get("CORS_ALLOW_ORIGINS", "").strip()
if _origins_env:
    _allow_origins = [o.strip() for o in _origins_env.split(",") if o.strip()]
else:
    _allow_origins = [
        "http://127.0.0.1:8000",
        "http://localhost:8000",
        "http://127.0.0.1:8010",
        "http://localhost:8010",
        "http://127.0.0.1:8012",
        "http://localhost:8012",
        "http://127.0.0.1:8015",
        "http://localhost:8015",
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = None
CACHE = {}
ANALYZER_CACHE = {}


def cstyle_basic_fix(code: str) -> str:
    """Very simple fallback fixer for C-style languages when models return unchanged.
    - Adds semicolons to probable statements missing them.
    - Trims trailing whitespace and normalizes tabs to 4 spaces.
    - Ensures final closing braces if obvious imbalance.
    This is conservative and only applied as a last resort.
    """
    lines = code.splitlines()
    out = []
    for l in lines:
        t = l.rstrip()
        # Skip blank or block lines
        if t == "" or t.endswith("{") or t.endswith(":") or t.endswith(")") and t.strip() in [")", "]", "}"]:
            out.append(t)
            continue
        # If looks like a statement but lacks semicolon, add it
        looks_stmt = False
        # common patterns: assignment, return, call, var decl
        if any(sym in t for sym in ["=", "+", "-", "*", "/"]) or t.strip().startswith("return"):
            looks_stmt = True
        if ("(" in t and ")" in t) and not t.strip().endswith("{"):
            looks_stmt = True
        if looks_stmt and not t.strip().endswith(";") and not t.strip().endswith("}"):
            t = t + ";"
        out.append(t.replace("\t", "    "))
    fixed = "\n".join(out)
    # Balance braces: add missing closing braces at end if imbalance detected
    open_br = fixed.count("{")
    close_br = fixed.count("}")
    if open_br > close_br:
        fixed = fixed + "\n" + ("}" * (open_br - close_br))
    return fixed


def approx_tokens(s: str) -> int:
    return sum(len(line.split()) for line in s.splitlines())


def split_code(code: str, chunk_size: int = 1024) -> List[str]:
    lines = code.splitlines()
    chunks: List[str] = []
    buf: List[str] = []
    tokens = 0
    def is_boundary(l: str) -> bool:
        t = l.strip()
        return (
            t.startswith("def ") or t.startswith("class ") or t.startswith("function ")
            or t.startswith("class ") or t.startswith("public ") or t.startswith("private ")
            or t.startswith("protected ") or t.startswith("static ")
        )
    for l in lines:
        tokens += len(l.split())
        buf.append(l)
        if tokens >= chunk_size and (is_boundary(l) or len(buf) > 60):
            chunks.append("\n".join(buf))
            buf = []
            tokens = 0
    if buf:
        chunks.append("\n".join(buf))
    return chunks


def build_prompt(language: str, code: str, context: Optional[str] = None) -> str:
    instructions = (
        f"You are CodeSaviour, a code-fixing assistant. The language is {language}. "
        "Fix the provided code: correct syntax errors, obvious logic bugs, and formatting. Preserve original intent. "
        "Return ONLY the corrected code with no commentary, no surrounding quotes, and no extra lines."
    )
    extra = f"\nAdditional context: {context}\n" if context else "\n"
    return f"{instructions}{extra}\n=== BUGGED CODE START ===\n{code}\n=== BUGGED CODE END ===\n"


def build_fix_fewshot(language: str) -> str:
    # Minimal few-shot to improve accuracy
    return (
        "Examples (buggy -> fixed):\n"
        "# buggy\n"
        "def add(x, y):\n    return x - y\n"
        "# fixed\n"
        "def add(x, y):\n    return x + y\n\n"
        "// buggy\nfunction greet(name){\n  console.log('Hi'+name)\n}\n"
        "// fixed\nfunction greet(name) {\n  console.log('Hi ' + name);\n}\n"
    )


def make_chunk_prompt(language: str, code_chunk: str) -> str:
    base = (
        f"You are a code fixing AI for {language}. Detect all bugs in this snippet and provide only the corrected code. "
        "Do not explain. Do not wrap the code."
    )
    examples = build_fix_fewshot(language)
    return f"{base}\n\n{examples}\n\n=== SNIPPET START ===\n{code_chunk}\n=== SNIPPET END ===\n"


async def fix_chunk_async(client: httpx.AsyncClient, language: str, code_chunk: str, model: str, headers: dict, retries: int = 1) -> str:
    cache_key = hashlib.sha256((language + "\n" + code_chunk).encode("utf-8")).hexdigest()
    if cache_key in CACHE:
        return CACHE[cache_key]
    prompt = make_chunk_prompt(language, code_chunk)
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are CodeSaviour. Return only corrected code."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "max_tokens": 2048,
    }
    last_err = None
    for attempt in range(retries + 1):
        try:
            r = await client.post(f"{FIREWORKS_BASE_URL}/chat/completions", json=payload, headers=headers, timeout=20.0)
            r.raise_for_status()
            data = r.json()
            content = (data.get("choices", [{}])[0].get("message", {}).get("content", "") or "").strip()
            # Only accept and cache if model returned a non-empty change
            if content and content.strip() != code_chunk.strip():
                CACHE[cache_key] = content
                return content
        except httpx.HTTPStatusError as e:
            # Authorization / quota issues: abort quickly and bubble up to caller
            if getattr(e, "response", None) is not None and e.response.status_code in (401, 403, 429):
                raise
            last_err = e
            await asyncio.sleep(0.3 * (attempt + 1))
        except Exception as e:
            last_err = e
            await asyncio.sleep(0.3 * (attempt + 1))
    # Fallback to returning original chunk if fixing failed
    return code_chunk


async def fix_chunk_with_fallback_async(client: httpx.AsyncClient, language: str, code_chunk: str, models: List[str], headers: dict, retries_per_model: int = 1, abort_on_auth_error: bool = True) -> str:
    """Try multiple models in order until one returns a changed (non-identical) chunk.
    Abort immediately on authorization/quota errors to avoid long timeouts.
    """
    cache_key = hashlib.sha256((language + "\n" + code_chunk).encode("utf-8")).hexdigest()
    if cache_key in CACHE:
        cached = CACHE[cache_key]
        if cached and cached.strip() != code_chunk.strip():
            return cached
        # If cached equals original, ignore and try again with fallback models
    prompt = make_chunk_prompt(language, code_chunk)
    last_err: Optional[Exception] = None
    for model in models:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are CodeSaviour. Return only corrected code."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
            "max_tokens": 2048,
        }
        for attempt in range(retries_per_model + 1):
            try:
                r = await client.post(f"{FIREWORKS_BASE_URL}/chat/completions", json=payload, headers=headers, timeout=20.0)
                r.raise_for_status()
                data = r.json()
                content = (data.get("choices", [{}])[0].get("message", {}).get("content", "") or "").strip()
                if content and content.strip() != code_chunk.strip():
                    CACHE[cache_key] = content
                    return content
            except httpx.HTTPStatusError as e:
                # If authentication/quota errors occur, abort quickly so caller can surface a helpful error
                if abort_on_auth_error and getattr(e, "response", None) is not None and e.response.status_code in (401, 403, 429):
                    raise
                last_err = e
                await asyncio.sleep(0.35 * (attempt + 1))
            except Exception as e:
                last_err = e
                await asyncio.sleep(0.35 * (attempt + 1))
        # try next model
    # If none succeeded, return original chunk to avoid server error
    return code_chunk


def build_analysis_prompt(language: str, code: str) -> str:
    instructions = (
        f"You are a precise static analyzer for {language} code. "
        "Return ONLY a JSON object with the fields: "
        "'errors' (integer), 'warnings' (integer), "
        "'error_items' (array of objects with keys 'line' (number|null) and 'message' (string)), "
        "'warning_items' (array of objects with keys 'line' (number|null) and 'message' (string)). "
        "Errors: syntax errors, undefined names, type/logic bugs, security-critical issues. "
        "Warnings: risky patterns, code smells, style problems, non-critical concerns. "
        "If a line can't be determined, use null. Do not include explanations or any text outside the JSON."
    )
    return f"{instructions}\n\n=== CODE START ===\n{code}\n=== CODE END ===\n"


class FixRequest(BaseModel):
    language: str
    code: str
    context: Optional[str] = None


class AnalyzeRequest(BaseModel):
    language: str
    code: str


class DiffFixRequest(BaseModel):
    language: str
    path: Optional[str] = None  # relative path for diff headers
    code: str
    verify: bool = True


class DiffBatchItem(BaseModel):
    language: str
    path: Optional[str] = None
    code: str


class DiffBatchRequest(BaseModel):
    files: List[DiffBatchItem]


@app.on_event("startup")
def startup_event():
    global client
    client = OpenAI(base_url=FIREWORKS_BASE_URL, api_key=FIREWORKS_API_KEY)


@app.get("/api/status")
def api_status():
    key_present = bool(os.environ.get("FIREWORKS_API_KEY"))
    return {
        "fireworks_key_present": key_present,
        "base_url": FIREWORKS_BASE_URL,
        "model": FIREWORKS_MODEL,
        "models": FIREWORKS_MODELS,
        "site_url": SITE_URL,
        "openrouter_key_present": bool(OPENROUTER_API_KEY),
        "openrouter_base_url": OPENROUTER_BASE_URL,
        "openrouter_model": OPENROUTER_MODEL,
    }


@app.post("/api/fix")
async def fix_code(req: FixRequest):
    # Split into chunks and fix in parallel for low latency
    code = req.code or ""
    language = req.language or ""
    chunks = split_code(code, chunk_size=1024)
    # Prefer single-shot for moderate inputs to preserve context and avoid truncation
    if approx_tokens(code) <= 1200:
        chunks = [code]
    # Safety: ensure we always have at least one chunk
    if not chunks:
        chunks = [code]
    headers = {
        "Authorization": f"Bearer {FIREWORKS_API_KEY}",
        "Content-Type": "application/json",
    }
    # If API key is missing, surface a clear error instead of timing out
    if not FIREWORKS_API_KEY:
        return JSONResponse(status_code=401, content={
            "fixed": code,
            "error": {
                "type": "missing_api_key",
                "detail": "FIREWORKS_API_KEY is not set",
                "hint": "Add FIREWORKS_API_KEY to .env or environment and restart the API server.",
            }
        })
    async with httpx.AsyncClient() as ac:
        try:
            tasks = [fix_chunk_with_fallback_async(ac, language, ch, FIREWORKS_MODELS or [FIREWORKS_MODEL], headers, retries_per_model=1) for ch in chunks]
            results: List[str] = await asyncio.gather(*tasks)
        except httpx.HTTPStatusError as e:
            status = e.response.status_code if getattr(e, "response", None) is not None else 502
            detail = ""
            try:
                detail = e.response.text[:200] if e.response is not None else ""
            except Exception:
                detail = str(e)
            # If unauthorized, try a single-call completion via OpenAI client before returning
            if status in (401, 403):
                try:
                    prompt = build_prompt(language, code, req.context)
                    for model in (FIREWORKS_MODELS or [FIREWORKS_MODEL]):
                        try:
                            completion = client.chat.completions.create(
                                model=model,
                                messages=[{"role": "user", "content": prompt}],
                                temperature=0,
                                max_tokens=2048,
                            )
                            candidate = (completion.choices[0].message.content if completion and completion.choices else "").strip()
                            if candidate and candidate.strip() != (code or "").strip():
                                return {"fixed": candidate}
                        except Exception:
                            continue
                except Exception:
                    pass
            return JSONResponse(status_code=(status if status in (401, 403, 429) else 502), content={
                "fixed": code,
                "error": {
                    "type": "fireworks_http_error",
                    "status": status,
                    "detail": detail,
                    "hint": "Check FIREWORKS_API_KEY and FIREWORKS_BASE_URL.",
                }
            })
        except Exception as e:
            return JSONResponse(status_code=502, content={
                "fixed": code,
                "error": {
                    "type": "fireworks_request_failed",
                    "detail": str(e)[:200],
                    "hint": "Network error or invalid base URL.",
                }
            })

    # If nothing changed after the parallel pass, escalate retries across all models
    fixed = "\n\n".join(results).strip()
    unchanged_all = all((r or "").strip() == (chunks[i] or "").strip() for i, r in enumerate(results))
    if unchanged_all:
        async with httpx.AsyncClient() as ac:
            tasks = [fix_chunk_with_fallback_async(ac, language, ch, FIREWORKS_MODELS or [FIREWORKS_MODEL], headers, retries_per_model=2) for ch in chunks]
            escalated: List[str] = await asyncio.gather(*tasks)
        if any((escalated[i] or "").strip() != (chunks[i] or "").strip() for i in range(len(chunks))):
            fixed = "\n\n".join(escalated).strip()

    if not fixed or fixed.strip() == (code or "").strip():
        # Fallback single-call if something went wrong
        prompt = build_prompt(language, code, req.context)
        for model in (FIREWORKS_MODELS or [FIREWORKS_MODEL]):
            try:
                completion = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=2048,
                )
                candidate = (completion.choices[0].message.content if completion and completion.choices else "").strip()
                if candidate and candidate.strip() != code.strip():
                    fixed = candidate
                    break
            except Exception:
                continue
        if not fixed or fixed.strip() == (code or "").strip():
            # Last-resort fallback: apply conservative C-style fixer for supported languages
            lang_lower = (language or "").lower()
            cstyle_langs = {"javascript","typescript","java","csharp","cpp","c"}
            if lang_lower in cstyle_langs:
                heuristic = cstyle_basic_fix(code)
                fixed = heuristic if heuristic.strip() else code
            else:
                fixed = code
    fixed_fmt = apply_formatter(language, fixed) or fixed
    return {"fixed": fixed_fmt}


@app.post("/api/analyze")
def analyze_code(req: AnalyzeRequest):
    import json
    lang = (req.language or "").lower()

    def norm(items):
        out = []
        for it in (items or []):
            try:
                line = it.get("line")
                if line is not None:
                    line = int(line)
                msg = str(it.get("message", ""))
                out.append({"line": line, "message": msg})
            except Exception:
                pass
        return out

    # Call OpenRouter for structured analysis
    errors = 0
    warnings = 0
    error_items = []
    warning_items = []
    llm_ok = False
    try:
        prompt = build_analysis_prompt(req.language, req.code)
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": SITE_URL,
            "X-Title": SITE_TITLE,
        }
        payload = {
            "model": OPENROUTER_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"},
            "temperature": 0,
        }
        resp = httpx.post(f"{OPENROUTER_BASE_URL}/chat/completions", json=payload, headers=headers, timeout=20.0)
        resp.raise_for_status()
        data = resp.json()
        content = (data.get("choices", [{}])[0].get("message", {}).get("content", "") or "").strip()
        parsed = json.loads(content or "{}")
        errors = int(parsed.get("errors", 0))
        warnings = int(parsed.get("warnings", 0))
        error_items = parsed.get("error_items", []) or []
        warning_items = parsed.get("warning_items", []) or []
        llm_ok = True
    except Exception:
        llm_ok = False

    # Python: augment with local analyzers and heuristics for precision
    if lang == "python":
        import tempfile
        import os as _os
        from subprocess import run, PIPE

        # pyflakes output
        with tempfile.NamedTemporaryFile(delete=False, suffix=".py", mode="w", encoding="utf-8") as tf:
            tf.write(req.code)
            path = tf.name
        try:
            proc = run(["pyflakes", path], stdout=PIPE, stderr=PIPE, text=True)
            output = (proc.stdout or "") + (proc.stderr or "")
        finally:
            try:
                _os.unlink(path)
            except Exception:
                pass

        # quick syntax check
        try:
            compile(req.code, "x.py", "exec")
        except SyntaxError as se:
            error_items.append({"line": getattr(se, "lineno", None), "message": str(se)})

        pf_error_items = []
        pf_warning_items = []
        for l in [l for l in output.splitlines() if l.strip()]:
            parts = l.split(":", 3)
            line_no = None
            msg_text = l
            if len(parts) >= 4:
                try:
                    line_no = int(parts[1])
                except Exception:
                    line_no = None
                msg_text = parts[3].strip()
            msg_lower = msg_text.lower()
            is_error = (
                "syntaxerror" in msg_lower
                or "invalid syntax" in msg_lower
                or "undefined name" in msg_lower
                or "used before assignment" in msg_lower
                or "not defined" in msg_lower
                or "could not compile" in msg_lower
            )
            if is_error:
                pf_error_items.append({"line": line_no, "message": msg_text})
            else:
                pf_warning_items.append({"line": line_no, "message": msg_text})

        # broader analyzers
        try:
            diags = run_python_analyzers(req.code)
        except Exception:
            diags = []
        diag_error_items = []
        diag_warning_items = []
        for d in diags:
            line_no = int(d.get("line", 0)) or None
            msg_text = str(d.get("message", ""))
            sev = str(d.get("severity", "warning")).lower()
            msg_lower = msg_text.lower()
            is_error = (
                sev == "error"
                or "undefined name" in msg_lower
                or "used before assignment" in msg_lower
                or "division by zero" in msg_lower
                or "cannot import" in msg_lower
                or "nameerror" in msg_lower
                or "typeerror" in msg_lower
                or "syntaxerror" in msg_lower
                or "invalid syntax" in msg_lower
                or (d.get("rule", "").lower().startswith("b"))
            )
            if is_error:
                diag_error_items.append({"line": line_no, "message": msg_text})
            else:
                diag_warning_items.append({"line": line_no, "message": msg_text})

        # lightweight heuristics
        try:
            import re as _re
            lines = (req.code or "").splitlines()
            heuristic_rules = [
                {"rx": r"\beval\s*\(", "msg": "Use of eval()", "sev": "error"},
                {"rx": r"\bexec\s*\(", "msg": "Use of exec()", "sev": "error"},
                {"rx": r"\bos\.system\s*\(", "msg": "Use of os.system()", "sev": "error"},
                {"rx": r"SELECT\b.*\+", "msg": "Possible SQL injection via string concatenation", "sev": "error"},
                {"rx": r"/\s*0(?!\.)", "msg": "Literal division by zero", "sev": "error"},
                {"rx": r"\bprint\s*\(.*\+.*\)", "msg": "String concatenation in print may coerce types", "sev": "warning"},
            ]
            for i, l in enumerate(lines, start=1):
                for r in heuristic_rules:
                    if _re.search(r["rx"], l):
                        item = {"line": i, "message": r["msg"]}
                        key = (item["line"], item["message"]) 
                        if r["sev"] == "error":
                            if key not in {(it.get("line"), it.get("message")) for it in error_items}:
                                error_items.append(item)
                        else:
                            if key not in {(it.get("line"), it.get("message")) for it in warning_items}:
                                warning_items.append(item)
        except Exception:
            pass

        # merge de-duplicated
        seen = set()
        merged_error_items = []
        for it in (error_items or []) + pf_error_items + diag_error_items:
            key = (it.get("line"), it.get("message"))
            if key not in seen:
                seen.add(key)
                merged_error_items.append(it)
        seen = set()
        merged_warning_items = []
        for it in (warning_items or []) + pf_warning_items + diag_warning_items:
            key = (it.get("line"), it.get("message"))
            if key not in seen:
                seen.add(key)
                merged_warning_items.append(it)
        return {
            "errors": len(merged_error_items),
            "warnings": len(merged_warning_items),
            "error_items": norm(merged_error_items),
            "warning_items": norm(merged_warning_items),
        }

    # Merge LLM results with local analyzers and cross-language heuristics for all languages
    base_err = norm(error_items)
    base_warn = norm(warning_items)
    diags = []
    try:
        diags = gather_diagnostics(lang, req.code or "")
    except Exception:
        diags = []
    diag_err = []
    diag_warn = []
    for d in diags:
        ln = int(d.get("line", 0)) or None
        msg = str(d.get("message", ""))
        sev = str(d.get("severity", "warning")).lower()
        if sev == "error":
            diag_err.append({"line": ln, "message": msg})
        else:
            diag_warn.append({"line": ln, "message": msg})
    # Cross-language security, env, and concurrency heuristics
    heur_err = []
    heur_warn = []
    try:
        heur = cross_language_heuristics(lang, req.code or "")
        for h in (heur or []):
            ln = int(h.get("line", 0)) or None
            msg = str(h.get("message", ""))
            sev = str(h.get("severity", "warning")).lower()
            if sev == "error":
                heur_err.append({"line": ln, "message": msg})
            else:
                heur_warn.append({"line": ln, "message": msg})
    except Exception:
        pass
    # If we have anything (from LLM, analyzers, or heuristics), return merged unique set
    merged_err = []
    merged_warn = []
    seen_e = set()
    seen_w = set()
    for it in (base_err or []) + diag_err + heur_err:
        key = (it.get("line"), it.get("message"))
        if key not in seen_e:
            seen_e.add(key)
            merged_err.append(it)
    for it in (base_warn or []) + diag_warn + heur_warn:
        key = (it.get("line"), it.get("message"))
        if key not in seen_w:
            seen_w.add(key)
            merged_warn.append(it)
    if llm_ok or merged_err or merged_warn:
        return {
            "errors": len(merged_err),
            "warnings": len(merged_warn),
            "error_items": merged_err,
            "warning_items": merged_warn,
        }

    # Fallback generic heuristics if nothing else produced results
    code = req.code or ""
    lines = code.splitlines()
    errors = 0
    warnings = 0
    error_items = []
    warning_items = []
    for open_ch, close_ch in [("(", ")"), ("{", "}"), ("[", "]")]:
        bal = 0
        for ch in code:
            if ch == open_ch:
                bal += 1
            elif ch == close_ch:
                bal = max(0, bal - 1)
        if bal > 0:
            errors += bal
            error_items.append({"line": None, "message": f"Unbalanced {open_ch}{close_ch}: missing {close_ch}"})
    import re as _re2
    for i, l in enumerate(lines, start=1):
        if _re2.search(r"\berror\b", l, flags=_re2.IGNORECASE):
            errors += 1
            error_items.append({"line": i, "message": "Contains keyword 'error'"})
        if _re2.search(r"\bwarn(?:ing)?\b", l, flags=_re2.IGNORECASE):
            warnings += 1
            warning_items.append({"line": i, "message": "Contains warning keyword"})
    for i, l in enumerate(lines, start=1):
        if len(l) > 120:
            warnings += 1
            warning_items.append({"line": i, "message": "Line exceeds 120 characters"})
        if _re2.search(r"\s+$", l):
            warnings += 1
            warning_items.append({"line": i, "message": "Trailing whitespace"})
    return {
        "errors": errors,
        "warnings": warnings,
        "error_items": error_items,
        "warning_items": warning_items,
    }


def _which(cmd: str) -> Optional[str]:
    """Resolve a command path with optional environment variable overrides.
    If an env var override is provided, prefer it; otherwise fall back to PATH.
    Supported overrides:
      FLAKE8_PATH, PYFLAKES_PATH, BANDIT_PATH, MYPY_PATH,
      ESLINT_PATH, CPP_CHECK_PATH, CLANG_TIDY_PATH, PMD_PATH
    """
    try:
        import shutil
        env_map = {
            "flake8": "FLAKE8_PATH",
            "pyflakes": "PYFLAKES_PATH",
            "bandit": "BANDIT_PATH",
            "mypy": "MYPY_PATH",
            "eslint": "ESLINT_PATH",
            "cppcheck": "CPP_CHECK_PATH",
            "clang-tidy": "CLANG_TIDY_PATH",
            "pmd": "PMD_PATH",
        }
        override = os.environ.get(env_map.get(cmd, ""), "").strip()
        if override:
            # If a direct path is provided, use it; else resolve via PATH
            if os.path.isfile(override):
                return override
            resolved = shutil.which(override)
            if resolved:
                return resolved
        return shutil.which(cmd)
    except Exception:
        return None


def _sanitize_json(obj):
    # Remove dangerous prototype pollution keys recursively
    dangerous = "__proto__", "constructor", "prototype"
    if isinstance(obj, dict):
        return {k: _sanitize_json(v) for k, v in obj.items() if k not in dangerous}
    if isinstance(obj, list):
        return [_sanitize_json(x) for x in obj]
    return obj


def _truncate_lines(s: str, max_lines: int = 200) -> str:
    lines = s.splitlines()
    if len(lines) <= max_lines:
        return s
    return "\n".join(lines[:max_lines])


def _redact_secrets(text: str) -> str:
    import re
    # Basic redaction of obvious key patterns
    text = re.sub(r"sk-[a-zA-Z0-9_-]{20,}", "sk-REDACTED", text)
    text = re.sub(r"(?i)aws_[a-z_]*\s*=\s*['\"][^'\"]+['\"]", "AWS_KEY=REDACTED", text)
    text = re.sub(r"(?i)password\s*[:=]\s*[^\n]+", "password=REDACTED", text)
    # Strip absolute Windows paths
    text = re.sub(r"[A-Za-z]:\\[^\s]+", "<REDACTED_PATH>", text)
    return text


def cross_language_heuristics(lang: str, code: str) -> List[dict]:
    """Lightweight multi-language heuristics for security, env, and concurrency.
    Returns list of {line, message, severity}.
    """
    import re
    lines = (code or "").splitlines()
    L = (lang or "").lower()
    items: List[dict] = []

    def add(line, msg, sev="warning"):
        items.append({"line": line, "message": msg, "severity": sev})

    # Secrets and tokens
    secret_patterns = [
        r"AKIA[0-9A-Z]{16}",
        r"(?i)(api[_-]?key|secret|token)\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{16,}['\"]?",
        r"-----BEGIN (?:RSA|EC|DSA) PRIVATE KEY-----",
    ]
    for i, l in enumerate(lines, 1):
        for rx in secret_patterns:
            if re.search(rx, l):
                add(i, "Potential secret detected in code", "error")
                break

    # Command execution / shell
    if L == "python":
        for i, l in enumerate(lines, 1):
            if re.search(r"\bos\.system\s*\(", l):
                add(i, "Use of os.system()", "error")
            if re.search(r"subprocess\.(Popen|call|run)\s*\(.*shell\s*=\s*True", l):
                add(i, "subprocess with shell=True is unsafe", "error")
    if L in {"javascript", "typescript"}:
        for i, l in enumerate(lines, 1):
            if re.search(r"child_process\.(exec|execSync|spawn)\s*\(", l):
                add(i, "child_process execution can be unsafe", "error")
            if re.search(r"\beval\s*\(", l) or re.search(r"\bnew\s+Function\s*\(", l):
                add(i, "Use of eval/Function is unsafe", "error")
            if re.search(r"\.innerHTML\s*=", l):
                add(i, "Setting innerHTML can lead to XSS", "error")
            # Simple async misuse: fetch without await inside async function (approximate)
            if "fetch(" in l and "await" not in l:
                add(i, "Possible missing await on fetch()", "warning")
    if L == "java":
        for i, l in enumerate(lines, 1):
            if re.search(r"Runtime\.getRuntime\(\)\.exec\(", l) or re.search(r"new\s+ProcessBuilder\(", l):
                add(i, "External process execution: validate and sanitize inputs", "error")
            if re.search(r"Connection\s+.+=\s*DriverManager\.getConnection\(", l):
                add(i, "Verify DB connection lifecycle and close in finally", "warning")
    if L == "csharp":
        for i, l in enumerate(lines, 1):
            if re.search(r"new\s+Process(StartInfo)?\(", l) or re.search(r"ProcessStartInfo", l):
                add(i, "ProcessStart input sanitization required", "error")
            if re.search(r"new\s+SqlConnection\(", l):
                add(i, "Ensure SqlConnection disposed (using/await using)", "warning")
    if L in {"c", "cpp"}:
        for i, l in enumerate(lines, 1):
            if re.search(r"\bsystem\s*\(", l):
                add(i, "Use of system() is unsafe", "error")
            if re.search(r"\bstrcpy\s*\(|\bstrcat\s*\(|\bgets\s*\(", l):
                add(i, "Potential buffer overflow-prone function", "error")
        # crude thread join check
        if "pthread_create(" in code and "pthread_join(" not in code:
            add(None, "Thread created without join detected (pthread)", "warning")

    # SQL concatenation across languages (very rough)
    for i, l in enumerate(lines, 1):
        if re.search(r"\b(SELECT|INSERT|UPDATE|DELETE)\b.*(\+|\%\w|\{.+\})", l, re.IGNORECASE):
            add(i, "Possible SQL injection via dynamic query construction", "error")

    # Environment and logging of secrets
    for i, l in enumerate(lines, 1):
        if re.search(r"process\.env\.[A-Z0-9_]+", l) and re.search(r"console\.(log|debug)", l):
            add(i, "Leaking environment variable to logs", "warning")
        if re.search(r"os\.getenv\(\s*['\"][A-Za-z0-9_]+['\"]\s*\)", l) and re.search(r"print\(|logging\.", l):
            add(i, "Printing environment variable to output", "warning")
        if re.search(r"[A-Za-z]:\\\\", l) or "/etc/" in l:
            add(i, "Hardcoded absolute path detected", "warning")

    return items


def apply_formatter(language: str, code: str) -> Optional[str]:
    """Optionally format the code using local formatters if available."""
    import tempfile
    from subprocess import run, PIPE
    lang = (language or "").lower()
    suffix_map = {
        "python": ".py",
        "javascript": ".js",
        "typescript": ".ts",
        "c": ".c",
        "cpp": ".cpp",
        "java": ".java",
        "csharp": ".cs",
    }
    suffix = suffix_map.get(lang, ".txt")
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, mode="w", encoding="utf-8") as tf:
        tf.write(code or "")
        path = tf.name
    try:
        if lang == "python":
            black = _which("black")
            if black:
                run([black, path, "--quiet"], stdout=PIPE, stderr=PIPE, text=True)
        elif lang in {"javascript", "typescript"}:
            prettier = _which("prettier")
            if prettier:
                run([prettier, "--write", path], stdout=PIPE, stderr=PIPE, text=True)
        elif lang in {"c", "cpp"}:
            clangfmt = _which("clang-format")
            if clangfmt:
                out = run([clangfmt, path], stdout=PIPE, stderr=PIPE, text=True)
                if out.stdout:
                    return out.stdout
        # Read back file for formatters that rewrite in place
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            return None
    finally:
        try:
            os.unlink(path)
        except Exception:
            pass


def run_python_analyzers(code: str) -> List[dict]:
    """Run local Python analyzers, preferring JSON outputs when available.
    Returns a list of diagnostics dicts: {file,line,rule,severity,message}.
    """
    import tempfile
    import os as _os
    from subprocess import run, PIPE

    diagnostics: List[dict] = []

    with tempfile.NamedTemporaryFile(delete=False, suffix=".py", mode="w", encoding="utf-8") as tf:
        tf.write(code)
        path = tf.name
    try:
        # flake8 JSON (if available)
        flake8 = _which("flake8")
        if flake8:
            # Some flake8 distros use --format=json
            proc = run([flake8, "--format=json", path], stdout=PIPE, stderr=PIPE, text=True)
            out = proc.stdout.strip()
            if out:
                try:
                    import json as _json
                    data = _json.loads(out)
                    for file_path, items in (data.items() if isinstance(data, dict) else []):
                        for it in items:
                            diagnostics.append({
                                "file": os.path.relpath(file_path),
                                "line": int(it.get("line_number") or it.get("line", 0)),
                                "rule": str(it.get("code", "flake8")),
                                "severity": "warning",
                                "message": str(it.get("text", "")),
                            })
                except Exception:
                    pass
        # pyflakes textual (fallback)
        pyflakes = _which("pyflakes")
        if pyflakes:
            proc = run([pyflakes, path], stdout=PIPE, stderr=PIPE, text=True)
            output = (proc.stdout or "") + (proc.stderr or "")
            for l in output.splitlines():
                parts = l.split(":", 3)
                line_no = None
                msg_text = l.strip()
                if len(parts) >= 4:
                    try:
                        line_no = int(parts[1])
                    except Exception:
                        line_no = None
                    msg_text = parts[3].strip()
                diagnostics.append({
                    "file": os.path.basename(path),
                    "line": line_no or 0,
                    "rule": "pyflakes",
                    "severity": "warning",
                    "message": msg_text,
                })
        # bandit JSON (security)
        bandit = _which("bandit")
        if bandit:
            proc = run([bandit, "-f", "json", "-q", "-r", path], stdout=PIPE, stderr=PIPE, text=True)
            out = proc.stdout.strip()
            if out:
                try:
                    import json as _json
                    data = _json.loads(out)
                    for it in (data.get("results") or []):
                        diagnostics.append({
                            "file": os.path.basename(path),
                            "line": int(it.get("line_number", 0)),
                            "rule": str(it.get("test_id", "bandit")),
                            "severity": str(it.get("issue_severity", "warning")).lower(),
                            "message": str(it.get("issue_text", "")),
                        })
                except Exception:
                    pass
        # mypy JSON (types)
        mypy = _which("mypy")
        if mypy:
            proc = run([mypy, "--error-format=json", path], stdout=PIPE, stderr=PIPE, text=True)
            out = proc.stdout.strip()
            if out:
                try:
                    import json as _json
                    data = _json.loads(f"[{out}]" if out and out.strip().startswith("{") and not out.strip().endswith("]") else out)
                    for it in (data if isinstance(data, list) else []):
                        diagnostics.append({
                            "file": os.path.basename(path),
                            "line": int(it.get("line", 0)),
                            "rule": str(it.get("code", "mypy")),
                            "severity": "error" if str(it.get("severity", "error")).lower() == "error" else "warning",
                            "message": str(it.get("message", "")),
                        })
                except Exception:
                    pass
    finally:
        try:
            _os.unlink(path)
        except Exception:
            pass
    return diagnostics


def run_eslint(code: str, ext: str) -> List[dict]:
    import tempfile
    import os as _os
    from subprocess import run, PIPE
    diagnostics: List[dict] = []
    eslint = _which("eslint")
    if not eslint:
        return diagnostics
    suffix = ".ts" if ext == "ts" else ".js"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, mode="w", encoding="utf-8") as tf:
        tf.write(code)
        path = tf.name
    try:
        proc = run([eslint, "-f", "json", path], stdout=PIPE, stderr=PIPE, text=True)
        out = proc.stdout.strip()
        if out:
            try:
                import json as _json
                data = _json.loads(out)
                for file_rep in (data if isinstance(data, list) else []):
                    for it in (file_rep.get("messages") or []):
                        diagnostics.append({
                            "file": os.path.basename(path),
                            "line": int(it.get("line", 0)),
                            "rule": str(it.get("ruleId", "eslint")),
                            "severity": "error" if int(it.get("severity", 1)) == 2 else "warning",
                            "message": str(it.get("message", "")),
                        })
            except Exception:
                pass
    finally:
        try:
            _os.unlink(path)
        except Exception:
            pass
    return diagnostics


def gather_diagnostics(lang: str, code: str, path: Optional[str] = None) -> List[dict]:
    key = hashlib.sha256((lang + "\n" + code + (path or "")).encode("utf-8")).hexdigest()
    if key in ANALYZER_CACHE:
        return ANALYZER_CACHE[key]
    lang_l = (lang or "").lower()
    diags: List[dict] = []
    if lang_l == "python":
        diags = run_python_analyzers(code)
    elif lang_l in {"javascript", "typescript"}:
        diags = run_eslint(code, "ts" if lang_l == "typescript" else "js")
    elif lang_l in {"cpp", "c"}:
        diags = run_cpp_analyzers(code, lang_l)
    elif lang_l == "java":
        diags = run_java_pmd(code)
    else:
        # TODO: add spotbugs/pmd, dotnet-format, clang-tidy/cppcheck integration when available
        diags = []
    ANALYZER_CACHE[key] = diags
    return diags


def build_diff_prompt(lang: str, path: Optional[str], diagnostics: List[dict], source: str) -> str:
    import json as _json
    rel_path = path or "unknown/file"
    safe_diags = _sanitize_json(diagnostics)
    # Summarize to required fields
    normalized = []
    for d in (safe_diags or []):
        try:
            normalized.append({
                "file": str(d.get("file", rel_path)),
                "line": int(d.get("line", 0)),
                "rule": str(d.get("rule", "")),
                "severity": str(d.get("severity", "warning")),
                "message": str(d.get("message", ""))[:400],
            })
        except Exception:
            pass
    diag_json = _truncate_lines(_json.dumps(normalized, ensure_ascii=False, separators=(",", ":")), 200)
    src = _redact_secrets(source)
    tmpl = (
        "You are a code fixer.\n"
        f"LANGUAGE: {lang}\n"
        f"FILE: {rel_path}\n"
        "ANALYZER_DIAGNOSTICS (JSON, top 200 lines):\n"
        f"{diag_json}\n"
        "FILE_CONTENT:\n"
        f"{src}\n\n"
        "TASK:\n"
        "1. Fix ONLY reported issues.\n"
        "2. Keep API logic unchanged.\n"
        "3. Follow language formatter rules.\n"
        "4. Output a unified diff starting with --- a/<file>, +++ b/<file>.\n"
        "5. No prose outside the diff.\n"
    )
    return tmpl


def apply_unified_diff(original: str, diff_text: str) -> Optional[str]:
    """Very basic unified diff applier for single-file diffs. Returns new content or None.
    Supports context (' '), removals ('-'), additions ('+'). Ignores hunk ranges.
    """
    lines = original.splitlines()
    new_lines: List[str] = []
    i = 0
    in_hunk = False
    for raw in diff_text.splitlines():
        if raw.startswith("--- ") or raw.startswith("+++ "):
            # headers ignored
            continue
        if raw.startswith("@@"):
            in_hunk = True
            continue
        if not in_hunk:
            # ignore any preamble
            continue
        if raw.startswith(" "):
            ctx = raw[1:]
            if i >= len(lines):
                return None
            if lines[i] != ctx:
                # context mismatch
                return None
            new_lines.append(lines[i])
            i += 1
        elif raw.startswith("-"):
            del_line = raw[1:]
            if i >= len(lines):
                return None
            if lines[i] != del_line:
                return None
            # skip the original line (deleted)
            i += 1
        elif raw.startswith("+"):
            add_line = raw[1:]
            new_lines.append(add_line)
        else:
            # unknown marker, ignore
            continue
    # append remaining original lines
    if i < len(lines):
        new_lines.extend(lines[i:])
    return "\n".join(new_lines)


@app.post("/api/diff_fix")
async def diff_fix(req: DiffFixRequest):
    """Run local analyzers, then ask Fireworks AI to return a unified diff patch for the file."""
    lang = req.language
    path = req.path or "unknown/file"
    source = req.code or ""
    diags = gather_diagnostics(lang, source, path)
    prompt = build_diff_prompt(lang, path, diags, source)
    # Precompute summary to return even if remote call fails
    before_errors = sum(1 for d in diags if str(d.get("severity", "warning")) == "error")
    before_warnings = sum(1 for d in diags if str(d.get("severity", "warning")) != "error")
    rules = {}
    for d in diags:
        rules[d.get("rule", "")] = rules.get(d.get("rule", ""), 0) + 1

    headers = {
        "Authorization": f"Bearer {FIREWORKS_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": FIREWORKS_MODEL,
        "messages": [
            {"role": "system", "content": "You output only unified diffs."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1,
        "max_tokens": 1200,
    }
    async with httpx.AsyncClient() as ac:
        try:
            r = await ac.post(f"{FIREWORKS_BASE_URL}/chat/completions", json=payload, headers=headers, timeout=30.0)
            r.raise_for_status()
            data = r.json()
        except httpx.HTTPStatusError as e:
            status = e.response.status_code if getattr(e, "response", None) is not None else 502
            detail = ""
            try:
                detail = e.response.text[:200] if e.response is not None else ""
            except Exception:
                detail = str(e)
            return JSONResponse(status_code=(status if status in (401, 403, 429) else 502), content={
                "diff": "",
                "summary": {
                    "file": path,
                    "before_errors": before_errors,
                    "before_warnings": before_warnings,
                    "rules": rules,
                },
                "verification": {"applied": False, "after_errors": None, "after_warnings": None},
                "error": {
                    "type": "fireworks_http_error",
                    "status": status,
                    "detail": detail,
                    "hint": "Set a valid FIREWORKS_API_KEY and check FIREWORKS_BASE_URL.",
                },
            })
        except Exception as e:
            return JSONResponse(status_code=502, content={
                "diff": "",
                "summary": {
                    "file": path,
                    "before_errors": before_errors,
                    "before_warnings": before_warnings,
                    "rules": rules,
                },
                "verification": {"applied": False, "after_errors": None, "after_warnings": None},
                "error": {
                    "type": "fireworks_request_failed",
                    "detail": str(e)[:200],
                    "hint": "Network error or invalid base URL.",
                },
            })
    diff = (data.get("choices", [{}])[0].get("message", {}).get("content", "") or "").strip()

    # Build summary
    before_errors = sum(1 for d in diags if str(d.get("severity", "warning")) == "error")
    before_warnings = sum(1 for d in diags if str(d.get("severity", "warning")) != "error")
    rules = {}
    for d in diags:
        rules[d.get("rule", "")] = rules.get(d.get("rule", ""), 0) + 1

    verification = {"applied": False, "after_errors": None, "after_warnings": None}
    if req.verify and diff:
        try:
            new_code = apply_unified_diff(source, diff)
            if isinstance(new_code, str) and new_code:
                after_diags = gather_diagnostics(lang, new_code)
                verification = {
                    "applied": True,
                    "after_errors": sum(1 for d in after_diags if str(d.get("severity", "warning")) == "error"),
                    "after_warnings": sum(1 for d in after_diags if str(d.get("severity", "warning")) != "error"),
                }
        except Exception:
            pass

    return {
        "diff": diff,
        "summary": {
            "file": path,
            "before_errors": before_errors,
            "before_warnings": before_warnings,
            "rules": rules,
        },
        "verification": verification,
    }


@app.post("/api/diff_batch")
async def diff_batch(req: DiffBatchRequest):
    results = []
    headers = {
        "Authorization": f"Bearer {FIREWORKS_API_KEY}",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient() as ac:
        tasks = []
        for item in req.files:
            lang = item.language
            path = item.path or "unknown/file"
            src = item.code or ""
            diags = gather_diagnostics(lang, src, path)
            prompt = build_diff_prompt(lang, path, diags, src)
            payload = {
                "model": FIREWORKS_MODEL,
                "messages": [
                    {"role": "system", "content": "You output only unified diffs."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.1,
                "max_tokens": 1200,
            }
            tasks.append(ac.post(f"{FIREWORKS_BASE_URL}/chat/completions", json=payload, headers=headers, timeout=30.0))
        responses = await asyncio.gather(*tasks)
        for idx, r in enumerate(responses):
            try:
                r.raise_for_status()
                data = r.json()
                diff = (data.get("choices", [{}])[0].get("message", {}).get("content", "") or "").strip()
            except Exception:
                diff = ""
            item = req.files[idx]
            results.append({"file": item.path or "unknown/file", "diff": diff})
    return {"results": results}


def run_cpp_analyzers(code: str, lang_l: str) -> List[dict]:
    import tempfile
    import os as _os
    from subprocess import run, PIPE
    diagnostics: List[dict] = []
    suffix = ".c" if lang_l == "c" else ".cpp"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, mode="w", encoding="utf-8") as tf:
        tf.write(code)
        path = tf.name
    try:
        # cppcheck with templated output for easy parsing
        cppcheck = _which("cppcheck")
        if cppcheck:
            tpl = "{file}:{line}:{severity}:{id}:{message}"
            proc = run([cppcheck, "--enable=all", f"--template={tpl}", path], stdout=PIPE, stderr=PIPE, text=True)
            out = (proc.stdout or "") + (proc.stderr or "")
            for l in out.splitlines():
                d = parse_cppcheck_template_line(l)
                if d:
                    diagnostics.append(d)
        # clang-tidy (text) — may produce limited results without compile database
        tidy = _which("clang-tidy")
        if tidy:
            proc = run([tidy, path, "-checks=*", "-quiet"], stdout=PIPE, stderr=PIPE, text=True)
            out = (proc.stdout or "") + (proc.stderr or "")
            for l in out.splitlines():
                d = parse_clang_tidy_line(l, default_file=path)
                if d:
                    diagnostics.append(d)
    finally:
        try:
            _os.unlink(path)
        except Exception:
            pass
    return diagnostics


def run_java_pmd(code: str) -> List[dict]:
    import tempfile
    import os as _os
    from subprocess import run, PIPE
    diagnostics: List[dict] = []
    pmd = _which("pmd")
    if not pmd:
        return diagnostics
    with tempfile.NamedTemporaryFile(delete=False, suffix=".java", mode="w", encoding="utf-8") as tf:
        tf.write(code)
        path = tf.name
    try:
        # Use error-prone ruleset by default; output JSON
        proc = run([pmd, "-d", path, "-R", "category/java/errorprone.xml", "-f", "json"], stdout=PIPE, stderr=PIPE, text=True)
        out = proc.stdout.strip()
        if out:
            try:
                import json as _json
                data = _json.loads(out)
                diagnostics.extend(parse_pmd_json(data, default_file=path))
            except Exception:
                pass
    finally:
        try:
            _os.unlink(path)
        except Exception:
            pass
    return diagnostics


def parse_cppcheck_template_line(line: str) -> Optional[dict]:
    parts = line.split(":", 4)
    if len(parts) != 5:
        return None
    f, line_no, sev, rule, msg = parts
    try:
        return {
            "file": os.path.basename(f),
            "line": int(line_no or 0),
            "rule": rule,
            "severity": str(sev).lower(),
            "message": msg.strip(),
        }
    except Exception:
        return None


def parse_clang_tidy_line(line: str, default_file: Optional[str] = None) -> Optional[dict]:
    import re
    rx = re.compile(r"^(.*?):(\d+):(\d+):\s+(warning|error):\s+(.*?)(?:\s+\[(.*?)\])?$")
    m = rx.match(line.strip())
    if not m:
        return None
    f, line_no, col, sev, msg, rule = m.groups()
    try:
        return {
            "file": os.path.basename(f or (default_file or "unknown")),
            "line": int(line_no or 0),
            "rule": rule or "clang-tidy",
            "severity": str(sev).lower(),
            "message": (msg or "").strip(),
        }
    except Exception:
        return None


def parse_pmd_json(data: dict, default_file: Optional[str] = None) -> List[dict]:
    out: List[dict] = []
    try:
        for file_rep in (data.get("files") or []):
            fname = os.path.basename(file_rep.get("filename", default_file or ""))
            for it in (file_rep.get("violations") or []):
                try:
                    out.append({
                        "file": fname,
                        "line": int(it.get("beginline", 0)),
                        "rule": str(it.get("rule", "pmd")),
                        "severity": str(it.get("priority", "warning")),
                        "message": str(it.get("description", "")),
                    })
                except Exception:
                    continue
    except Exception:
        pass
    return out