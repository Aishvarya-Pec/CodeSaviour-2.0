import os
import sys
import uvicorn

# Ensure project root is on sys.path so `server` package/module resolves
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

# Do not set secrets here; rely on .env or environment variables
if not os.environ.get("OPENROUTER_API_KEY"):
    print("[WARN] OPENROUTER_API_KEY is not set. Ensure .env or environment has the key.")

from server.app import app  # import after path adjustment

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001, log_level="info")