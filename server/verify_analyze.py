import os, sys
from fastapi.testclient import TestClient

# Ensure project root on sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from server.app import app

client = TestClient(app)

# Show status to confirm OpenRouter config exposure
status = client.get("/api/status").json()
print("/api/status:")
print(status)

payload = {
    "language": "python",
    "code": "def f():\n  h=1\n  return unknown_var\n\nquery=2\n",
}

resp = client.post("/api/analyze", json=payload)
print("/api/analyze status:", resp.status_code)
print(resp.json())