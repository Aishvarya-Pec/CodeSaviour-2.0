import httpx

BASE = "http://127.0.0.1:8001"

def main():
    try:
        r = httpx.get(f"{BASE}/docs", timeout=5)
        print(f"GET /docs: {r.status_code}")
    except Exception as e:
        print(f"GET /docs error: {e}")

    try:
        payload = {"language": "python", "code": "print(1)"}
        r = httpx.post(f"{BASE}/api/analyze", json=payload, timeout=8)
        print(f"POST /api/analyze: {r.status_code} {r.text[:120]}")
    except Exception as e:
        print(f"POST /api/analyze error: {e}")

    try:
        buggy = "def add(x y):\n    return x + y"  # missing comma
        payload = {"language": "python", "code": buggy}
        r = httpx.post(f"{BASE}/api/fix", json=payload, timeout=12)
        txt = r.json().get("fixed", "") if r.status_code == 200 else r.text
        print(f"POST /api/fix: {r.status_code} {str(txt)[:120]}")
    except Exception as e:
        print(f"POST /api/fix error: {e}")

if __name__ == "__main__":
    main()