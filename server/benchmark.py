import asyncio
import time
import json
import random
import os
from typing import List, Tuple, Dict

import httpx

# If network to localhost is blocked or unstable, run in-process using ASGI app.
USE_INPROCESS = True
BASE_URL = "http://127.0.0.1:8001"
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parent))


def py_sample(i: int) -> Tuple[str, int]:
    parts = []
    expected = 0
    # Syntax/indent/name/type/value/index/key/attr/import/zero-div/recursion
    parts.append(
        """
def bad_func{i}(x y):
print('missing colon and indent')
y = x + '1'  # TypeError
z = unknown_var  # NameError
arr = [1,2]
q = arr[100]  # IndexError
d = {'a': 1}
v = d['missing']  # KeyError
import not_a_module  # ImportError
res = 1/0  # ZeroDivisionError
def recurse(n):
    return recurse(n)  # RecursionError
"""
    )
    expected += 10
    parts.append(
        """
class Broken{i}:
    def __init__(self):
        self.name = 'x'
    def run(self):
        return self.nam  # AttributeError
"""
    )
    expected += 1
    parts.append(
        """
def parse_val():
    return int('abc')  # ValueError
assert 2 + 2 == 5  # AssertionError
"""
    )
    expected += 2
    # Make it heavier by adding filler functions with style issues
    for k in range(10):
        parts.append(f"def f{i}_{k}(a,b)\n    return a-b\n")  # missing colon/indent
        expected += 2
    return "\n".join(parts), expected


def js_sample(i: int) -> Tuple[str, int]:
    parts = []
    expected = 0
    parts.append(
        """
function bad{i}(x y) {
console.log('Syntax error missing comma')
let a = 10;
let b = '5';
let c = a + b(); // TypeError calling non-function
let d = unknown; // ReferenceError
let arr = [1,2];
arr.length = -1; // RangeError
decodeURI('%E0%A4%A'); // URIError
}
"""
    )
    expected += 6
    parts.append(
        """
// Promise errors
Promise.allSettled([Promise.reject('x'), Promise.reject('y')]).then(() => {
  throw new AggregateError([new Error('a'), new Error('b')], 'agg');
});
"""
    )
    expected += 1
    # Filler functions with missing semicolons and logic mistakes
    for k in range(10):
        parts.append(f"function f{i}_{k}(a,b) {{ return a-b }}")
        expected += 1
    return "\n".join(parts), expected


def java_sample(i: int) -> Tuple[str, int]:
    code = f"""
public class Broken{i} {{
    public static void main(String[] args) {{
        String s = null;
        System.out.println(s.length()); // NullPointerException
        int[] arr = new int[2];
        int x = arr[5]; // ArrayIndexOutOfBoundsException
        int y = 1/0; // ArithmeticException
        Object o = new Integer(5);
        String z = (String) o; // ClassCastException
        int n = Integer.parseInt("abc"); // NumberFormatException
    }}
}}
"""
    expected = 5
    return code, expected


def csharp_sample(i: int) -> Tuple[str, int]:
    code = f"""
using System;
using System.IO;
class Broken{i} {{
    static void Main() {{
        string s = null;
        Console.WriteLine(s.Length); // NullReferenceException
        int[] a = new int[2];
        int x = a[5]; // IndexOutOfRangeException
        int z = 1/0; // DivideByZeroException
        object o = 5;
        string t = (string)o; // InvalidCastException
        int n = int.Parse("abc"); // FormatException
        File.ReadAllText("missing.txt"); // FileNotFoundException
    }}
}}
"""
    expected = 7
    return code, expected


def cpp_sample(i: int) -> Tuple[str, int]:
    code = f"""
#include <iostream>
int* leak{i}() {{ int* p = new int[10]; p[20] = 5; return p; }} // buffer overflow + leak
int main() {{
    int *p = nullptr; std::cout << *p; // segmentation fault
    int a = 1/0; // divide by zero (UB)
    int arr[2]; arr[5] = 3; // out of bounds
    return 0
}}
"""
    expected = 4
    return code, expected


def gen_samples(lang: str, n: int = 10) -> List[Tuple[str, int]]:
    if lang == "python":
        return [py_sample(i) for i in range(n)]
    if lang == "javascript":
        return [js_sample(i) for i in range(n)]
    if lang == "java":
        return [java_sample(i) for i in range(n)]
    if lang == "csharp":
        return [csharp_sample(i) for i in range(n)]
    if lang in ("cpp", "c"):
        return [cpp_sample(i) for i in range(n)]
    return []


def ensure_bench_files(base: str = "bench") -> None:
    os.makedirs(base, exist_ok=True)
    specs = [
        ("python", ".py"),
        ("javascript", ".js"),
        ("java", ".java"),
        ("csharp", ".cs"),
        ("cpp", ".cpp"),
    ]
    for lang, ext in specs:
        lang_dir = os.path.join(base, lang)
        os.makedirs(lang_dir, exist_ok=True)
        # Create 10 files if missing
        files = [f for f in os.listdir(lang_dir) if f.endswith(ext)]
        if len(files) >= 10:
            continue
        samples = gen_samples(lang, 10)
        for i, (code, expected) in enumerate(samples, start=1):
            path = os.path.join(lang_dir, f"sample_{i}{ext}")
            if os.path.exists(path):
                continue
            header = {
                "python": f"# EXPECTED_ISSUES: {expected}\n",
                "javascript": f"// EXPECTED_ISSUES: {expected}\n",
                "java": f"// EXPECTED_ISSUES: {expected}\n",
                "csharp": f"// EXPECTED_ISSUES: {expected}\n",
                "cpp": f"// EXPECTED_ISSUES: {expected}\n",
            }[lang]
            with open(path, "w", encoding="utf-8") as f:
                f.write(header)
                f.write(code)


def read_bench_samples(lang: str, base: str = "bench") -> List[Tuple[str, int]]:
    ext_map = {
        "python": ".py",
        "javascript": ".js",
        "java": ".java",
        "csharp": ".cs",
        "cpp": ".cpp",
    }
    ext = ext_map[lang]
    lang_dir = os.path.join(base, lang)
    samples: List[Tuple[str, int]] = []
    if not os.path.isdir(lang_dir):
        return samples
    for name in sorted(os.listdir(lang_dir)):
        if not name.endswith(ext):
            continue
        path = os.path.join(lang_dir, name)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        expected = 1
        # Parse EXPECTED_ISSUES from header comment
        for line in text.splitlines()[:5]:
            if "EXPECTED_ISSUES:" in line:
                try:
                    expected = int(line.split(":", 1)[1].strip())
                except Exception:
                    expected = 1
                break
        samples.append((text, expected))
    return samples


async def call(session: httpx.AsyncClient, path: str, payload: Dict) -> Tuple[Dict, float]:
    t0 = time.perf_counter()
    r = await session.post(path, json=payload, timeout=30.0)
    t1 = time.perf_counter()
    r.raise_for_status()
    return r.json(), (t1 - t0)


async def run_bench(lang: str) -> Dict:
    # Ensure files exist and then read them
    ensure_bench_files()
    samples = read_bench_samples(lang)
    det_rates = []
    fix_rates = []
    det_lat = []
    fix_lat = []
    # Create an async client either in-process or via network
    if USE_INPROCESS:
        # Import FastAPI app and bind the client to the ASGI app for local calls
        import app as server_app
        ac = httpx.AsyncClient(app=server_app.app, base_url="http://test")
    else:
        ac = httpx.AsyncClient(base_url=BASE_URL)
    async with ac:
        for code, expected in samples:
            analyze_data, a_lat = await call(ac, "/api/analyze", {"language": lang, "code": code})
            det_lat.append(a_lat)
            total_before = int(analyze_data.get("errors", 0)) + int(analyze_data.get("warnings", 0))
            # detection percent vs expected injected issues
            detection = min(total_before, max(1, expected)) / max(1, expected) * 100.0
            det_rates.append(detection)

            fix_data, f_lat = await call(ac, "/api/fix", {"language": lang, "code": code})
            fix_lat.append(f_lat)
            fixed_code = fix_data.get("fixed", "")
            fixed_analysis, _ = await call(ac, "/api/analyze", {"language": lang, "code": fixed_code})
            total_after = int(fixed_analysis.get("errors", 0)) + int(fixed_analysis.get("warnings", 0))
            fix_percent = (max(0, total_before - total_after) / max(1, total_before)) * 100.0
            fix_rates.append(fix_percent)

    def avg(xs: List[float]) -> float:
        return sum(xs) / max(1, len(xs))

    return {
        "language": lang,
        "detection_percent": round(avg(det_rates), 2),
        "fixing_percent": round(avg(fix_rates), 2),
        "latency_analyze_sec": round(avg(det_lat), 3),
        "latency_fix_sec": round(avg(fix_lat), 3),
    }


async def main():
    langs = ["python", "javascript", "java", "csharp", "cpp"]
    results = await asyncio.gather(*(run_bench(l) for l in langs))
    print(json.dumps({"results": results}, indent=2))


if __name__ == "__main__":
    asyncio.run(main())