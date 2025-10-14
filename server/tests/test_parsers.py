import os
import sys
import json

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from server.app import (
    parse_cppcheck_template_line,
    parse_clang_tidy_line,
    parse_pmd_json,
)


def test_parse_cppcheck_template_line_valid():
    line = "file.cpp:12:warning:unusedVariable:Variable 'x' is unused"
    d = parse_cppcheck_template_line(line)
    assert d and d["file"] == "file.cpp" and d["line"] == 12
    assert d["severity"] == "warning" and d["rule"] == "unusedVariable"


def test_parse_cppcheck_template_line_invalid():
    assert parse_cppcheck_template_line("not:a:valid:template") is None


def test_parse_clang_tidy_line_valid():
    line = "main.cpp:10:5: warning: use auto [modernize-use-auto]"
    d = parse_clang_tidy_line(line, default_file="main.cpp")
    assert d and d["file"] == "main.cpp" and d["line"] == 10
    assert d["severity"] == "warning" and d["rule"] == "modernize-use-auto"


def test_parse_clang_tidy_line_invalid():
    assert parse_clang_tidy_line("random noise", default_file="x.cpp") is None


def test_parse_pmd_json_valid():
    payload = {
        "files": [
            {
                "filename": "Sample.java",
                "violations": [
                    {
                        "beginline": 7,
                        "rule": "UnusedImports",
                        "priority": "warning",
                        "description": "Avoid unused imports",
                    }
                ],
            }
        ]
    }
    diags = parse_pmd_json(payload, default_file="Sample.java")
    assert len(diags) == 1
    d = diags[0]
    assert d["file"] == "Sample.java" and d["line"] == 7
    assert d["rule"] == "UnusedImports" and d["severity"] == "warning"