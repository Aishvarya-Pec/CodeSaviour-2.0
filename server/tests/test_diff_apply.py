import os
import sys
import pytest

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from server.app import apply_unified_diff


def test_apply_unified_diff_simple_replace():
    original = "a\nb\nc"
    diff = """--- a/x
+++ b/x
@@
 a
-b
+b2
 c
"""
    result = apply_unified_diff(original, diff)
    assert isinstance(result, str)
    assert result == "a\nb2\nc"


def test_apply_unified_diff_add_line():
    original = "a\nb\nc"
    diff = """--- a/x
+++ b/x
@@
 a
 b
+bb
 c
"""
    result = apply_unified_diff(original, diff)
    assert isinstance(result, str)
    assert result == "a\nb\nbb\nc"


def test_apply_unified_diff_context_mismatch_returns_none():
    original = "a\nb\nc"
    diff = """--- a/x
+++ b/x
@@
 x
-b
+b2
 c
"""
    result = apply_unified_diff(original, diff)
    assert result is None


def test_apply_unified_diff_multi_hunk_success():
    original = "1\n2\n3\n4\n5\n6"
    diff = """--- a/x
+++ b/x
@@
 1
-2
+two
 3
@@
 4
+fourpointfive
 5
 6
"""
    result = apply_unified_diff(original, diff)
    assert isinstance(result, str)
    assert result == "1\ntwo\n3\n4\nfourpointfive\n5\n6"


def test_apply_unified_diff_multi_hunk_conflict():
    original = "1\n2\n3\n4\n5\n6"
    diff = """--- a/x
+++ b/x
@@
 1
-2
+two
 3
@@
 X
-5
+five
 6
"""
    result = apply_unified_diff(original, diff)
    assert result is None