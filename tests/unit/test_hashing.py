"""hashing.py must be stable: same content in, same hash out, order-independent."""

from __future__ import annotations

from pathlib import Path

from evalguard.hashing import combine_hashes, hash_file, sha256_json, sha256_text


def test_sha256_text_is_deterministic() -> None:
    assert sha256_text("hello world") == sha256_text("hello world")


def test_sha256_text_differs_on_different_input() -> None:
    assert sha256_text("hello") != sha256_text("hellO")


def test_sha256_text_is_a_hex_sha256() -> None:
    digest = sha256_text("anything")
    assert len(digest) == 64
    assert all(c in "0123456789abcdef" for c in digest)


def test_sha256_json_is_order_independent_for_dict_keys() -> None:
    a = {"name": "v1", "version": "1", "role": "production"}
    b = {"role": "production", "version": "1", "name": "v1"}
    assert sha256_json(a) == sha256_json(b)


def test_sha256_json_differs_on_different_values() -> None:
    assert sha256_json({"a": 1}) != sha256_json({"a": 2})


def test_sha256_json_handles_nested_structures_and_enums() -> None:
    from evalguard.enums import PromptRole

    payload = {"role": PromptRole.PRODUCTION, "nested": {"cases": [1, 2, 3]}}
    # Must not raise, and must be stable across repeated calls.
    assert sha256_json(payload) == sha256_json(payload)


def test_hash_file_matches_sha256_text_of_its_contents(tmp_path: Path) -> None:
    content = "the quick brown fox\n"
    f = tmp_path / "sample.txt"
    f.write_text(content, encoding="utf-8")
    assert hash_file(f) == sha256_text(content)


def test_hash_file_changes_when_content_changes(tmp_path: Path) -> None:
    f = tmp_path / "sample.txt"
    f.write_text("v1", encoding="utf-8")
    h1 = hash_file(f)
    f.write_text("v2", encoding="utf-8")
    h2 = hash_file(f)
    assert h1 != h2


def test_combine_hashes_is_order_independent() -> None:
    h1, h2, h3 = "aaa", "bbb", "ccc"
    assert combine_hashes(h1, h2, h3) == combine_hashes(h3, h1, h2)


def test_combine_hashes_changes_when_inputs_change() -> None:
    assert combine_hashes("aaa", "bbb") != combine_hashes("aaa", "ccc")
