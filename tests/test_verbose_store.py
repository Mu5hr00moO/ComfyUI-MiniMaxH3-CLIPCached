"""Unit tests for minimaxh3_clipcache.verbose_store.

Pure dict/JSON round-trips -- no GPU, no ComfyUI (mirrors tests/test_store.py).
"""

import json
import logging

from minimaxh3_clipcache.verbose_store import (
    DEFAULT_USER_METADATA,
    load_verbose,
    save_verbose,
)

FINGERPRINT_A = "a" * 64
FINGERPRINT_B = "b" * 64


def _system(prompt="a prompt", clip_name="qwen3vl_x.safetensors"):
    return {
        "prompt": prompt,
        "clip_name": clip_name,
        "clip_file_size": 27_000_000_000,
        "clip_mtime_ns": 123456789,
        "cache_schema_version": 1,
        "references": [],
    }


def _verbose_file(tmp_path, fingerprint):
    return tmp_path / "{}.verbose.json".format(fingerprint)


def test_a_round_trip_fresh_entry_gets_default_user(tmp_path):
    system = _system()
    save_verbose(FINGERPRINT_A, system, tmp_path)
    loaded = load_verbose(FINGERPRINT_A, tmp_path)

    assert loaded["fingerprint"] == FINGERPRINT_A
    assert loaded["system"] == system
    assert loaded["user"] == DEFAULT_USER_METADATA

    # The default must be copied, not shared -- mutating a loaded entry
    # must not bleed into the module-level DEFAULT_USER_METADATA.
    loaded["user"]["tags"].append("leaked")
    assert DEFAULT_USER_METADATA["tags"] == []


def test_b_backfill_preserves_user_edits(tmp_path):
    save_verbose(FINGERPRINT_A, _system(prompt="old"), tmp_path)

    # Simulate a future Phase-5 /update endpoint editing the "user" block
    # directly on disk.
    path = _verbose_file(tmp_path, FINGERPRINT_A)
    data = json.loads(path.read_bytes())
    data["user"] = {"name": "Sidewalk interview", "notes": "keeper",
                    "tags": ["night", "dialogue"], "favorite": True}
    path.write_bytes(json.dumps(data).encode("utf-8"))

    # A later system backfill (e.g. on the next HIT) refreshes "system"
    # from freshly verified data...
    new_system = _system(prompt="refreshed prompt", clip_name="qwen3vl_y.safetensors")
    save_verbose(FINGERPRINT_A, new_system, tmp_path)

    loaded = load_verbose(FINGERPRINT_A, tmp_path)
    assert loaded["system"] == new_system
    assert loaded["user"] == {"name": "Sidewalk interview", "notes": "keeper",
                              "tags": ["night", "dialogue"], "favorite": True}


def test_c_load_missing_fingerprint_returns_none(tmp_path):
    assert load_verbose("no-such-fingerprint", tmp_path) is None


def test_d_load_corrupted_json_returns_none_with_warning(tmp_path, caplog):
    save_verbose(FINGERPRINT_A, _system(), tmp_path)
    _verbose_file(tmp_path, FINGERPRINT_A).write_bytes(b"\xff\xfe\x00not json at all\x01\x02")

    with caplog.at_level(logging.WARNING):
        result = load_verbose(FINGERPRINT_A, tmp_path)

    assert result is None
    assert any(r.levelno == logging.WARNING for r in caplog.records)
    assert any(FINGERPRINT_A in r.getMessage() for r in caplog.records)


def test_d_corrupted_sidecar_makes_save_fall_back_to_default_user(tmp_path):
    # save_verbose() must treat an unparseable existing sidecar the same as
    # a missing one: start "user" from DEFAULT_USER_METADATA rather than
    # raise.
    _verbose_file(tmp_path, FINGERPRINT_A).write_bytes(b"not json at all")
    save_verbose(FINGERPRINT_A, _system(), tmp_path)

    loaded = load_verbose(FINGERPRINT_A, tmp_path)
    assert loaded["user"] == DEFAULT_USER_METADATA


def test_e_no_leftover_tmp_files_after_save(tmp_path):
    save_verbose(FINGERPRINT_A, _system(), tmp_path)
    assert list(tmp_path.glob("*.tmp-*")) == []
