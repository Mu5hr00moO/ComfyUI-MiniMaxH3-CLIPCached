"""Unit tests for caching.loader: resolve_clip_stat() and
build_clip_loader_fn(). No GPU, no real encoder load -- comfy.sd.load_clip
is monkeypatched with a call counter to prove the returned loader is lazy.
"""

import pytest

import comfy.sd

from caching.loader import build_clip_loader_fn, resolve_clip_stat

REAL_CLIP_NAME = "qwen3vl_32b_minimax_h3_int8_convrot.safetensors"
EXPECTED_FILE_SIZE = 27141342152


def test_a_resolve_clip_stat_real_file():
    file_size, mtime_ns = resolve_clip_stat(REAL_CLIP_NAME)

    assert file_size == EXPECTED_FILE_SIZE
    assert mtime_ns > 0


def test_b_resolve_clip_stat_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        resolve_clip_stat("this_file_does_not_exist.safetensors")


def test_c_build_clip_loader_fn_is_lazy(monkeypatch):
    calls = {"count": 0}

    def fake_load_clip(**kwargs):
        calls["count"] += 1
        return "fake_clip_object"

    monkeypatch.setattr(comfy.sd, "load_clip", fake_load_clip)

    loader_fn = build_clip_loader_fn(REAL_CLIP_NAME)
    assert calls["count"] == 0

    result = loader_fn()
    assert calls["count"] == 1
    assert result == "fake_clip_object"
