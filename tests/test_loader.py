"""Unit tests for minimaxh3_clipcache.loader: resolve_clip_stat() and
build_clip_loader_fn(). No GPU, no real encoder load -- comfy.sd.load_clip
is monkeypatched with a call counter to prove the returned loader is lazy.
"""

import pytest

import comfy.sd
import folder_paths

from minimaxh3_clipcache.loader import build_clip_loader_fn, resolve_clip_stat

REAL_CLIP_NAME = "qwen3vl_32b_minimax_h3_int8_convrot.safetensors"
EXPECTED_FILE_SIZE = 27141342152

# These two tests exercise the real folder_paths lookup and therefore need
# the actual H3 encoder checkpoint on disk. It is ~27 GB and only present on
# the maintainer's machine, so skip (don't fail) where it is absent: CI, or
# a fork cloned by another user.
_real_clip_present = folder_paths.get_full_path("text_encoders", REAL_CLIP_NAME) is not None
requires_real_clip = pytest.mark.skipif(
    not _real_clip_present,
    reason="requires local H3 encoder checkpoint ({}), not present in this environment".format(REAL_CLIP_NAME),
)


@requires_real_clip
def test_a_resolve_clip_stat_real_file():
    file_size, mtime_ns = resolve_clip_stat(REAL_CLIP_NAME)

    assert file_size == EXPECTED_FILE_SIZE
    assert mtime_ns > 0


def test_b_resolve_clip_stat_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        resolve_clip_stat("this_file_does_not_exist.safetensors")


@requires_real_clip
def test_c_build_clip_loader_fn_is_lazy(monkeypatch):
    # loader_fn() calls folder_paths.get_full_path_or_raise() before it ever
    # reaches the monkeypatched load_clip, so the real file still has to exist.
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
