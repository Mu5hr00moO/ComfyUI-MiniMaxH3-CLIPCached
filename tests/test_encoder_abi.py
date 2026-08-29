"""Unit tests for minimaxh3_clipcache.encoder_abi.get_encoder_abi_id.

The success path exercises the real comfy.text_encoders.minimax module and
the real comfyui_version, both importable under pytest thanks to
conftest.py putting the ComfyUI root on sys.path. The failure path is
simulated by knocking comfy.text_encoders.minimax out of sys.modules. No
GPU, no ComfyUI startup, no model load.
"""

import logging
import sys
from pathlib import Path

import pytest

from minimaxh3_clipcache import encoder_abi
from minimaxh3_clipcache.encoder_abi import get_encoder_abi_id


@pytest.fixture(autouse=True)
def _fresh_process_cache():
    """get_encoder_abi_id() caches its result process-wide; reset before and
    after every test so tests never see each other's cached value."""
    encoder_abi._reset_for_tests()
    yield
    encoder_abi._reset_for_tests()


def test_a_success_returns_str_and_true():
    abi_id, available = get_encoder_abi_id()

    assert available is True
    assert isinstance(abi_id, str)
    assert ":" in abi_id
    # "{comfyui_version}:{sha256 hexdigest}" -- the digest half is 64 hex chars.
    version_part, _, hash_part = abi_id.partition(":")
    assert version_part
    assert len(hash_part) == 64
    int(hash_part, 16)  # raises ValueError if not valid hex


def test_b_second_call_is_cached_and_does_not_recompute(monkeypatch):
    # First call succeeds and populates the process-wide cache.
    first_id, first_available = get_encoder_abi_id()
    assert first_available is True

    # Now make any fresh recomputation impossible: reading the source file
    # would blow up. A real cache returns the stored value without touching it.
    def _boom(self, *args, **kwargs):
        raise AssertionError("read_bytes() must not be called again -- the "
                             "result is cached process-wide after the first call")

    monkeypatch.setattr(Path, "read_bytes", _boom)

    second_id, second_available = get_encoder_abi_id()

    assert (second_id, second_available) == (first_id, first_available)


def test_c_failure_returns_none_false(monkeypatch):
    # Knock the encoder module out of sys.modules: `import
    # comfy.text_encoders.minimax` then raises ImportError.
    monkeypatch.setitem(sys.modules, "comfy.text_encoders.minimax", None)

    abi_id, available = get_encoder_abi_id()

    assert abi_id is None
    assert available is False


def test_d_failure_warns_exactly_once_across_two_calls(monkeypatch, caplog):
    monkeypatch.setitem(sys.modules, "comfy.text_encoders.minimax", None)

    with caplog.at_level(logging.WARNING, logger="minimaxh3_clipcache.encoder_abi"):
        get_encoder_abi_id()
        get_encoder_abi_id()

    warnings = [r for r in caplog.records
               if r.levelno == logging.WARNING and "ENCODER ABI UNAVAILABLE" in r.getMessage()]
    assert len(warnings) == 1


def test_e_failure_is_cached_like_success(monkeypatch):
    monkeypatch.setitem(sys.modules, "comfy.text_encoders.minimax", None)
    assert get_encoder_abi_id() == (None, False)

    # Restore the module; the cached failure must still be returned because the
    # process-wide cache was already populated on the first (failing) call.
    monkeypatch.undo()
    assert get_encoder_abi_id() == (None, False)
