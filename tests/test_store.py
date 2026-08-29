"""Unit tests for minimaxh3_clipcache.store.save_conditioning / load_conditioning.

Pure torch.rand/torch.zeros stand-ins -- no GPU, no ComfyUI.
"""

import json
import logging
import os
from pathlib import Path

import pytest
import torch

from minimaxh3_clipcache import store
from minimaxh3_clipcache.locking import get_lock
from minimaxh3_clipcache.store import gc_orphaned_cache_files
from minimaxh3_clipcache.store import (
    inspect_conditioning_pair,
    load_conditioning,
    save_conditioning,
)

FINGERPRINT_A = "a" * 64
FINGERPRINT_B = "b" * 64


def _cond_variant_a():
    return [[torch.rand(1, 3, 5120, dtype=torch.float32),
             {"pooled_output": None, "minimax_token_tags": torch.zeros(3, dtype=torch.int64)}]]


def _cond_variant_b():
    return [[torch.rand(1, 1019, 5120, dtype=torch.float32),
             {"pooled_output": None, "minimax_token_tags": torch.zeros(1019, dtype=torch.int64)}]]


def _cond_variant_a_with_different_values():
    """Byte-for-byte the same skeleton as _cond_variant_a() -- identical
    nesting, keys, tensor shapes and dtypes -- but different numeric values in
    the tensors. This is the realistic shape of a cache_mode="refresh" of the
    same inputs: the structure never moves, only the encoded values do."""
    return [[torch.full((1, 3, 5120), 0.5, dtype=torch.float32),
             {"pooled_output": None, "minimax_token_tags": torch.ones(3, dtype=torch.int64)}]]


def _assert_cond_equal(original, loaded):
    assert type(loaded) is type(original)
    assert len(loaded) == len(original)
    for orig_entry, loaded_entry in zip(original, loaded):
        assert type(loaded_entry) is type(orig_entry)
        orig_tensor, orig_extra = orig_entry
        loaded_tensor, loaded_extra = loaded_entry
        assert torch.equal(orig_tensor, loaded_tensor)
        assert set(orig_extra.keys()) == set(loaded_extra.keys())
        for key in orig_extra:
            orig_val, loaded_val = orig_extra[key], loaded_extra[key]
            if isinstance(orig_val, torch.Tensor):
                assert torch.equal(orig_val, loaded_val)
            else:
                assert orig_val == loaded_val


def test_a_round_trip_variant_a(tmp_path):
    cond = _cond_variant_a()
    save_conditioning(FINGERPRINT_A, cond, tmp_path)
    loaded = load_conditioning(FINGERPRINT_A, tmp_path)
    _assert_cond_equal(cond, loaded)


def test_a_round_trip_variant_b(tmp_path):
    cond = _cond_variant_b()
    save_conditioning(FINGERPRINT_B, cond, tmp_path)
    loaded = load_conditioning(FINGERPRINT_B, tmp_path)
    _assert_cond_equal(cond, loaded)


def test_a_round_trip_preserves_tuple_vs_list(tmp_path):
    cond = (
        [torch.rand(2, 4), {"pooled_output": None}],
        [torch.zeros(1), {"minimax_keyframes": ({"resolved_frame_index": 0, "latent": torch.rand(1, 2, 3)},)}],
    )
    fp = "c" * 64
    save_conditioning(fp, cond, tmp_path)
    loaded = load_conditioning(fp, tmp_path)

    assert isinstance(loaded, tuple)
    assert isinstance(loaded[0], list)
    assert isinstance(loaded[1], list)
    assert isinstance(loaded[1][1]["minimax_keyframes"], tuple)
    assert torch.equal(loaded[0][0], cond[0][0])
    assert torch.equal(loaded[1][1]["minimax_keyframes"][0]["latent"], cond[1][1]["minimax_keyframes"][0]["latent"])
    assert loaded[1][1]["minimax_keyframes"][0]["resolved_frame_index"] == 0


def test_b_no_leftover_tmp_files_after_save(tmp_path):
    save_conditioning(FINGERPRINT_A, _cond_variant_a(), tmp_path)
    leftover = list(tmp_path.glob("*.tmp-*"))
    assert leftover == []


def test_b_failed_save_leaves_nothing_behind_and_reraises(tmp_path, monkeypatch):
    # Fail the second os.replace() -- the .json move -- on a fresh write,
    # after the .safetensors has already been published and the .json temp
    # is already written. Under the two-phase contract the exception
    # propagates untouched, save_conditioning() removes only its own
    # never-consumed temp file(s), and it deliberately does NOT undo the
    # .safetensors replace that already succeeded. That leaves a
    # self-healable orphan .safetensors with no .json -- cleaned up by
    # load_conditioning()'s own self-heal or gc_orphaned_cache_files(), see
    # store.py's Phase 2 comment.
    real_replace = os.replace
    calls = {"n": 0}

    def flaky_replace(src, dst):
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("simulated failure moving .json into place")
        return real_replace(src, dst)

    monkeypatch.setattr(store.os, "replace", flaky_replace)

    with pytest.raises(OSError, match="simulated failure moving .json"):
        save_conditioning(FINGERPRINT_A, _cond_variant_a(), tmp_path)

    assert calls["n"] == 2  # the second replace really was reached
    assert list(tmp_path.glob("*.tmp-*")) == []
    assert (tmp_path / "{}.safetensors".format(FINGERPRINT_A)).exists()
    assert not (tmp_path / "{}.json".format(FINGERPRINT_A)).exists()


def test_c_missing_safetensors_after_json_written_returns_none(tmp_path, caplog):
    save_conditioning(FINGERPRINT_A, _cond_variant_a(), tmp_path)
    (tmp_path / "{}.safetensors".format(FINGERPRINT_A)).unlink()

    with caplog.at_level(logging.WARNING):
        result = load_conditioning(FINGERPRINT_A, tmp_path)

    assert result is None
    assert any(FINGERPRINT_A in r.getMessage() for r in caplog.records)
    assert any(r.levelno == logging.WARNING for r in caplog.records)


def test_d_corrupted_json_returns_none(tmp_path, caplog):
    save_conditioning(FINGERPRINT_A, _cond_variant_a(), tmp_path)
    (tmp_path / "{}.json".format(FINGERPRINT_A)).write_bytes(b"\xff\xfe\x00\xffnot json at all\x01\x02")

    with caplog.at_level(logging.WARNING):
        result = load_conditioning(FINGERPRINT_A, tmp_path)

    assert result is None
    assert any(r.levelno == logging.WARNING for r in caplog.records)
    assert any(FINGERPRINT_A in r.getMessage() for r in caplog.records)


def test_d_safetensors_from_a_different_entry_returns_none(tmp_path, caplog):
    # A's .safetensors replaced wholesale with another entry's file. Each
    # save_conditioning() call stamps its own generation id into both
    # artifacts, so the swapped-in .safetensors carries a different
    # cache_generation_id than A's .json -- caught by the generation-id gate
    # before any tensor data is even read.
    save_conditioning(FINGERPRINT_A, _cond_variant_a(), tmp_path)
    other_fingerprint = "d" * 64
    save_conditioning(other_fingerprint, [torch.rand(2, 2)], tmp_path)

    (tmp_path / "{}.safetensors".format(FINGERPRINT_A)).write_bytes(
        (tmp_path / "{}.safetensors".format(other_fingerprint)).read_bytes())

    with caplog.at_level(logging.WARNING):
        result = load_conditioning(FINGERPRINT_A, tmp_path)

    assert result is None
    assert any("generation_id mismatch" in r.getMessage() for r in caplog.records)
    assert any(FINGERPRINT_A in r.getMessage() for r in caplog.records)


def test_d_safe_open_oserror_is_a_clean_miss(tmp_path, monkeypatch, caplog):
    save_conditioning(FINGERPRINT_A, _cond_variant_a(), tmp_path)

    def unreadable_header(*args, **kwargs):
        raise OSError("simulated filesystem read failure")

    monkeypatch.setattr(store, "safe_open", unreadable_header)
    with caplog.at_level(logging.WARNING):
        assert load_conditioning(FINGERPRINT_A, tmp_path) is None
    assert any("failed to read metadata" in r.getMessage() for r in caplog.records)


def test_d_load_file_runtimeerror_is_a_clean_miss(tmp_path, monkeypatch, caplog):
    save_conditioning(FINGERPRINT_A, _cond_variant_a(), tmp_path)

    def failing_load(*args, **kwargs):
        raise RuntimeError("simulated torch/safetensors runtime failure")

    monkeypatch.setattr(store, "load_file", failing_load)
    with caplog.at_level(logging.WARNING):
        assert load_conditioning(FINGERPRINT_A, tmp_path) is None
    assert any("failed to load" in r.getMessage() for r in caplog.records)


def test_d_pair_inspection_checks_generation_without_loading_tensors(tmp_path, monkeypatch):
    save_conditioning(FINGERPRINT_A, _cond_variant_a(), tmp_path)

    def must_not_load(*args, **kwargs):
        raise AssertionError("inspection must not load tensor payloads")

    monkeypatch.setattr(store, "load_file", must_not_load)
    assert inspect_conditioning_pair(FINGERPRINT_A, tmp_path) == (True, None)

    payload_path = tmp_path / "{}.json".format(FINGERPRINT_A)
    payload = json.loads(payload_path.read_bytes())
    payload["generation_id"] = "different-generation"
    payload_path.write_bytes(json.dumps(payload).encode("utf-8"))
    assert inspect_conditioning_pair(FINGERPRINT_A, tmp_path) == (
        False, "generation_mismatch",
    )


def test_d_skeleton_references_missing_tensor_key_returns_none(tmp_path, caplog):
    # Last line of defense, after the generation-id gate: the .json envelope is
    # valid and its generation id matches the .safetensors, but the skeleton
    # references a tensor path that isn't in the file. Keep the entry's real
    # generation id and swap only the skeleton so unflatten_tensors()' own
    # KeyError guard is the one that fires.
    from minimaxh3_clipcache.serialize import flatten_tensors

    save_conditioning(FINGERPRINT_A, _cond_variant_a(), tmp_path)
    json_path = tmp_path / "{}.json".format(FINGERPRINT_A)
    payload = json.loads(json_path.read_bytes())
    bad_skeleton, _ = flatten_tensors([torch.rand(2, 2)])  # references key "0", absent from A
    payload["skeleton"] = bad_skeleton
    json_path.write_bytes(json.dumps(payload).encode("utf-8"))

    with caplog.at_level(logging.WARNING):
        result = load_conditioning(FINGERPRINT_A, tmp_path)

    assert result is None
    assert any("reconstructing" in r.getMessage() for r in caplog.records)
    assert any(FINGERPRINT_A in r.getMessage() for r in caplog.records)


def test_generation_mismatch_with_identical_skeleton_structure_is_still_a_clean_miss(tmp_path, monkeypatch):
    """Residual gap z audytu: nawet gdy stary i nowy skeleton są
    strukturalnie identyczne (realistyczny refresh tych samych wejść),
    awaria DRUGIEGO os.replace (json) po udanym PIERWSZYM (safetensors)
    nie może zostać po cichu wczytana jako poprawny wpis - musi być
    jednoznacznym MISS-em, niezależnie od tego czy unflatten_tensors()
    przypadkiem by się nie wysypał na zgodnej strukturze."""
    old_cond = _cond_variant_a()
    save_conditioning(FINGERPRINT_A, old_cond, tmp_path)

    new_cond = _cond_variant_a_with_different_values()

    original_replace = os.replace

    def flaky_replace(src, dst):
        if str(dst).endswith(".json"):
            raise OSError("simulated failure on the second (json) replace")
        return original_replace(src, dst)

    monkeypatch.setattr(store.os, "replace", flaky_replace)

    with pytest.raises(OSError, match="simulated failure"):
        save_conditioning(FINGERPRINT_A, new_cond, tmp_path)

    loaded = load_conditioning(FINGERPRINT_A, tmp_path)
    assert loaded is None, (
        "generation mismatch must MISS even when skeleton structure matches"
    )


def test_e_missing_fingerprint_returns_none_without_warning(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        result = load_conditioning("no-such-fingerprint", tmp_path)

    assert result is None
    assert caplog.records == []


def test_f_load_self_heals_orphaned_safetensors_without_json(tmp_path, caplog):
    # Simulate a process killed between the two os.replace() calls in
    # save_conditioning(): a .safetensors exists with no matching .json.
    save_conditioning(FINGERPRINT_A, _cond_variant_a(), tmp_path)
    (tmp_path / "{}.json".format(FINGERPRINT_A)).unlink()
    assert (tmp_path / "{}.safetensors".format(FINGERPRINT_A)).exists()

    with caplog.at_level(logging.WARNING):
        result = load_conditioning(FINGERPRINT_A, tmp_path)

    assert result is None
    assert not (tmp_path / "{}.safetensors".format(FINGERPRINT_A)).exists()
    assert any(FINGERPRINT_A in r.getMessage() for r in caplog.records)
    assert any(r.levelno == logging.WARNING for r in caplog.records)


def test_f_load_self_heal_leaves_a_fresh_write_of_the_same_fingerprint_intact(tmp_path):
    # After the self-heal above, writing and reading the same fingerprint
    # again must work exactly as normal -- the orphan is gone, not the
    # ability to use that fingerprint.
    save_conditioning(FINGERPRINT_A, _cond_variant_a(), tmp_path)
    (tmp_path / "{}.json".format(FINGERPRINT_A)).unlink()
    load_conditioning(FINGERPRINT_A, tmp_path)  # triggers the self-heal

    cond = _cond_variant_b()
    save_conditioning(FINGERPRINT_A, cond, tmp_path)
    loaded = load_conditioning(FINGERPRINT_A, tmp_path)
    _assert_cond_equal(cond, loaded)


def test_g_gc_removes_only_orphaned_safetensors(tmp_path):
    cond_a = _cond_variant_a()
    save_conditioning(FINGERPRINT_A, cond_a, tmp_path)
    cond_b = _cond_variant_b()
    save_conditioning(FINGERPRINT_B, cond_b, tmp_path)
    orphan_fp = "e" * 64
    save_conditioning(orphan_fp, _cond_variant_a(), tmp_path)
    (tmp_path / "{}.json".format(orphan_fp)).unlink()

    removed = gc_orphaned_cache_files(tmp_path)

    assert removed == [orphan_fp]
    assert not (tmp_path / "{}.safetensors".format(orphan_fp)).exists()
    _assert_cond_equal(cond_a, load_conditioning(FINGERPRINT_A, tmp_path))
    _assert_cond_equal(cond_b, load_conditioning(FINGERPRINT_B, tmp_path))


def test_g_gc_never_deletes_non_cache_safetensors_files(tmp_path):
    user_file = tmp_path / "user-model.safetensors"
    user_file.write_bytes(b"not owned by this cache")
    uppercase_hash_file = tmp_path / (("A" * 64) + ".safetensors")
    uppercase_hash_file.write_bytes(b"also not a canonical cache filename")

    assert gc_orphaned_cache_files(tmp_path) == []
    assert user_file.read_bytes() == b"not owned by this cache"
    assert uppercase_hash_file.exists()


def test_g_gc_on_empty_or_missing_dir_returns_empty_list(tmp_path):
    assert gc_orphaned_cache_files(tmp_path) == []
    assert gc_orphaned_cache_files(tmp_path / "does-not-exist") == []


def _make_orphan(tmp_path, fingerprint):
    """Leave a "<fp>.safetensors" on disk with no matching "<fp>.json",
    exactly what save_conditioning() leaves behind if it is killed between
    its two os.replace() calls."""
    save_conditioning(fingerprint, _cond_variant_a(), tmp_path)
    (tmp_path / "{}.json".format(fingerprint)).unlink()
    assert (tmp_path / "{}.safetensors".format(fingerprint)).exists()


def test_g_gc_skips_an_orphan_while_a_writer_holds_its_fingerprint_lock(tmp_path):
    # A ".safetensors bez .json" state is exactly what save_conditioning()
    # shows in the window between its two os.replace() calls -- and during
    # that window it is still holding get_lock(fingerprint). GC must not
    # mistake a mid-publish entry for an orphan and delete the tensors out
    # from under the writer, which would then publish a .json pointing at
    # nothing.
    fingerprint = "f" * 64
    _make_orphan(tmp_path, fingerprint)

    lock = get_lock(fingerprint)
    assert lock.acquire(blocking=False)
    try:
        removed = gc_orphaned_cache_files(tmp_path)
        assert removed == []
        assert (tmp_path / "{}.safetensors".format(fingerprint)).exists()
    finally:
        lock.release()

    # Once the writer is done, the next Check sweeps it as normal.
    removed = gc_orphaned_cache_files(tmp_path)
    assert removed == [fingerprint]
    assert not (tmp_path / "{}.safetensors".format(fingerprint)).exists()


def test_g_gc_removes_an_orphan_when_no_writer_holds_the_lock_and_leaves_it_free(tmp_path):
    # Regression guard for the KROK E behaviour: an orphan whose fingerprint
    # lock is free is still removed on the spot, and GC must not leave that
    # lock held afterwards.
    fingerprint = "1" * 64
    _make_orphan(tmp_path, fingerprint)

    assert gc_orphaned_cache_files(tmp_path) == [fingerprint]
    assert not (tmp_path / "{}.safetensors".format(fingerprint)).exists()

    lock = get_lock(fingerprint)
    assert lock.acquire(blocking=False), "gc_orphaned_cache_files left a fingerprint lock held"
    lock.release()


# ---------------------------------------------------------------------------
# Refresh safety: a failed save_conditioning() over an ALREADY-CACHED
# fingerprint (the cache_mode="refresh" path) must never destroy or corrupt
# the entry it was refreshing. The old entry stays readable, its on-disk
# bytes stay byte-identical, and no temp litter is left behind.
# ---------------------------------------------------------------------------


def _read_entry_bytes(tmp_path, fingerprint):
    return (
        (tmp_path / "{}.safetensors".format(fingerprint)).read_bytes(),
        (tmp_path / "{}.json".format(fingerprint)).read_bytes(),
    )


def test_refresh_failure_preserves_old_entry_when_tensor_write_fails(tmp_path, monkeypatch):
    old_cond = _cond_variant_a()
    save_conditioning(FINGERPRINT_A, old_cond, tmp_path)
    st_before, js_before = _read_entry_bytes(tmp_path, FINGERPRINT_A)

    # A real safetensors write that dies partway through (disk full, killed
    # thread) can leave a partial temp file behind before it raises --
    # simulate that, so cleanup of our own temp files is actually exercised.
    def failing_save_file(tensors, path, metadata=None):
        Path(path).write_bytes(b"partial-safetensors-garbage")
        raise RuntimeError("simulated safetensors write failure")

    monkeypatch.setattr(store, "save_file", failing_save_file)

    with pytest.raises(RuntimeError, match="simulated safetensors write failure"):
        save_conditioning(FINGERPRINT_A, _cond_variant_b(), tmp_path)

    _assert_cond_equal(old_cond, load_conditioning(FINGERPRINT_A, tmp_path))
    st_after, js_after = _read_entry_bytes(tmp_path, FINGERPRINT_A)
    assert st_after == st_before
    assert js_after == js_before
    assert list(tmp_path.glob("*.tmp-*")) == []


def test_refresh_failure_preserves_old_entry_when_skeleton_write_fails(tmp_path, monkeypatch):
    old_cond = _cond_variant_a()
    save_conditioning(FINGERPRINT_A, old_cond, tmp_path)
    st_before, js_before = _read_entry_bytes(tmp_path, FINGERPRINT_A)

    # Exactly the audit scenario: the .safetensors temp write has already
    # succeeded and then writing the .json skeleton fails -- Phase 1 dies
    # after the first of its two writes, before anything is published.
    real_write_bytes = store.Path.write_bytes

    def flaky_write_bytes(self, data):
        if ".json.tmp-" in self.name:
            raise OSError("simulated failure writing .json skeleton")
        return real_write_bytes(self, data)

    monkeypatch.setattr(store.Path, "write_bytes", flaky_write_bytes)

    with pytest.raises(OSError, match="simulated failure writing .json skeleton"):
        save_conditioning(FINGERPRINT_A, _cond_variant_b(), tmp_path)

    _assert_cond_equal(old_cond, load_conditioning(FINGERPRINT_A, tmp_path))
    st_after, js_after = _read_entry_bytes(tmp_path, FINGERPRINT_A)
    assert st_after == st_before
    assert js_after == js_before
    assert list(tmp_path.glob("*.tmp-*")) == []
