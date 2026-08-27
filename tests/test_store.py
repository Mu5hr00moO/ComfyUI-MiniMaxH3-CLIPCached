"""Unit tests for minimaxh3_clipcache.store.save_conditioning / load_conditioning.

Pure torch.rand/torch.zeros stand-ins -- no GPU, no ComfyUI.
"""

import logging

import torch

from minimaxh3_clipcache.store import load_conditioning, save_conditioning

FINGERPRINT_A = "a" * 64
FINGERPRINT_B = "b" * 64


def _cond_variant_a():
    return [[torch.rand(1, 3, 5120, dtype=torch.float32),
             {"pooled_output": None, "minimax_token_tags": torch.zeros(3, dtype=torch.int64)}]]


def _cond_variant_b():
    return [[torch.rand(1, 1019, 5120, dtype=torch.float32),
             {"pooled_output": None, "minimax_token_tags": torch.zeros(1019, dtype=torch.int64)}]]


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


def test_d_skeleton_tensor_mismatch_returns_none(tmp_path, caplog):
    # Both files individually load fine, but the tensor paths in the A
    # skeleton ("0.0", "0.1.minimax_token_tags") don't exist in a
    # differently-shaped cond's tensor dict (just "0") -- a natural key
    # mismatch to reconstruct from.
    save_conditioning(FINGERPRINT_A, _cond_variant_a(), tmp_path)
    other_fingerprint = "d" * 64
    save_conditioning(other_fingerprint, [torch.rand(2, 2)], tmp_path)

    (tmp_path / "{}.safetensors".format(FINGERPRINT_A)).write_bytes(
        (tmp_path / "{}.safetensors".format(other_fingerprint)).read_bytes())

    with caplog.at_level(logging.WARNING):
        result = load_conditioning(FINGERPRINT_A, tmp_path)

    assert result is None
    assert any(r.levelno == logging.WARNING for r in caplog.records)
    assert any(FINGERPRINT_A in r.getMessage() for r in caplog.records)


def test_e_missing_fingerprint_returns_none_without_warning(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        result = load_conditioning("no-such-fingerprint", tmp_path)

    assert result is None
    assert caplog.records == []
