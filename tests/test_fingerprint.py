"""Unit tests for minimaxh3_clipcache.fingerprint.compute_fingerprint.

Pure torch.rand/torch.zeros stand-ins -- no GPU, no ComfyUI, no disk I/O.
"""

import hashlib
import json

import torch

import pytest

from minimaxh3_clipcache.fingerprint import _hash_tensor, compute_fingerprint

CLIP_NAME = "qwen3vl_32b_minimax_h3_int8_convrot.safetensors"
FILE_SIZE = 27141342152
MTIME_NS = 1712345678000000000
CTIME_NS = 1712345678999999999
ABI_ID = "test-abi-id"


def _fp(prompt="a test prompt", tokenize_kwargs=None, clip_name=CLIP_NAME,
        clip_file_size=FILE_SIZE, clip_mtime_ns=MTIME_NS, cache_schema_version=1,
        encoder_abi_id=ABI_ID, clip_ctime_ns=CTIME_NS):
    if tokenize_kwargs is None:
        tokenize_kwargs = {}
    return compute_fingerprint(prompt, tokenize_kwargs, clip_name, clip_file_size, clip_mtime_ns,
                                cache_schema_version, encoder_abi_id=encoder_abi_id,
                                clip_ctime_ns=clip_ctime_ns)


def test_a_identical_inputs_same_hash():
    img = torch.rand(1, 64, 64, 3)
    fp1 = _fp(tokenize_kwargs={"images": [img]})
    fp2 = _fp(tokenize_kwargs={"images": [img.clone()]})
    assert fp1 == fp2


def test_a_identical_inputs_same_object_same_hash():
    img = torch.rand(1, 64, 64, 3)
    kwargs = {"images": [img]}
    assert _fp(tokenize_kwargs=kwargs) == _fp(tokenize_kwargs=kwargs)


def test_b_different_prompt_different_hash():
    img = torch.rand(1, 64, 64, 3)
    fp1 = _fp(prompt="a test prompt", tokenize_kwargs={"images": [img]})
    fp2 = _fp(prompt="a different prompt", tokenize_kwargs={"images": [img.clone()]})
    assert fp1 != fp2


def test_c_single_pixel_change_different_hash():
    img1 = torch.zeros(1, 64, 64, 3)
    img2 = img1.clone()
    img2[0, 0, 0, 0] = 1.0
    fp1 = _fp(tokenize_kwargs={"images": [img1]})
    fp2 = _fp(tokenize_kwargs={"images": [img2]})
    assert fp1 != fp2


def test_d_same_content_different_shape_different_hash():
    flat = torch.rand(64 * 64 * 3)
    img_a = flat.reshape(1, 64, 64, 3)
    img_b = flat.reshape(1, 192, 64, 1)  # same underlying bytes, different shape
    fp1 = _fp(tokenize_kwargs={"images": [img_a]})
    fp2 = _fp(tokenize_kwargs={"images": [img_b]})
    assert fp1 != fp2


def test_e_different_clip_name_different_hash():
    img = torch.rand(1, 64, 64, 3)
    fp1 = _fp(tokenize_kwargs={"images": [img]}, clip_name="clip_a.safetensors")
    fp2 = _fp(tokenize_kwargs={"images": [img.clone()]}, clip_name="clip_b.safetensors")
    assert fp1 != fp2


def test_f_different_file_size_different_hash():
    img = torch.rand(1, 64, 64, 3)
    fp1 = _fp(tokenize_kwargs={"images": [img]}, clip_file_size=1000)
    fp2 = _fp(tokenize_kwargs={"images": [img.clone()]}, clip_file_size=1001)
    assert fp1 != fp2


def test_f_different_mtime_ns_different_hash():
    img = torch.rand(1, 64, 64, 3)
    fp1 = _fp(tokenize_kwargs={"images": [img]}, clip_mtime_ns=111)
    fp2 = _fp(tokenize_kwargs={"images": [img.clone()]}, clip_mtime_ns=222)
    assert fp1 != fp2


def test_f_same_name_size_mtime_but_different_ctime_different_hash():
    img = torch.rand(1, 64, 64, 3)
    fp1 = _fp(tokenize_kwargs={"images": [img]}, clip_ctime_ns=111)
    fp2 = _fp(tokenize_kwargs={"images": [img.clone()]}, clip_ctime_ns=222)
    assert fp1 != fp2


def test_f_scalar_sequence_boundaries_prevent_known_collision():
    # Before scalar values were length-delimited, both sequences fed the
    # literal bytes b"1.023.0" into SHA-256.
    assert _fp(tokenize_kwargs={"x": [1.0, 23.0]}) != \
           _fp(tokenize_kwargs={"x": [1.02, 3.0]})


def test_g_same_image_list_order_same_hash():
    img1 = torch.rand(1, 64, 64, 3)
    img2 = torch.rand(1, 64, 64, 3)
    fp1 = _fp(tokenize_kwargs={"images": [img1, img2]})
    fp2 = _fp(tokenize_kwargs={"images": [img1.clone(), img2.clone()]})
    assert fp1 == fp2


def test_g_reversed_image_order_different_hash():
    img1 = torch.rand(1, 64, 64, 3)
    img2 = torch.rand(1, 64, 64, 3)
    fp_forward = _fp(tokenize_kwargs={"images": [img1, img2]})
    fp_reversed = _fp(tokenize_kwargs={"images": [img2, img1]})
    assert fp_forward != fp_reversed


def test_h_empty_images_list_vs_missing_images_key_different_hash():
    # Design decision (see compute_fingerprint docstring): the stock node
    # always passes images=[] when there are no keyframes, so an empty list
    # and a wholly absent "images" key represent different call shapes and
    # must not collide under one cache key.
    fp_empty_list = _fp(tokenize_kwargs={"images": []})
    fp_missing_key = _fp(tokenize_kwargs={})
    assert fp_empty_list != fp_missing_key


def test_i_different_schema_version_different_hash():
    img = torch.rand(1, 64, 64, 3)
    fp1 = _fp(tokenize_kwargs={"images": [img]}, cache_schema_version=1)
    fp2 = _fp(tokenize_kwargs={"images": [img.clone()]}, cache_schema_version=2)
    assert fp1 != fp2


def test_i_different_encoder_abi_id_different_hash():
    # Plan audit point 1: the encoder ABI identity (comfyui_version + a hash of
    # comfy/text_encoders/minimax.py) is folded into the digest, so an upstream
    # tokenizer/preprocessing change invalidates old entries instead of being
    # served as a stale HIT.
    img = torch.rand(1, 64, 64, 3)
    fp1 = _fp(tokenize_kwargs={"images": [img]}, encoder_abi_id="0.34.2:aaaa")
    fp2 = _fp(tokenize_kwargs={"images": [img.clone()]}, encoder_abi_id="0.34.2:bbbb")
    assert fp1 != fp2


def test_i_encoder_abi_id_is_required_keyword_only():
    # No default: a caller can never silently omit it and fall back to a
    # weaker key. Positional call with every other argument still raises.
    with pytest.raises(TypeError):
        compute_fingerprint("a prompt", {"images": []}, CLIP_NAME, FILE_SIZE, MTIME_NS, 1)


def test_j_changing_only_first_frame_slot_different_hash():
    # images=[first_frame, last_frame], as the stock node builds it when both
    # are present -- changing only index 0 (first_frame) must change the hash
    # even though index 1 (last_frame) is untouched.
    first_frame = torch.rand(1, 64, 64, 3)
    last_frame = torch.rand(1, 64, 64, 3)
    other_first_frame = torch.rand(1, 64, 64, 3)
    fp1 = _fp(tokenize_kwargs={"images": [first_frame, last_frame]})
    fp2 = _fp(tokenize_kwargs={"images": [other_first_frame, last_frame.clone()]})
    assert fp1 != fp2


def test_k_changing_only_last_frame_slot_different_hash():
    # Same as above, mirrored: changing only index 1 (last_frame) must change
    # the hash even though index 0 (first_frame) is untouched.
    first_frame = torch.rand(1, 64, 64, 3)
    last_frame = torch.rand(1, 64, 64, 3)
    other_last_frame = torch.rand(1, 64, 64, 3)
    fp1 = _fp(tokenize_kwargs={"images": [first_frame, last_frame]})
    fp2 = _fp(tokenize_kwargs={"images": [first_frame.clone(), other_last_frame]})
    assert fp1 != fp2


def test_hash_is_full_sha256_hex_digest():
    fp = _fp()
    assert isinstance(fp, str)
    assert len(fp) == 64
    int(fp, 16)  # raises ValueError if not valid hex


def test_l_hash_tensor_float32_memoryview_matches_numpy_bytes_with_new_framing():
    # Regression guard: the allocation-free memoryview path must feed the
    # same tensor bytes as numpy.tobytes(); only the deliberate framing bytes
    # around metadata/payload are added by the new fingerprint algorithm.
    t = torch.arange(60, dtype=torch.float32).reshape(3, 4, 5)

    h_new = hashlib.sha256()
    _hash_tensor(h_new, t)

    tt = t.detach().cpu().contiguous()
    h_legacy = hashlib.sha256()
    metadata = json.dumps(
        {"shape": list(tt.shape), "dtype": str(tt.dtype)}, sort_keys=True,
    ).encode("utf-8")
    h_legacy.update(len(metadata).to_bytes(8, "big"))
    h_legacy.update(metadata)
    raw = tt.reshape(-1).view(torch.uint8).numpy().tobytes()
    h_legacy.update(len(raw).to_bytes(8, "big"))
    h_legacy.update(raw)

    assert h_new.hexdigest() == h_legacy.hexdigest()


def test_m_hash_tensor_bfloat16_does_not_raise_and_is_stable():
    # t.numpy() raises "unsupported ScalarType BFloat16"; the flat-uint8 view
    # must handle it and return a stable digest across calls.
    t = torch.randn(3, 4, 5).bfloat16()

    h1 = hashlib.sha256()
    _hash_tensor(h1, t)
    h2 = hashlib.sha256()
    _hash_tensor(h2, t)

    assert h1.hexdigest() == h2.hexdigest()


def test_n_bfloat16_image_fingerprint_stable_and_pixel_sensitive():
    img = torch.randn(1, 8, 8, 3).bfloat16()
    fp1 = _fp(tokenize_kwargs={"images": [img]})
    fp2 = _fp(tokenize_kwargs={"images": [img.clone()]})
    assert fp1 == fp2

    changed = img.clone()
    changed[0, 0, 0, 0] = changed[0, 0, 0, 0] + 5.0
    assert _fp(tokenize_kwargs={"images": [changed]}) != fp1
