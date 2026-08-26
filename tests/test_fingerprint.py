"""Unit tests for caching.fingerprint.compute_fingerprint.

Pure torch.rand/torch.zeros stand-ins -- no GPU, no ComfyUI, no disk I/O.
"""

import torch

from caching.fingerprint import compute_fingerprint

CLIP_NAME = "qwen3vl_32b_minimax_h3_int8_convrot.safetensors"
FILE_SIZE = 27141342152
MTIME_NS = 1712345678000000000


def _fp(prompt="a test prompt", tokenize_kwargs=None, clip_name=CLIP_NAME,
        clip_file_size=FILE_SIZE, clip_mtime_ns=MTIME_NS, cache_schema_version=1):
    if tokenize_kwargs is None:
        tokenize_kwargs = {}
    return compute_fingerprint(prompt, tokenize_kwargs, clip_name, clip_file_size, clip_mtime_ns,
                                cache_schema_version)


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


def test_hash_is_full_sha256_hex_digest():
    fp = _fp()
    assert isinstance(fp, str)
    assert len(fp) == 64
    int(fp, 16)  # raises ValueError if not valid hex
