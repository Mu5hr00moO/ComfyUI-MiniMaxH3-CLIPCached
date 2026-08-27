"""Unit tests for minimaxh3_clipcache.comparison._tensors_equal.

Pure torch tensors and comfy.nested_tensor.NestedTensor -- no GPU, no
ComfyUI startup, no model loading (see conftest.py for how `import comfy...`
resolves without launching ComfyUI).
"""

import comfy.nested_tensor
import pytest
import torch

from minimaxh3_clipcache.comparison import _tensors_equal


def _av_pair(seed=0.0):
    video = torch.arange(24, dtype=torch.float32).reshape(1, 2, 3, 4) + seed
    audio = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4) + seed
    return video, audio


def test_a_identical_plain_tensors_are_equal():
    a = torch.rand(2, 3)
    _tensors_equal("t", a, a.clone())


def test_b_different_plain_tensors_are_not_equal():
    a = torch.zeros(2, 3)
    b = torch.ones(2, 3)
    with pytest.raises(AssertionError):
        _tensors_equal("t", a, b)


def test_c_identical_nested_tensors_are_equal():
    video, audio = _av_pair()
    a = comfy.nested_tensor.NestedTensor((video.clone(), audio.clone()))
    b = comfy.nested_tensor.NestedTensor((video.clone(), audio.clone()))
    _tensors_equal("samples", a, b)


def test_d_nested_tensors_differing_in_audio_are_not_equal():
    video, audio = _av_pair()
    a = comfy.nested_tensor.NestedTensor((video.clone(), audio.clone()))
    other_audio = audio.clone()
    other_audio[0, 0, 0] += 1.0
    b = comfy.nested_tensor.NestedTensor((video.clone(), other_audio))
    with pytest.raises(AssertionError):
        _tensors_equal("samples", a, b)


def test_e_nested_tensor_alongside_plain_tensor_in_a_list():
    # Mirrors the shape of a real execute() output: [tensor, {"samples": NestedTensor(...)}]
    video, audio = _av_pair()
    plain = torch.rand(1, 5, 7)
    nested = comfy.nested_tensor.NestedTensor((video.clone(), audio.clone()))

    a = [plain.clone(), {"samples": nested}]
    b = [plain.clone(), {"samples": comfy.nested_tensor.NestedTensor((video.clone(), audio.clone()))}]
    _tensors_equal("output", a, b)

    mismatched_nested = comfy.nested_tensor.NestedTensor((video.clone(), audio.clone() + 1.0))
    b_mismatched = [plain.clone(), {"samples": mismatched_nested}]
    with pytest.raises(AssertionError):
        _tensors_equal("output", a, b_mismatched)
