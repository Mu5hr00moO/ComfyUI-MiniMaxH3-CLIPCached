"""Unit tests for minimaxh3_clipcache.serialize.flatten_tensors / unflatten_tensors.

Pure torch tensors -- no GPU, no ComfyUI.
"""

import pytest
import torch

from minimaxh3_clipcache.serialize import flatten_tensors, unflatten_tensors


def test_round_trip_nested_structure():
    obj = [[torch.rand(1, 3, 8),
            {"pooled_output": None,
             "minimax_token_tags": torch.zeros(3, dtype=torch.int64),
             "minimax_keyframes": ({"resolved_frame_index": 0, "latent": torch.rand(1, 2, 3)},)}]]
    skeleton, tensors = flatten_tensors(obj)
    restored = unflatten_tensors(skeleton, tensors)

    assert isinstance(restored[0][1]["minimax_keyframes"], tuple)
    assert torch.equal(restored[0][0], obj[0][0])
    assert torch.equal(restored[0][1]["minimax_token_tags"], obj[0][1]["minimax_token_tags"])
    assert torch.equal(restored[0][1]["minimax_keyframes"][0]["latent"],
                       obj[0][1]["minimax_keyframes"][0]["latent"])


def test_dotted_dict_key_raises_value_error():
    with pytest.raises(ValueError, match=r"dict key containing '\.'"):
        flatten_tensors({"a.b": torch.rand(2)})


def test_dotted_dict_key_nested_raises_value_error():
    with pytest.raises(ValueError, match=r"dict key containing '\.'"):
        flatten_tensors([{"ok": {"also.bad": torch.rand(2)}}])


@pytest.mark.parametrize("reserved_key", ["__tensor_ref__", "__type__"])
def test_reserved_dict_key_raises_value_error(reserved_key):
    with pytest.raises(ValueError, match="reserved dict key"):
        flatten_tensors({reserved_key: "real user value"})


def test_reserved_dict_key_nested_raises_value_error():
    with pytest.raises(ValueError, match="__tensor_ref__"):
        flatten_tensors([{"safe": {"__tensor_ref__": "not an internal marker"}}])
