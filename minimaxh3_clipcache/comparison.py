"""Structural equality checks for the tensors/dicts/lists that make up a
MiniMaxH3ImageToVideo conditioning or AV latent output.

Shared between scripts/test_cache_roundtrip.py (stock-CLIP-path vs
cached-CLIP-path comparison) and the phase 23 stock == cached-MISS ==
cached-HIT test -- both need to walk the same nested structure, and both
need torch.equal (not allclose): a cache hit must replay the exact bytes
read from disk, not merely numerically close values from a fresh encode.

comfy.nested_tensor.NestedTensor wraps the (video, audio) tensor pair used
by the AV latent's "samples" field (see _empty_av_latent in
comfy_extras/nodes_minimax_h3.py) -- it is not a torch.Tensor subclass, so
it needs its own branch, checked before the generic torch.Tensor branch.
"""

import comfy.nested_tensor
import torch


def _tensors_equal(path, a, b):
    if isinstance(a, comfy.nested_tensor.NestedTensor) or isinstance(b, comfy.nested_tensor.NestedTensor):
        assert isinstance(a, comfy.nested_tensor.NestedTensor) and isinstance(b, comfy.nested_tensor.NestedTensor), \
            "{}: type mismatch {} vs {}".format(path, type(a), type(b))
        assert a.is_nested == b.is_nested, \
            "{}: is_nested mismatch {!r} vs {!r}".format(path, a.is_nested, b.is_nested)
        assert len(a.tensors) == len(b.tensors), \
            "{}: tensors length mismatch {} vs {}".format(path, len(a.tensors), len(b.tensors))
        for i, (ta, tb) in enumerate(zip(a.tensors, b.tensors)):
            _tensors_equal("{}.tensors[{}]".format(path, i), ta, tb)
        return
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        assert type(a) is type(b), "{}: type mismatch {} vs {}".format(path, type(a), type(b))
        assert a.shape == b.shape, "{}: shape mismatch {} vs {}".format(path, a.shape, b.shape)
        assert a.dtype == b.dtype, "{}: dtype mismatch {} vs {}".format(path, a.dtype, b.dtype)
        assert torch.equal(a, b), "{}: tensors not exactly equal".format(path)
        return
    if isinstance(a, dict) or isinstance(b, dict):
        assert set(a.keys()) == set(b.keys()), "{}: dict key mismatch {} vs {}".format(path, a.keys(), b.keys())
        for k in a:
            _tensors_equal("{}[{!r}]".format(path, k), a[k], b[k])
        return
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        # list vs tuple is a real difference here: serialize.py round-trips
        # tuple-ness (via __type__: "tuple"), and the stock node returns
        # tuples in specific spots (e.g. minimax_keyframes), so a cached
        # replay that came back as a list where the stock path has a tuple
        # is a serialization bug, not an equal value.
        assert type(a) is type(b), \
            "{}: sequence type mismatch {} vs {}".format(path, type(a).__name__, type(b).__name__)
        assert len(a) == len(b), "{}: length mismatch {} vs {}".format(path, len(a), len(b))
        for i, (ea, eb) in enumerate(zip(a, b)):
            _tensors_equal("{}[{}]".format(path, i), ea, eb)
        return
    assert a == b, "{}: {!r} != {!r}".format(path, a, b)
