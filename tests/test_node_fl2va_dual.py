"""Unit tests for MiniMaxH3CLIPCachedFL2VADualRes -- the two-resolution
sibling of MiniMaxH3CLIPCachedFL2VA. Same no-GPU harness as test_node.py
(FakeRealClip, _patch_common, a monkeypatched MiniMaxH3ImageToVideo.execute,
the real CachedClipProxy writing/reading a tmp_path cache dir).

The load-bearing tests prove empirically that the node needs no
width/height-conditional logic of its own: a resolution-independent encoder
input yields a single real encode (the second resolution is a natural cache
HIT), a resolution-dependent one yields two -- in both cases the decision is
the existing fingerprint/proxy's alone.

nodes.py is loaded the same importlib way as in test_node.py -- never as a
bare ``import nodes``, which collides with ComfyUI's own top-level nodes.py.
"""

import importlib.util
import math
import os
import sys

import pytest
import torch

import comfy.model_management
from comfy_extras.nodes_minimax_h3 import MiniMaxH3ImageToVideo

from minimaxh3_clipcache import last_used as last_used_module
from minimaxh3_clipcache.proxy import MINIMAX_H3_HIDDEN_DIM

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMFYUI_ROOT = os.environ.get(
    "COMFYUI_ROOT",
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
)

CLIP_NAME = "fake_clip.safetensors"
FAKE_FILE_SIZE = 111
FAKE_MTIME_NS = 222
FAKE_CTIME_NS = 333


@pytest.fixture(autouse=True)
def _reset_last_used():
    """minimaxh3_clipcache.last_used is process-wide module state, so without
    a reset each test would see fingerprints recorded by earlier ones."""
    last_used_module._reset_for_tests()
    yield
    last_used_module._reset_for_tests()


def _load_node_module():
    spec = importlib.util.spec_from_file_location(
        "minimaxh3clipcached_nodes_fl2va_dual_under_test", os.path.join(REPO_ROOT, "nodes.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def node_module_with_real_comfy_nodes():
    """Yield a freshly loaded repo nodes.py whose ``import nodes`` resolves to
    ComfyUI's real top-level nodes module, so INPUT_TYPES() (which reads
    ``nodes.MAX_RESOLUTION``) does not raise under pytest. Mirrors the fixture
    in test_clip_name_node.py."""
    saved = sys.modules.get("nodes")
    real_spec = importlib.util.spec_from_file_location(
        "nodes", os.path.join(COMFYUI_ROOT, "nodes.py"))
    real_nodes = importlib.util.module_from_spec(real_spec)
    sys.modules["nodes"] = real_nodes
    try:
        real_spec.loader.exec_module(real_nodes)
        assert hasattr(real_nodes, "MAX_RESOLUTION")
        yield _load_node_module()
    finally:
        if saved is not None:
            sys.modules["nodes"] = saved
        else:
            sys.modules.pop("nodes", None)


class FakeRealClip:
    def __init__(self):
        self.patcher = "fake_patcher"
        self.tokenize_calls = 0
        self.encode_calls = 0

    def tokenize(self, prompt, **kwargs):
        self.tokenize_calls += 1
        return ("real_tokens", prompt, kwargs)

    def encode_from_tokens_scheduled(self, tokens):
        self.encode_calls += 1
        return [[torch.zeros(1, MINIMAX_H3_HIDDEN_DIM), {"pooled_output": None}]]


def _make_unload_counter():
    calls = {"count": 0, "args": []}

    def fake_unload(*args, **kwargs):
        calls["count"] += 1
        calls["args"].append((args, kwargs))

    return fake_unload, calls


def _patch_common(monkeypatch, node_module, tmp_path, fake_execute, real_clip):
    monkeypatch.setattr(MiniMaxH3ImageToVideo, "execute", classmethod(fake_execute))
    monkeypatch.setattr(node_module, "resolve_clip_stat",
                        lambda clip_name: (FAKE_FILE_SIZE, FAKE_MTIME_NS, FAKE_CTIME_NS))
    monkeypatch.setattr(node_module, "build_clip_loader_fn",
                        lambda clip_name: (lambda: real_clip))
    monkeypatch.setattr(node_module, "CACHE_DIR", tmp_path)

    fake_unload, unload_calls = _make_unload_counter()
    monkeypatch.setattr(comfy.model_management, "unload_model_and_clones", fake_unload)
    return unload_calls


# --- regression: the original single-resolution node is untouched ----------

def test_original_fl2va_still_delegates_a_single_encode(monkeypatch, tmp_path):
    """MiniMaxH3CLIPCachedFL2VA.execute() -- now a thin wrapper over
    _execute_fl2va_once() -- must still run exactly one stock encode at the
    one (width, height) it was given, returning the stock (cond, latent)."""
    node_module = _load_node_module()
    real_clip = FakeRealClip()
    seen = []

    def fake_execute(cls, clip, vae, prompt, width, height, length,
                     first_frame=None, last_frame=None):
        seen.append((width, height))
        tokens = clip.tokenize(prompt, images=[])
        cond = clip.encode_from_tokens_scheduled(tokens)
        return (cond, "latent_fake")

    unload_calls = _patch_common(monkeypatch, node_module, tmp_path, fake_execute, real_clip)

    node = node_module.MiniMaxH3CLIPCachedFL2VA()
    cond, latent = node.execute(
        clip_name=CLIP_NAME, vae="fake_vae", prompt="a prompt",
        width=1344, height=768, length=124,
    )

    assert latent == "latent_fake"
    assert torch.equal(cond[0][0], torch.zeros(1, MINIMAX_H3_HIDDEN_DIM))
    assert seen == [(1344, 768)]
    assert real_clip.encode_calls == 1
    assert unload_calls["count"] == 1


# --- the dual node splits the two resolutions, shares everything else ------

def test_dual_runs_both_resolutions_with_shared_inputs(monkeypatch, tmp_path):
    """The dual node must call the shared encode path twice: once with
    (width, height), once with (width2, height2), and every other argument
    (prompt, vae, length, first_frame, last_frame) identical between the two
    calls -- proven by inspecting what reached the stock execute() each
    time."""
    node_module = _load_node_module()
    real_clip = FakeRealClip()
    calls = []

    def fake_execute(cls, clip, vae, prompt, width, height, length,
                     first_frame=None, last_frame=None):
        calls.append(dict(width=width, height=height, length=length, prompt=prompt,
                          vae=vae, first_frame=first_frame, last_frame=last_frame))
        # width/height-dependent encoder input so the two calls are distinct
        img = torch.zeros(1, height, width, 3)
        tokens = clip.tokenize(prompt, images=[img])
        cond = clip.encode_from_tokens_scheduled(tokens)
        return (cond, "latent_{}x{}".format(width, height))

    _patch_common(monkeypatch, node_module, tmp_path, fake_execute, real_clip)

    ff = torch.zeros(1, 8, 8, 3)
    lf = torch.ones(1, 8, 8, 3)
    node = node_module.MiniMaxH3CLIPCachedFL2VADualRes()
    out = node.execute(
        clip_name=CLIP_NAME, vae="fake_vae", prompt="shared prompt",
        width=1344, height=768, width2=1920, height2=1088, length=124,
        first_frame=ff, last_frame=lf,
    )

    assert len(out) == 4
    cond, latent, cond2, latent2 = out
    assert latent == "latent_1344x768"
    assert latent2 == "latent_1920x1088"

    assert len(calls) == 2
    assert (calls[0]["width"], calls[0]["height"]) == (1344, 768)
    assert (calls[1]["width"], calls[1]["height"]) == (1920, 1088)
    for shared in ("length", "prompt", "vae"):
        assert calls[0][shared] == calls[1][shared]
    assert calls[0]["first_frame"] is ff and calls[1]["first_frame"] is ff
    assert calls[0]["last_frame"] is lf and calls[1]["last_frame"] is lf


# --- load-bearing: the cache mechanism alone decides encode count ----------

def test_dual_resolution_independent_input_encodes_once(monkeypatch, tmp_path):
    """When what reaches clip.tokenize() does not depend on width/height
    (t2va with no keyframes), the two resolutions share one fingerprint: the
    real encoder loads exactly once and the second resolution is served from
    cache -- with no width/height-conditional logic in the node."""
    node_module = _load_node_module()
    real_clip = FakeRealClip()

    def fake_execute(cls, clip, vae, prompt, width, height, length,
                     first_frame=None, last_frame=None):
        tokens = clip.tokenize(prompt, images=[])  # deliberately ignores width/height
        cond = clip.encode_from_tokens_scheduled(tokens)
        return (cond, "latent_fake")

    unload_calls = _patch_common(monkeypatch, node_module, tmp_path, fake_execute, real_clip)

    node = node_module.MiniMaxH3CLIPCachedFL2VADualRes()
    node.execute(
        clip_name=CLIP_NAME, vae="fake_vae", prompt="a prompt",
        width=1344, height=768, width2=1920, height2=1088, length=124,
    )

    assert real_clip.encode_calls == 1
    assert unload_calls["count"] == 1


def test_dual_resolution_dependent_input_encodes_twice(monkeypatch, tmp_path):
    """When a keyframe makes the pixels handed to clip.tokenize() differ by
    resolution (the real stock behaviour with first_frame/last_frame), the
    two resolutions have different fingerprints and both encode for real --
    exactly as two separate nodes would, again with no conditional logic in
    the node."""
    node_module = _load_node_module()
    real_clip = FakeRealClip()

    def fake_execute(cls, clip, vae, prompt, width, height, length,
                     first_frame=None, last_frame=None):
        img = torch.zeros(1, height, width, 3)  # simulate _resize to (width, height)
        tokens = clip.tokenize(prompt, images=[img])
        cond = clip.encode_from_tokens_scheduled(tokens)
        return (cond, "latent_fake")

    unload_calls = _patch_common(monkeypatch, node_module, tmp_path, fake_execute, real_clip)

    node = node_module.MiniMaxH3CLIPCachedFL2VADualRes()
    node.execute(
        clip_name=CLIP_NAME, vae="fake_vae", prompt="a prompt",
        width=1344, height=768, width2=1920, height2=1088, length=124,
        first_frame=torch.zeros(1, 8, 8, 3),
    )

    assert real_clip.encode_calls == 2
    assert unload_calls["count"] == 2


def test_dual_same_resolution_twice_still_encodes_once(monkeypatch, tmp_path):
    """Even with a resolution-dependent encoder input, asking for the same
    resolution twice (width2==width, height2==height) is one fingerprint and
    one real encode -- the second call is a cache HIT."""
    node_module = _load_node_module()
    real_clip = FakeRealClip()

    def fake_execute(cls, clip, vae, prompt, width, height, length,
                     first_frame=None, last_frame=None):
        img = torch.zeros(1, height, width, 3)
        tokens = clip.tokenize(prompt, images=[img])
        cond = clip.encode_from_tokens_scheduled(tokens)
        return (cond, "latent_fake")

    unload_calls = _patch_common(monkeypatch, node_module, tmp_path, fake_execute, real_clip)

    node = node_module.MiniMaxH3CLIPCachedFL2VADualRes()
    node.execute(
        clip_name=CLIP_NAME, vae="fake_vae", prompt="a prompt",
        width=1344, height=768, width2=1344, height2=768, length=124,
        first_frame=torch.zeros(1, 8, 8, 3),
    )

    assert real_clip.encode_calls == 1
    assert unload_calls["count"] == 1


# --- schema / registration ------------------------------------------------

def test_dual_return_spec_and_category():
    node_module = _load_node_module()
    cls = node_module.MiniMaxH3CLIPCachedFL2VADualRes
    assert cls.RETURN_TYPES == ("CONDITIONING", "LATENT", "CONDITIONING", "LATENT")
    assert cls.RETURN_NAMES == ("positive", "latent", "positive_2", "latent_2")
    assert cls.CATEGORY == "model/conditioning/minimax/cached"
    assert cls.FUNCTION == "execute"


def test_dual_is_changed_is_the_shared_common_body():
    """IS_CHANGED is unchanged from the single-resolution node: refresh -> a
    fresh NaN every call, auto -> a stable self-equal value."""
    node_module = _load_node_module()
    cls = node_module.MiniMaxH3CLIPCachedFL2VADualRes

    first = cls.IS_CHANGED(cache_mode="refresh", clip_name=CLIP_NAME, prompt="p")
    second = cls.IS_CHANGED(cache_mode="refresh", clip_name=CLIP_NAME, prompt="p")
    assert isinstance(first, float) and math.isnan(first)
    assert isinstance(second, float) and math.isnan(second)
    assert not (first == second)

    assert cls.IS_CHANGED(cache_mode="auto", prompt="p") == \
           cls.IS_CHANGED(cache_mode="auto", prompt="p")


def test_dual_input_types_adds_second_resolution_only(node_module_with_real_comfy_nodes):
    m = node_module_with_real_comfy_nodes
    req = m.MiniMaxH3CLIPCachedFL2VADualRes.INPUT_TYPES()["required"]
    fl2va_req = m.MiniMaxH3CLIPCachedFL2VA.INPUT_TYPES()["required"]

    assert "width2" in req and "height2" in req
    # width2/height2 carry the same numeric constraints as width/height
    for k in ("min", "max", "step", "default"):
        assert req["width2"][1][k] == fl2va_req["width"][1][k]
        assert req["height2"][1][k] == fl2va_req["height"][1][k]
    # the shared inputs are otherwise identical to the single-resolution node
    assert req["clip_name"] == fl2va_req["clip_name"]
    assert req["prompt"] == fl2va_req["prompt"]
    assert req["length"] == fl2va_req["length"]

    opt = m.MiniMaxH3CLIPCachedFL2VADualRes.INPUT_TYPES()["optional"]
    assert set(opt) == {"first_frame", "last_frame", "cache_mode"}


def test_dual_registered_in_node_mappings():
    spec = importlib.util.spec_from_file_location(
        "minimaxh3clipcached_package_fl2va_dual_test", os.path.join(REPO_ROOT, "__init__.py"))
    package = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = package
    spec.loader.exec_module(package)

    assert package.NODE_CLASS_MAPPINGS["MiniMaxH3CLIPCachedFL2VADualRes"].__name__ == \
        "MiniMaxH3CLIPCachedFL2VADualRes"
    assert package.NODE_DISPLAY_NAME_MAPPINGS["MiniMaxH3CLIPCachedFL2VADualRes"] == \
        "MiniMax H3 CLIP-Cached FL2VA (Dual Resolution)"
