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
import json
import logging
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
    (width, height), once with (width_upscale, height_upscale), and every other argument
    (prompt, vae, length, first_frame, last_frame) identical between the two
    calls -- proven by inspecting what reached the stock execute() each
    time."""
    node_module = _load_node_module()

    class ResolutionAwareClip(FakeRealClip):
        """Test-local override of the encoder: the returned conditioning
        carries the (width, height) it was asked to encode, recovered from
        the image tensor's shape in the tokens. The module-level
        FakeRealClip returns one constant value for every resolution, so a
        swapped ``return (cond_upscale, latent, cond)`` in nodes.py would
        slip past a bare ``cond2 is not None`` check. This subclass is
        confined to this one test and does not touch the shared class."""

        def encode_from_tokens_scheduled(self, tokens):
            self.encode_calls += 1
            _marker, _prompt, kwargs = tokens
            img = kwargs["images"][0]
            height, width = int(img.shape[1]), int(img.shape[2])
            main = torch.zeros(1, MINIMAX_H3_HIDDEN_DIM)
            main[0, 0] = float(width)
            main[0, 1] = float(height)
            return [[main, {"pooled_output": None}]]

    real_clip = ResolutionAwareClip()
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
        width=1344, height=768, width_upscale=1920, height_upscale=1088, length=124,
        first_frame=ff, last_frame=lf,
    )

    assert len(out) == 3
    cond, latent, cond2 = out
    assert latent == "latent_1344x768"
    # cond (base) and cond2 (upscale) must be distinguishable AND each must
    # carry its own resolution -- not merely non-None. This fails if the two
    # CONDITIONING slots in the node's return tuple are swapped.
    assert not torch.equal(cond[0][0], cond2[0][0])
    assert (cond[0][0][0, 0].item(), cond[0][0][0, 1].item()) == (1344.0, 768.0)
    assert (cond2[0][0][0, 0].item(), cond2[0][0][0, 1].item()) == (1920.0, 1088.0)

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
        width=1344, height=768, width_upscale=1920, height_upscale=1088, length=124,
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
        width=1344, height=768, width_upscale=1920, height_upscale=1088, length=124,
        first_frame=torch.zeros(1, 8, 8, 3),
    )

    assert real_clip.encode_calls == 2
    assert unload_calls["count"] == 2


def test_dual_resolution_dependent_input_cross_links_the_two_verbose_entries(monkeypatch, tmp_path):
    """When the two resolutions land on two distinct fingerprints, each
    entry's verbose sidecar must carry the other's fingerprint and pixel size
    (paired_fingerprint / paired_width / paired_height), so the Cache Manager
    can later show them as one row instead of duplicating the prompt."""
    node_module = _load_node_module()
    real_clip = FakeRealClip()

    def fake_execute(cls, clip, vae, prompt, width, height, length,
                     first_frame=None, last_frame=None):
        img = torch.zeros(1, height, width, 3)  # simulate _resize to (width, height)
        tokens = clip.tokenize(prompt, images=[img])
        cond = clip.encode_from_tokens_scheduled(tokens)
        return (cond, "latent_fake")

    _patch_common(monkeypatch, node_module, tmp_path, fake_execute, real_clip)

    node = node_module.MiniMaxH3CLIPCachedFL2VADualRes()
    node.execute(
        clip_name=CLIP_NAME, vae="fake_vae", prompt="a prompt",
        width=1344, height=768, width_upscale=1920, height_upscale=1088, length=124,
        first_frame=torch.zeros(1, 8, 8, 3),
    )

    sidecars = sorted(tmp_path.glob("*.verbose.json"))
    assert len(sidecars) == 2
    v_a = json.loads(sidecars[0].read_bytes())
    v_b = json.loads(sidecars[1].read_bytes())

    assert v_a["system"]["paired_fingerprint"] == v_b["fingerprint"]
    assert v_b["system"]["paired_fingerprint"] == v_a["fingerprint"]
    assert (v_a["system"]["paired_width"], v_a["system"]["paired_height"]) == \
           (v_b["system"]["width"], v_b["system"]["height"])
    assert (v_b["system"]["paired_width"], v_b["system"]["paired_height"]) == \
           (v_a["system"]["width"], v_a["system"]["height"])
    assert {(v_a["system"]["width"], v_a["system"]["height"]),
            (v_b["system"]["width"], v_b["system"]["height"])} == {(1344, 768), (1920, 1088)}
    # the base-resolution entry (width / height) is stamped False, the
    # upscale-resolution entry (width_upscale / height_upscale) True
    by_res = {(v["system"]["width"], v["system"]["height"]): v["system"] for v in (v_a, v_b)}
    assert by_res[(1344, 768)]["is_upscale_target"] is False
    assert by_res[(1920, 1088)]["is_upscale_target"] is True


def test_dual_resolution_independent_input_writes_no_pairing(monkeypatch, tmp_path):
    """When both resolutions share one fingerprint there is only one cache
    entry and nothing to pair -- its sidecar carries no paired_fingerprint."""
    node_module = _load_node_module()
    real_clip = FakeRealClip()

    def fake_execute(cls, clip, vae, prompt, width, height, length,
                     first_frame=None, last_frame=None):
        tokens = clip.tokenize(prompt, images=[])  # ignores width/height
        cond = clip.encode_from_tokens_scheduled(tokens)
        return (cond, "latent_fake")

    _patch_common(monkeypatch, node_module, tmp_path, fake_execute, real_clip)

    node = node_module.MiniMaxH3CLIPCachedFL2VADualRes()
    node.execute(
        clip_name=CLIP_NAME, vae="fake_vae", prompt="a prompt",
        width=1344, height=768, width_upscale=1920, height_upscale=1088, length=124,
    )

    sidecars = list(tmp_path.glob("*.verbose.json"))
    assert len(sidecars) == 1
    assert "paired_fingerprint" not in json.loads(sidecars[0].read_bytes())["system"]


def test_dual_same_resolution_twice_still_encodes_once(monkeypatch, tmp_path):
    """Even with a resolution-dependent encoder input, asking for the same
    resolution twice (width_upscale==width, height_upscale==height) is one fingerprint and
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
        width=1344, height=768, width_upscale=1344, height_upscale=768, length=124,
        first_frame=torch.zeros(1, 8, 8, 3),
    )

    assert real_clip.encode_calls == 1
    assert unload_calls["count"] == 1


# --- generate_upscale_cond: skip the second encode entirely --------------

def test_dual_generate_upscale_cond_false_skips_the_second_encode(monkeypatch, tmp_path):
    """With generate_upscale_cond=False the upscale-resolution encode must
    not run at all: on a resolution-dependent input (which normally forces
    two real encodes) the real encoder is loaded exactly once, and
    positive_upscale comes back as None."""
    node_module = _load_node_module()
    real_clip = FakeRealClip()

    def fake_execute(cls, clip, vae, prompt, width, height, length,
                     first_frame=None, last_frame=None):
        img = torch.zeros(1, height, width, 3)  # resolution-dependent encoder input
        tokens = clip.tokenize(prompt, images=[img])
        cond = clip.encode_from_tokens_scheduled(tokens)
        return (cond, "latent_{}x{}".format(width, height))

    unload_calls = _patch_common(monkeypatch, node_module, tmp_path, fake_execute, real_clip)

    node = node_module.MiniMaxH3CLIPCachedFL2VADualRes()
    out = node.execute(
        clip_name=CLIP_NAME, vae="fake_vae", prompt="a prompt",
        width=1344, height=768, width_upscale=1920, height_upscale=1088, length=124,
        first_frame=torch.zeros(1, 8, 8, 3), generate_upscale_cond=False,
    )

    assert len(out) == 3
    cond, latent, cond_upscale = out
    assert latent == "latent_1344x768"
    assert cond_upscale is None
    assert real_clip.encode_calls == 1
    assert unload_calls["count"] == 1


def test_dual_generate_upscale_cond_false_does_not_pair(monkeypatch, tmp_path):
    """With generate_upscale_cond=False there is no second fingerprint, so
    _pair_verbose_entries() is never called and only the base-resolution
    sidecar is written (no paired_fingerprint on it)."""
    node_module = _load_node_module()
    real_clip = FakeRealClip()

    pair_calls = {"count": 0}

    def spy_pair(*args, **kwargs):
        pair_calls["count"] += 1

    monkeypatch.setattr(node_module, "_pair_verbose_entries", spy_pair)

    def fake_execute(cls, clip, vae, prompt, width, height, length,
                     first_frame=None, last_frame=None):
        img = torch.zeros(1, height, width, 3)
        tokens = clip.tokenize(prompt, images=[img])
        cond = clip.encode_from_tokens_scheduled(tokens)
        return (cond, "latent_fake")

    _patch_common(monkeypatch, node_module, tmp_path, fake_execute, real_clip)

    node = node_module.MiniMaxH3CLIPCachedFL2VADualRes()
    node.execute(
        clip_name=CLIP_NAME, vae="fake_vae", prompt="a prompt",
        width=1344, height=768, width_upscale=1920, height_upscale=1088, length=124,
        first_frame=torch.zeros(1, 8, 8, 3), generate_upscale_cond=False,
    )

    assert pair_calls["count"] == 0
    sidecars = list(tmp_path.glob("*.verbose.json"))
    assert len(sidecars) == 1
    assert "paired_fingerprint" not in json.loads(sidecars[0].read_bytes())["system"]


def test_dual_generate_upscale_cond_true_is_the_default(monkeypatch, tmp_path):
    """Passing generate_upscale_cond=True explicitly is identical to omitting
    it: the upscale encode runs and the two entries are paired, exactly as
    before this switch existed."""
    node_module = _load_node_module()
    real_clip = FakeRealClip()

    pair_calls = {"count": 0}
    monkeypatch.setattr(
        node_module, "_pair_verbose_entries",
        lambda *a, **k: pair_calls.__setitem__("count", pair_calls["count"] + 1))

    def fake_execute(cls, clip, vae, prompt, width, height, length,
                     first_frame=None, last_frame=None):
        img = torch.zeros(1, height, width, 3)
        tokens = clip.tokenize(prompt, images=[img])
        cond = clip.encode_from_tokens_scheduled(tokens)
        return (cond, "latent_fake")

    _patch_common(monkeypatch, node_module, tmp_path, fake_execute, real_clip)

    node = node_module.MiniMaxH3CLIPCachedFL2VADualRes()
    out = node.execute(
        clip_name=CLIP_NAME, vae="fake_vae", prompt="a prompt",
        width=1344, height=768, width_upscale=1920, height_upscale=1088, length=124,
        first_frame=torch.zeros(1, 8, 8, 3), generate_upscale_cond=True,
    )

    assert out[2] is not None
    assert real_clip.encode_calls == 2
    assert pair_calls["count"] == 1


def test_dual_generate_upscale_cond_false_logs_one_info_line(monkeypatch, tmp_path, caplog):
    """generate_upscale_cond=False emits exactly one INFO record tagged
    [UPSCALE COND SKIPPED], carrying the first 12 chars of the base-
    resolution fingerprint. With the default (True) that line is absent."""
    node_module = _load_node_module()
    real_clip = FakeRealClip()

    def fake_execute(cls, clip, vae, prompt, width, height, length,
                     first_frame=None, last_frame=None):
        img = torch.zeros(1, height, width, 3)
        tokens = clip.tokenize(prompt, images=[img])
        cond = clip.encode_from_tokens_scheduled(tokens)
        return (cond, "latent_fake")

    _patch_common(monkeypatch, node_module, tmp_path, fake_execute, real_clip)

    node = node_module.MiniMaxH3CLIPCachedFL2VADualRes()
    kwargs = dict(
        clip_name=CLIP_NAME, vae="fake_vae", prompt="a prompt",
        width=1344, height=768, width_upscale=1920, height_upscale=1088, length=124,
        first_frame=torch.zeros(1, 8, 8, 3),
    )

    with caplog.at_level(logging.INFO):
        node.execute(generate_upscale_cond=False, **kwargs)

    skipped = [r for r in caplog.records
               if r.levelno == logging.INFO and "[UPSCALE COND SKIPPED]" in r.getMessage()]
    assert len(skipped) == 1
    fp_prefix = json.loads(
        next(iter(tmp_path.glob("*.verbose.json"))).read_bytes())["fingerprint"][:12]
    assert fp_prefix in skipped[0].getMessage()

    caplog.clear()
    with caplog.at_level(logging.INFO):
        node.execute(generate_upscale_cond=True, **kwargs)
    assert not any("[UPSCALE COND SKIPPED]" in r.getMessage() for r in caplog.records)


# --- schema / registration ------------------------------------------------

def test_dual_return_spec_and_category():
    node_module = _load_node_module()
    cls = node_module.MiniMaxH3CLIPCachedFL2VADualRes
    assert cls.RETURN_TYPES == ("CONDITIONING", "LATENT", "CONDITIONING")
    assert cls.RETURN_NAMES == ("positive", "latent", "positive_upscale")
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

    assert "width_upscale" in req and "height_upscale" in req
    # width_upscale/height_upscale carry the same numeric constraints as width/height
    for k in ("min", "max", "step", "default"):
        assert req["width_upscale"][1][k] == fl2va_req["width"][1][k]
        assert req["height_upscale"][1][k] == fl2va_req["height"][1][k]
    # the shared inputs are otherwise identical to the single-resolution node
    assert req["clip_name"] == fl2va_req["clip_name"]
    assert req["prompt"] == fl2va_req["prompt"]
    assert req["length"] == fl2va_req["length"]

    opt = m.MiniMaxH3CLIPCachedFL2VADualRes.INPUT_TYPES()["optional"]
    assert set(opt) == {"first_frame", "last_frame", "generate_upscale_cond", "cache_mode"}
    assert opt["generate_upscale_cond"][0] == "BOOLEAN"
    assert opt["generate_upscale_cond"][1]["default"] is True
    assert "tooltip" in opt["generate_upscale_cond"][1]


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
