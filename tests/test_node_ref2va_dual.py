"""Unit tests for MiniMaxH3CLIPCachedRef2VADualRes -- the two-resolution
sibling of MiniMaxH3CLIPCachedRef2VA. Same no-GPU harness as
test_node_ref2va.py (FakeRealClip, _patch_common, a monkeypatched
MiniMaxH3ReferenceToVideo.execute, the real CachedClipProxy writing/reading a
tmp_path cache dir).

The load-bearing tests prove empirically that the node needs no
width/height-conditional logic of its own: a resolution-independent encoder
input (no references / ref_image_size="max") yields a single real encode,
a resolution-dependent one ("match" sizing on a real reference image)
yields two -- in both cases the decision is the existing fingerprint/proxy's
alone.
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
from comfy_extras.nodes_minimax_h3 import MiniMaxH3ReferenceToVideo

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
    last_used_module._reset_for_tests()
    yield
    last_used_module._reset_for_tests()


def _load_node_module():
    spec = importlib.util.spec_from_file_location(
        "minimaxh3clipcached_nodes_ref2va_dual_under_test", os.path.join(REPO_ROOT, "nodes.py"))
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
    monkeypatch.setattr(MiniMaxH3ReferenceToVideo, "execute", classmethod(fake_execute))
    monkeypatch.setattr(node_module, "resolve_clip_stat",
                        lambda clip_name: (FAKE_FILE_SIZE, FAKE_MTIME_NS, FAKE_CTIME_NS))
    monkeypatch.setattr(node_module, "build_clip_loader_fn",
                        lambda clip_name: (lambda: real_clip))
    monkeypatch.setattr(node_module, "CACHE_DIR", tmp_path)

    fake_unload, unload_calls = _make_unload_counter()
    monkeypatch.setattr(comfy.model_management, "unload_model_and_clones", fake_unload)
    return unload_calls


def _execute(node, **overrides):
    kwargs = dict(
        clip_name=CLIP_NAME, vae="fake_vae", audio_vae="fake_audio_vae",
        prompt="a ref2va prompt", width=1344, height=768,
        width_upscale=1920, height_upscale=1088, length=124,
    )
    kwargs.update(overrides)
    return node.execute(**kwargs)


# --- regression: the original single-resolution node is untouched ----------

def test_original_ref2va_still_delegates_a_single_encode(monkeypatch, tmp_path):
    """MiniMaxH3CLIPCachedRef2VA.execute() -- now a thin wrapper that folds
    its flat slots into dicts and calls _execute_ref2va_once() once -- must
    still run exactly one stock encode and pass the slot dicts through."""
    node_module = _load_node_module()
    real_clip = FakeRealClip()
    seen = {}

    def fake_execute(cls, clip, vae, audio_vae, prompt, width, height, length,
                     ref_image_size="match", ref_images=None, ref_videos=None,
                     ref_video_audios=None, ref_audios=None):
        seen.update(ref_images=ref_images, ref_videos=ref_videos, wh=(width, height))
        tokens = clip.tokenize(prompt, minimax_ref_items=[])
        cond = clip.encode_from_tokens_scheduled(tokens)
        return (cond, "latent_fake")

    unload_calls = _patch_common(monkeypatch, node_module, tmp_path, fake_execute, real_clip)

    img = torch.zeros(1, 4, 4, 3)
    node = node_module.MiniMaxH3CLIPCachedRef2VA()
    cond, latent = node.execute(
        clip_name=CLIP_NAME, vae="fake_vae", audio_vae="fake_audio_vae",
        prompt="p", width=1344, height=768, length=124, ref_image_2=img,
    )

    assert latent == "latent_fake"
    assert list(seen["ref_images"].keys()) == ["ref_image_2"]
    assert seen["wh"] == (1344, 768)
    assert real_clip.encode_calls == 1
    assert unload_calls["count"] == 1


# --- the dual node splits the two resolutions, shares everything else ------

def test_dual_runs_both_resolutions_with_shared_inputs(monkeypatch, tmp_path):
    node_module = _load_node_module()

    class ResolutionAwareClip(FakeRealClip):
        """Test-local override of the encoder: the returned conditioning
        carries the (width, height) it was asked to encode, recovered from
        the reference item tensor's shape in the tokens. The module-level
        FakeRealClip returns one constant value for every resolution, so a
        swapped ``return (cond_upscale, latent, cond)`` in nodes.py would
        slip past a bare ``cond2 is not None`` check. This subclass is
        confined to this one test and does not touch the shared class."""

        def encode_from_tokens_scheduled(self, tokens):
            self.encode_calls += 1
            _marker, _prompt, kwargs = tokens
            data = kwargs["minimax_ref_items"][0]["data"]
            height, width = int(data.shape[1]) * 16, int(data.shape[2]) * 16
            main = torch.zeros(1, MINIMAX_H3_HIDDEN_DIM)
            main[0, 0] = float(width)
            main[0, 1] = float(height)
            return [[main, {"pooled_output": None}]]

    real_clip = ResolutionAwareClip()
    calls = []

    def fake_execute(cls, clip, vae, audio_vae, prompt, width, height, length,
                     ref_image_size="match", ref_images=None, ref_videos=None,
                     ref_video_audios=None, ref_audios=None):
        calls.append(dict(width=width, height=height, length=length, prompt=prompt,
                          vae=vae, audio_vae=audio_vae, ref_image_size=ref_image_size,
                          ref_images=ref_images, ref_videos=ref_videos))
        # width/height-dependent encoder input so the two calls are distinct
        item = {"type": "image", "data": torch.zeros(1, height // 16, width // 16, 3)}
        tokens = clip.tokenize(prompt, minimax_ref_items=[item])
        cond = clip.encode_from_tokens_scheduled(tokens)
        return (cond, "latent_{}x{}".format(width, height))

    _patch_common(monkeypatch, node_module, tmp_path, fake_execute, real_clip)

    img = torch.zeros(1, 4, 4, 3)
    node = node_module.MiniMaxH3CLIPCachedRef2VADualRes()
    out = _execute(node, ref_image_0=img, ref_image_size="match")

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
    for shared in ("length", "prompt", "vae", "audio_vae", "ref_image_size"):
        assert calls[0][shared] == calls[1][shared]
    # the reference dicts are the very same objects, built once by the wrapper
    assert calls[0]["ref_images"] is calls[1]["ref_images"]
    assert list(calls[0]["ref_images"].keys()) == ["ref_image_0"]


# --- load-bearing: the cache mechanism alone decides encode count ----------

def test_dual_resolution_independent_input_encodes_once(monkeypatch, tmp_path):
    """No references reach the encoder (or ref_image_size="max"): the
    encoder input is resolution-independent, so the two resolutions share one
    fingerprint and the real encoder loads exactly once."""
    node_module = _load_node_module()
    real_clip = FakeRealClip()

    def fake_execute(cls, clip, vae, audio_vae, prompt, width, height, length,
                     ref_image_size="match", ref_images=None, ref_videos=None,
                     ref_video_audios=None, ref_audios=None):
        tokens = clip.tokenize(prompt, minimax_ref_items=[])  # ignores width/height
        cond = clip.encode_from_tokens_scheduled(tokens)
        return (cond, "latent_fake")

    unload_calls = _patch_common(monkeypatch, node_module, tmp_path, fake_execute, real_clip)

    node = node_module.MiniMaxH3CLIPCachedRef2VADualRes()
    _execute(node)

    assert real_clip.encode_calls == 1
    assert unload_calls["count"] == 1


def test_dual_resolution_dependent_input_encodes_twice(monkeypatch, tmp_path):
    """A real reference image under ref_image_size="match" is scaled to the
    generation's pixel area, so the pixels handed to the encoder differ by
    resolution: two fingerprints, two real encodes -- exactly as two
    separate nodes would."""
    node_module = _load_node_module()
    real_clip = FakeRealClip()

    def fake_execute(cls, clip, vae, audio_vae, prompt, width, height, length,
                     ref_image_size="match", ref_images=None, ref_videos=None,
                     ref_video_audios=None, ref_audios=None):
        # simulate 'match' sizing: a ref tensor whose shape depends on width*height
        item = {"type": "image", "data": torch.zeros(1, height // 16, width // 16, 3)}
        tokens = clip.tokenize(prompt, minimax_ref_items=[item])
        cond = clip.encode_from_tokens_scheduled(tokens)
        return (cond, "latent_fake")

    unload_calls = _patch_common(monkeypatch, node_module, tmp_path, fake_execute, real_clip)

    node = node_module.MiniMaxH3CLIPCachedRef2VADualRes()
    _execute(node, ref_image_0=torch.zeros(1, 4, 4, 3), ref_image_size="match")

    assert real_clip.encode_calls == 2
    assert unload_calls["count"] == 2


def test_dual_same_resolution_twice_still_encodes_once(monkeypatch, tmp_path):
    node_module = _load_node_module()
    real_clip = FakeRealClip()

    def fake_execute(cls, clip, vae, audio_vae, prompt, width, height, length,
                     ref_image_size="match", ref_images=None, ref_videos=None,
                     ref_video_audios=None, ref_audios=None):
        item = {"type": "image", "data": torch.zeros(1, height // 16, width // 16, 3)}
        tokens = clip.tokenize(prompt, minimax_ref_items=[item])
        cond = clip.encode_from_tokens_scheduled(tokens)
        return (cond, "latent_fake")

    unload_calls = _patch_common(monkeypatch, node_module, tmp_path, fake_execute, real_clip)

    node = node_module.MiniMaxH3CLIPCachedRef2VADualRes()
    _execute(node, width=1344, height=768, width_upscale=1344, height_upscale=768,
             ref_image_0=torch.zeros(1, 4, 4, 3))

    assert real_clip.encode_calls == 1
    assert unload_calls["count"] == 1


# --- dual-resolution Cache Manager entry pairing --------------------------

def test_dual_resolution_dependent_input_cross_links_the_two_verbose_entries(monkeypatch, tmp_path):
    """A real reference under ref_image_size="match" makes the encoder input
    differ by resolution: two fingerprints, two cache entries. Each entry's
    verbose sidecar must carry the other's fingerprint and pixel size."""
    node_module = _load_node_module()
    real_clip = FakeRealClip()

    def fake_execute(cls, clip, vae, audio_vae, prompt, width, height, length,
                     ref_image_size="match", ref_images=None, ref_videos=None,
                     ref_video_audios=None, ref_audios=None):
        item = {"type": "image", "data": torch.zeros(1, height // 16, width // 16, 3)}
        tokens = clip.tokenize(prompt, minimax_ref_items=[item])
        cond = clip.encode_from_tokens_scheduled(tokens)
        return (cond, "latent_fake")

    _patch_common(monkeypatch, node_module, tmp_path, fake_execute, real_clip)

    node = node_module.MiniMaxH3CLIPCachedRef2VADualRes()
    _execute(node, ref_image_0=torch.zeros(1, 4, 4, 3), ref_image_size="match")

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
    """No references reach the encoder: both resolutions share one
    fingerprint, one cache entry, nothing to pair -- no paired_fingerprint.

    The single sidecar is stamped with the BASE resolution (width/height),
    not the upscale one: the upscale pass is a cache HIT that would otherwise
    move the informational generation-size trio forward to side B, so
    _pair_verbose_entries() finalizes it back on the shared-fingerprint
    branch -- the same helper as FL2VA DualRes."""
    node_module = _load_node_module()
    real_clip = FakeRealClip()

    def fake_execute(cls, clip, vae, audio_vae, prompt, width, height, length,
                     ref_image_size="match", ref_images=None, ref_videos=None,
                     ref_video_audios=None, ref_audios=None):
        tokens = clip.tokenize(prompt, minimax_ref_items=[])  # ignores width/height
        cond = clip.encode_from_tokens_scheduled(tokens)
        return (cond, "latent_fake")

    _patch_common(monkeypatch, node_module, tmp_path, fake_execute, real_clip)

    node = node_module.MiniMaxH3CLIPCachedRef2VADualRes()
    _execute(node)

    sidecars = list(tmp_path.glob("*.verbose.json"))
    assert len(sidecars) == 1
    system = json.loads(sidecars[0].read_bytes())["system"]
    assert "paired_fingerprint" not in system
    assert (system["width"], system["height"]) == (1344, 768)
    assert system["megapixels"] == 1.03


# --- generate_upscale_cond: skip the second encode entirely --------------

def test_dual_generate_upscale_cond_false_skips_the_second_encode(monkeypatch, tmp_path):
    """With generate_upscale_cond=False the upscale-resolution encode must
    not run at all: on a resolution-dependent input (a real reference under
    ref_image_size="match", which normally forces two real encodes) the real
    encoder is loaded exactly once, and positive_upscale comes back as
    None."""
    node_module = _load_node_module()
    real_clip = FakeRealClip()

    def fake_execute(cls, clip, vae, audio_vae, prompt, width, height, length,
                     ref_image_size="match", ref_images=None, ref_videos=None,
                     ref_video_audios=None, ref_audios=None):
        item = {"type": "image", "data": torch.zeros(1, height // 16, width // 16, 3)}
        tokens = clip.tokenize(prompt, minimax_ref_items=[item])
        cond = clip.encode_from_tokens_scheduled(tokens)
        return (cond, "latent_{}x{}".format(width, height))

    unload_calls = _patch_common(monkeypatch, node_module, tmp_path, fake_execute, real_clip)

    node = node_module.MiniMaxH3CLIPCachedRef2VADualRes()
    out = _execute(node, ref_image_0=torch.zeros(1, 4, 4, 3), ref_image_size="match",
                   generate_upscale_cond=False)

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

    def fake_execute(cls, clip, vae, audio_vae, prompt, width, height, length,
                     ref_image_size="match", ref_images=None, ref_videos=None,
                     ref_video_audios=None, ref_audios=None):
        item = {"type": "image", "data": torch.zeros(1, height // 16, width // 16, 3)}
        tokens = clip.tokenize(prompt, minimax_ref_items=[item])
        cond = clip.encode_from_tokens_scheduled(tokens)
        return (cond, "latent_fake")

    _patch_common(monkeypatch, node_module, tmp_path, fake_execute, real_clip)

    node = node_module.MiniMaxH3CLIPCachedRef2VADualRes()
    _execute(node, ref_image_0=torch.zeros(1, 4, 4, 3), ref_image_size="match",
             generate_upscale_cond=False)

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

    def fake_execute(cls, clip, vae, audio_vae, prompt, width, height, length,
                     ref_image_size="match", ref_images=None, ref_videos=None,
                     ref_video_audios=None, ref_audios=None):
        item = {"type": "image", "data": torch.zeros(1, height // 16, width // 16, 3)}
        tokens = clip.tokenize(prompt, minimax_ref_items=[item])
        cond = clip.encode_from_tokens_scheduled(tokens)
        return (cond, "latent_fake")

    _patch_common(monkeypatch, node_module, tmp_path, fake_execute, real_clip)

    node = node_module.MiniMaxH3CLIPCachedRef2VADualRes()
    out = _execute(node, ref_image_0=torch.zeros(1, 4, 4, 3), ref_image_size="match",
                   generate_upscale_cond=True)

    assert out[2] is not None
    assert real_clip.encode_calls == 2
    assert pair_calls["count"] == 1


def test_dual_generate_upscale_cond_false_logs_one_info_line(monkeypatch, tmp_path, caplog):
    """generate_upscale_cond=False emits exactly one INFO record tagged
    [UPSCALE COND SKIPPED], carrying the first 12 chars of the base-
    resolution fingerprint. With the default (True) that line is absent."""
    node_module = _load_node_module()
    real_clip = FakeRealClip()

    def fake_execute(cls, clip, vae, audio_vae, prompt, width, height, length,
                     ref_image_size="match", ref_images=None, ref_videos=None,
                     ref_video_audios=None, ref_audios=None):
        item = {"type": "image", "data": torch.zeros(1, height // 16, width // 16, 3)}
        tokens = clip.tokenize(prompt, minimax_ref_items=[item])
        cond = clip.encode_from_tokens_scheduled(tokens)
        return (cond, "latent_fake")

    _patch_common(monkeypatch, node_module, tmp_path, fake_execute, real_clip)

    node = node_module.MiniMaxH3CLIPCachedRef2VADualRes()

    with caplog.at_level(logging.INFO):
        _execute(node, ref_image_0=torch.zeros(1, 4, 4, 3), ref_image_size="match",
                 generate_upscale_cond=False)

    skipped = [r for r in caplog.records
               if r.levelno == logging.INFO and "[UPSCALE COND SKIPPED]" in r.getMessage()]
    assert len(skipped) == 1
    fp_prefix = json.loads(
        next(iter(tmp_path.glob("*.verbose.json"))).read_bytes())["fingerprint"][:12]
    assert fp_prefix in skipped[0].getMessage()

    caplog.clear()
    with caplog.at_level(logging.INFO):
        _execute(node, ref_image_0=torch.zeros(1, 4, 4, 3), ref_image_size="match",
                 generate_upscale_cond=True)
    assert not any("[UPSCALE COND SKIPPED]" in r.getMessage() for r in caplog.records)


# --- schema / registration ------------------------------------------------

def test_dual_return_spec_and_category():
    node_module = _load_node_module()
    cls = node_module.MiniMaxH3CLIPCachedRef2VADualRes
    assert cls.RETURN_TYPES == ("CONDITIONING", "LATENT", "CONDITIONING")
    assert cls.RETURN_NAMES == ("positive", "latent", "positive_upscale")
    assert cls.CATEGORY == "model/conditioning/minimax/cached"
    assert cls.FUNCTION == "execute"


def test_dual_is_changed_is_the_shared_common_body():
    node_module = _load_node_module()
    cls = node_module.MiniMaxH3CLIPCachedRef2VADualRes

    first = cls.IS_CHANGED(cache_mode="refresh", clip_name=CLIP_NAME, prompt="p")
    second = cls.IS_CHANGED(cache_mode="refresh", clip_name=CLIP_NAME, prompt="p")
    assert isinstance(first, float) and math.isnan(first)
    assert isinstance(second, float) and math.isnan(second)
    assert not (first == second)

    assert cls.IS_CHANGED(cache_mode="auto", prompt="p") == \
           cls.IS_CHANGED(cache_mode="auto", prompt="p")


def test_dual_input_types_adds_second_resolution_only(node_module_with_real_comfy_nodes):
    m = node_module_with_real_comfy_nodes
    req = m.MiniMaxH3CLIPCachedRef2VADualRes.INPUT_TYPES()["required"]
    ref2va_req = m.MiniMaxH3CLIPCachedRef2VA.INPUT_TYPES()["required"]

    assert "width_upscale" in req and "height_upscale" in req
    for k in ("min", "max", "step", "default"):
        assert req["width_upscale"][1][k] == ref2va_req["width"][1][k]
        assert req["height_upscale"][1][k] == ref2va_req["height"][1][k]
    assert req["clip_name"] == ref2va_req["clip_name"]
    assert req["audio_vae"] == ref2va_req["audio_vae"]
    assert req["ref_image_size"] == ref2va_req["ref_image_size"]

    # optional block is the same fixed ref_* slots as the single-resolution
    # node; cache_mode is present but its tooltip is tailored ("Applies to
    # both resolutions").
    dual_opt = m.MiniMaxH3CLIPCachedRef2VADualRes.INPUT_TYPES()["optional"]
    single_opt = m.MiniMaxH3CLIPCachedRef2VA.INPUT_TYPES()["optional"]
    # the dual node adds one optional over the single-resolution node: the
    # generate_upscale_cond switch that gates the second encode.
    assert set(dual_opt) == set(single_opt) | {"generate_upscale_cond"}
    ref_slots = [k for k in single_opt if k != "cache_mode"]
    assert {k: dual_opt[k] for k in ref_slots} == {k: single_opt[k] for k in ref_slots}
    assert dual_opt["cache_mode"][0] == single_opt["cache_mode"][0] == ["auto", "refresh"]
    assert dual_opt["generate_upscale_cond"][0] == "BOOLEAN"
    assert dual_opt["generate_upscale_cond"][1]["default"] is True
    assert "tooltip" in dual_opt["generate_upscale_cond"][1]


def test_dual_registered_in_node_mappings():
    spec = importlib.util.spec_from_file_location(
        "minimaxh3clipcached_package_ref2va_dual_test", os.path.join(REPO_ROOT, "__init__.py"))
    package = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = package
    spec.loader.exec_module(package)

    assert package.NODE_CLASS_MAPPINGS["MiniMaxH3CLIPCachedRef2VADualRes"].__name__ == \
        "MiniMaxH3CLIPCachedRef2VADualRes"
    assert package.NODE_DISPLAY_NAME_MAPPINGS["MiniMaxH3CLIPCachedRef2VADualRes"] == \
        "MiniMax H3 CLIP-Cached Ref2VA (Dual Resolution)"
