"""Unit tests for the public MiniMaxH3CLIPCachedFL2VA node (nodes.py at
repo root) and its NODE_CLASS_MAPPINGS (__init__.py). No GPU, no real
encoder, no real ComfyUI startup -- MiniMaxH3ImageToVideo.execute(), the
cache loader helpers, and unload_model_and_clones() are all monkeypatched.

Both files are loaded via importlib with a private module name, the same
way ComfyUI's own custom-node loader (nodes.load_custom_node() in the main
ComfyUI repo) loads a custom node package in production -- never as a bare
`import nodes`, which would collide with ComfyUI's own top-level nodes.py.
"""

import importlib.util
import math
import os
import sys

import pytest
import torch

import comfy.model_management
from comfy_extras.nodes_minimax_h3 import MiniMaxH3ImageToVideo

from minimaxh3_clipcache.proxy import MINIMAX_H3_HIDDEN_DIM, CachedClipProxy

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CLIP_NAME = "fake_clip.safetensors"
FAKE_FILE_SIZE = 111
FAKE_MTIME_NS = 222


def _load_node_module():
    spec = importlib.util.spec_from_file_location(
        "minimaxh3clipcached_nodes_under_test", os.path.join(REPO_ROOT, "nodes.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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
                         lambda clip_name: (FAKE_FILE_SIZE, FAKE_MTIME_NS))
    monkeypatch.setattr(node_module, "build_clip_loader_fn",
                         lambda clip_name: (lambda: real_clip))
    monkeypatch.setattr(node_module, "CACHE_DIR", tmp_path)

    fake_unload, unload_calls = _make_unload_counter()
    monkeypatch.setattr(comfy.model_management, "unload_model_and_clones", fake_unload)
    return unload_calls


def test_a_execute_not_touching_clip_never_unloads(monkeypatch, tmp_path):
    node_module = _load_node_module()
    real_clip = FakeRealClip()

    def fake_execute(cls, clip, vae, prompt, width, height, length,
                      first_frame=None, last_frame=None):
        return ("cond_fake", "latent_fake")

    unload_calls = _patch_common(monkeypatch, node_module, tmp_path, fake_execute, real_clip)

    node = node_module.MiniMaxH3CLIPCachedFL2VA()
    cond, latent = node.execute(
        clip_name=CLIP_NAME, vae="fake_vae", prompt="a prompt",
        width=1344, height=768, length=124,
    )

    assert (cond, latent) == ("cond_fake", "latent_fake")
    assert unload_calls["count"] == 0


def test_b_execute_touching_clip_unloads_exactly_once(monkeypatch, tmp_path):
    node_module = _load_node_module()
    real_clip = FakeRealClip()

    def fake_execute(cls, clip, vae, prompt, width, height, length,
                      first_frame=None, last_frame=None):
        tokens = clip.tokenize(prompt, images=[])
        cond = clip.encode_from_tokens_scheduled(tokens)
        return (cond, "latent_fake")

    unload_calls = _patch_common(monkeypatch, node_module, tmp_path, fake_execute, real_clip)

    node = node_module.MiniMaxH3CLIPCachedFL2VA()
    cond, latent = node.execute(
        clip_name=CLIP_NAME, vae="fake_vae", prompt="a prompt",
        width=1344, height=768, length=124,
    )

    assert real_clip.tokenize_calls == 1
    assert real_clip.encode_calls == 1
    assert torch.equal(cond[0][0], torch.zeros(1, MINIMAX_H3_HIDDEN_DIM))
    assert unload_calls["count"] == 1
    assert unload_calls["args"][0][0] == (real_clip.patcher,)


def test_c_execute_raising_after_loading_clip_still_unloads_and_propagates(monkeypatch, tmp_path):
    node_module = _load_node_module()
    real_clip = FakeRealClip()

    def fake_execute(cls, clip, vae, prompt, width, height, length,
                      first_frame=None, last_frame=None):
        # Touch the clip first so proxy.did_load_real_clip becomes True...
        tokens = clip.tokenize(prompt, images=[])
        clip.encode_from_tokens_scheduled(tokens)
        # ...then blow up the way the stock node could on a real failure.
        raise RuntimeError("simulated failure")

    unload_calls = _patch_common(monkeypatch, node_module, tmp_path, fake_execute, real_clip)

    node = node_module.MiniMaxH3CLIPCachedFL2VA()
    with pytest.raises(RuntimeError, match="simulated failure"):
        node.execute(
            clip_name=CLIP_NAME, vae="fake_vae", prompt="a prompt",
            width=1344, height=768, length=124,
        )

    assert real_clip.tokenize_calls == 1
    assert real_clip.encode_calls == 1
    assert unload_calls["count"] == 1
    assert unload_calls["args"][0][0] == (real_clip.patcher,)


def _make_spy_cached_clip_proxy():
    """A CachedClipProxy subclass that records every constructor call
    (args, kwargs) before delegating to the real __init__ -- so the rest of
    execute() (tokenize/encode_from_tokens_scheduled/did_load_real_clip) still
    behaves exactly like the real proxy, we just also get to inspect what it
    was built with.

    nodes.py does `from minimaxh3_clipcache.proxy import CachedClipProxy`, which binds a
    private name inside nodes.py's own module namespace at import time --
    patching minimaxh3_clipcache.proxy.CachedClipProxy afterwards would not reach that
    already-bound name. So the test patches node_module.CachedClipProxy
    directly, the same way _patch_common() already patches
    resolve_clip_stat/build_clip_loader_fn/CACHE_DIR on node_module.
    """
    construction_calls = []

    class SpyCachedClipProxy(CachedClipProxy):
        def __init__(self, *args, **kwargs):
            construction_calls.append((args, kwargs))
            super().__init__(*args, **kwargs)

    return SpyCachedClipProxy, construction_calls


def test_d_cache_mode_auto_builds_proxy_with_force_refresh_false(monkeypatch, tmp_path):
    node_module = _load_node_module()
    real_clip = FakeRealClip()

    def fake_execute(cls, clip, vae, prompt, width, height, length,
                      first_frame=None, last_frame=None):
        tokens = clip.tokenize(prompt, images=[])
        cond = clip.encode_from_tokens_scheduled(tokens)
        return (cond, "latent_fake")

    unload_calls = _patch_common(monkeypatch, node_module, tmp_path, fake_execute, real_clip)

    SpyCachedClipProxy, construction_calls = _make_spy_cached_clip_proxy()
    monkeypatch.setattr(node_module, "CachedClipProxy", SpyCachedClipProxy)

    node = node_module.MiniMaxH3CLIPCachedFL2VA()
    cond, latent = node.execute(
        clip_name=CLIP_NAME, vae="fake_vae", prompt="a prompt",
        width=1344, height=768, length=124, cache_mode="auto",
    )

    assert len(construction_calls) == 1
    _, kwargs = construction_calls[0]
    assert kwargs["force_refresh"] is False
    # rest of execute() still works exactly as with the real proxy
    assert real_clip.tokenize_calls == 1
    assert real_clip.encode_calls == 1
    assert torch.equal(cond[0][0], torch.zeros(1, MINIMAX_H3_HIDDEN_DIM))
    assert unload_calls["count"] == 1


def test_e_cache_mode_refresh_builds_proxy_with_force_refresh_true(monkeypatch, tmp_path):
    node_module = _load_node_module()
    real_clip = FakeRealClip()

    def fake_execute(cls, clip, vae, prompt, width, height, length,
                      first_frame=None, last_frame=None):
        tokens = clip.tokenize(prompt, images=[])
        cond = clip.encode_from_tokens_scheduled(tokens)
        return (cond, "latent_fake")

    unload_calls = _patch_common(monkeypatch, node_module, tmp_path, fake_execute, real_clip)

    SpyCachedClipProxy, construction_calls = _make_spy_cached_clip_proxy()
    monkeypatch.setattr(node_module, "CachedClipProxy", SpyCachedClipProxy)

    node = node_module.MiniMaxH3CLIPCachedFL2VA()
    cond, latent = node.execute(
        clip_name=CLIP_NAME, vae="fake_vae", prompt="a prompt",
        width=1344, height=768, length=124, cache_mode="refresh",
    )

    assert len(construction_calls) == 1
    _, kwargs = construction_calls[0]
    assert kwargs["force_refresh"] is True
    assert real_clip.tokenize_calls == 1
    assert real_clip.encode_calls == 1
    assert torch.equal(cond[0][0], torch.zeros(1, MINIMAX_H3_HIDDEN_DIM))
    assert unload_calls["count"] == 1


def test_g_is_changed_refresh_forces_reexecution_every_call():
    """cache_mode="refresh" must return a value that never equals itself
    between two consecutive Queue clicks, so ComfyUI's signature comparison
    always misses and execute() actually re-runs. NaN is that value."""
    node_module = _load_node_module()
    cls = node_module.MiniMaxH3CLIPCachedFL2VA

    # ComfyUI hands IS_CHANGED every graph input as a kwarg -- make sure the
    # signature absorbs the ones we don't name.
    first = cls.IS_CHANGED(cache_mode="refresh", clip_name=CLIP_NAME, vae="v",
                            prompt="p", width=1344, height=768, length=124)
    second = cls.IS_CHANGED(cache_mode="refresh", clip_name=CLIP_NAME, vae="v",
                             prompt="p", width=1344, height=768, length=124)

    assert isinstance(first, float) and math.isnan(first)
    assert isinstance(second, float) and math.isnan(second)
    # NaN != NaN -- this inequality is exactly what forces re-execution.
    assert not (first == second)


def test_h_is_changed_auto_is_stable_across_calls():
    """cache_mode="auto" (and the default) must return a stable, self-equal
    value so an unchanged graph still hits ComfyUI's own execution cache."""
    node_module = _load_node_module()
    cls = node_module.MiniMaxH3CLIPCachedFL2VA

    assert cls.IS_CHANGED(cache_mode="auto", prompt="p") == \
           cls.IS_CHANGED(cache_mode="auto", prompt="p")
    # default (IS_CHANGED called with no cache_mode at all, e.g. optional
    # input left unconnected) must also be stable
    assert cls.IS_CHANGED(prompt="p", width=1344) == cls.IS_CHANGED(prompt="p", width=1344)


def test_f_node_class_mappings_has_both_node_keys():
    """Regression: adding the Ref2VA node must not drop or shadow the FL2VA
    entry -- both keys must be present, each pointing at its own class."""
    spec = importlib.util.spec_from_file_location(
        "minimaxh3clipcached_package_under_test", os.path.join(REPO_ROOT, "__init__.py"))
    package = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = package
    spec.loader.exec_module(package)

    assert set(package.NODE_CLASS_MAPPINGS.keys()) == {
        "MiniMaxH3CLIPCachedFL2VA", "MiniMaxH3CLIPCachedRef2VA"}
    assert package.NODE_CLASS_MAPPINGS["MiniMaxH3CLIPCachedFL2VA"].__name__ == "MiniMaxH3CLIPCachedFL2VA"
    assert package.NODE_CLASS_MAPPINGS["MiniMaxH3CLIPCachedRef2VA"].__name__ == "MiniMaxH3CLIPCachedRef2VA"
    assert set(package.NODE_DISPLAY_NAME_MAPPINGS.keys()) == set(package.NODE_CLASS_MAPPINGS.keys())
