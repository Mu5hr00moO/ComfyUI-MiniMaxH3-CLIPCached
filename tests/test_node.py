"""Unit tests for the public MiniMaxH3CLIPCachedImageToVideo node (nodes.py at
repo root) and its NODE_CLASS_MAPPINGS (__init__.py). No GPU, no real
encoder, no real ComfyUI startup -- MiniMaxH3ImageToVideo.execute(), the
cache loader helpers, and unload_model_and_clones() are all monkeypatched.

Both files are loaded via importlib with a private module name, the same
way ComfyUI's own custom-node loader (nodes.load_custom_node() in the main
ComfyUI repo) loads a custom node package in production -- never as a bare
`import nodes`, which would collide with ComfyUI's own top-level nodes.py.
"""

import importlib.util
import os
import sys

import comfy.model_management
from comfy_extras.nodes_minimax_h3 import MiniMaxH3ImageToVideo

from caching.proxy import CachedClipProxy

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
        return "real_cond"


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

    node = node_module.MiniMaxH3CLIPCachedImageToVideo()
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

    node = node_module.MiniMaxH3CLIPCachedImageToVideo()
    cond, latent = node.execute(
        clip_name=CLIP_NAME, vae="fake_vae", prompt="a prompt",
        width=1344, height=768, length=124,
    )

    assert real_clip.tokenize_calls == 1
    assert real_clip.encode_calls == 1
    assert cond == "real_cond"
    assert unload_calls["count"] == 1
    assert unload_calls["args"][0][0] == (real_clip.patcher,)


def _make_spy_cached_clip_proxy():
    """A CachedClipProxy subclass that records every constructor call
    (args, kwargs) before delegating to the real __init__ -- so the rest of
    execute() (tokenize/encode_from_tokens_scheduled/did_load_real_clip) still
    behaves exactly like the real proxy, we just also get to inspect what it
    was built with.

    nodes.py does `from caching.proxy import CachedClipProxy`, which binds a
    private name inside nodes.py's own module namespace at import time --
    patching caching.proxy.CachedClipProxy afterwards would not reach that
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

    node = node_module.MiniMaxH3CLIPCachedImageToVideo()
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
    assert cond == "real_cond"
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

    node = node_module.MiniMaxH3CLIPCachedImageToVideo()
    cond, latent = node.execute(
        clip_name=CLIP_NAME, vae="fake_vae", prompt="a prompt",
        width=1344, height=768, length=124, cache_mode="refresh",
    )

    assert len(construction_calls) == 1
    _, kwargs = construction_calls[0]
    assert kwargs["force_refresh"] is True
    assert real_clip.tokenize_calls == 1
    assert real_clip.encode_calls == 1
    assert cond == "real_cond"
    assert unload_calls["count"] == 1


def test_f_node_class_mappings_has_exactly_one_matching_key():
    spec = importlib.util.spec_from_file_location(
        "minimaxh3clipcached_package_under_test", os.path.join(REPO_ROOT, "__init__.py"))
    package = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = package
    spec.loader.exec_module(package)

    assert list(package.NODE_CLASS_MAPPINGS.keys()) == ["MiniMaxH3CLIPCachedImageToVideo"]
    assert package.NODE_CLASS_MAPPINGS["MiniMaxH3CLIPCachedImageToVideo"].__name__ == "MiniMaxH3CLIPCachedImageToVideo"
