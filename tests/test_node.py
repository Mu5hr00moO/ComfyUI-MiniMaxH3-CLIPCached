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
import logging
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

    node = node_module.MiniMaxH3CLIPCachedImageToVideo()
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
    assert torch.equal(cond[0][0], torch.zeros(1, MINIMAX_H3_HIDDEN_DIM))
    assert unload_calls["count"] == 1


# --- Phase 2: verbose-metadata sync after a successful execute() ---

FAKE_FINGERPRINT = "abcdef01" * 8  # 64 hex chars, like a real fingerprint


def _fake_execute_setting_proxy_state(last_hit, last_core_cache_written,
                                      fingerprint=FAKE_FINGERPRINT):
    """A stock-execute() stand-in that just stamps the proxy with the state
    a real encode_from_tokens_scheduled() would have left, so the test can
    drive _sync_verbose_metadata() through each HIT/MISS combination without
    touching a real encoder or the disk cache."""

    def fake_execute(cls, clip, vae, prompt, width, height, length,
                     first_frame=None, last_frame=None):
        clip.last_fingerprint = fingerprint
        clip.last_hit = last_hit
        clip.last_core_cache_written = last_core_cache_written
        return ("cond_fake", "latent_fake")

    return fake_execute


def _patch_verbose(monkeypatch, node_module, load_return):
    """Patch load_verbose/save_verbose on the node module (nodes.py binds
    them by name at import, same reasoning as _make_spy_cached_clip_proxy)
    and record every save_verbose call."""
    save_calls = []

    def fake_save_verbose(fingerprint, system, cache_dir):
        save_calls.append({"fingerprint": fingerprint, "system": system, "cache_dir": cache_dir})

    monkeypatch.setattr(node_module, "save_verbose", fake_save_verbose)
    monkeypatch.setattr(node_module, "load_verbose", lambda fingerprint, cache_dir: load_return)
    return save_calls


def _patch_thumbnails(monkeypatch, node_module):
    """Patch save_thumbnail on the node module so tests never touch Pillow
    or the disk, and record the (fingerprint, index) it was asked for."""
    thumb_calls = []

    def fake_save_thumbnail(image, fingerprint, index, cache_dir, max_size=256):
        thumb_calls.append({"fingerprint": fingerprint, "index": index, "cache_dir": cache_dir})
        return "thumbnails/{}_{}.jpg".format(fingerprint, index)

    monkeypatch.setattr(node_module, "save_thumbnail", fake_save_thumbnail)
    return thumb_calls


@pytest.mark.parametrize("first_frame, last_frame, expected_labels", [
    (torch.zeros(1, 2, 2, 3), torch.zeros(1, 2, 2, 3), ["first_frame", "last_frame"]),
    (torch.zeros(1, 2, 2, 3), None, ["first_frame"]),
    (None, torch.zeros(1, 2, 2, 3), ["last_frame"]),
    (None, None, []),
])
def test_g_fresh_miss_writes_verbose_with_positional_references_and_thumbnails(
        monkeypatch, tmp_path, first_frame, last_frame, expected_labels):
    node_module = _load_node_module()
    fake_execute = _fake_execute_setting_proxy_state(last_hit=False, last_core_cache_written=True)
    _patch_common(monkeypatch, node_module, tmp_path, fake_execute, FakeRealClip())
    save_calls = _patch_verbose(monkeypatch, node_module, load_return=None)
    thumb_calls = _patch_thumbnails(monkeypatch, node_module)

    node = node_module.MiniMaxH3CLIPCachedImageToVideo()
    result = node.execute(
        clip_name=CLIP_NAME, vae="fake_vae", prompt="a described prompt",
        width=1344, height=768, length=124,
        first_frame=first_frame, last_frame=last_frame,
    )

    assert result == ("cond_fake", "latent_fake")
    assert len(save_calls) == 1
    assert save_calls[0]["fingerprint"] == FAKE_FINGERPRINT
    assert save_calls[0]["cache_dir"] == tmp_path
    system = save_calls[0]["system"]
    assert system["prompt"] == "a described prompt"
    assert system["clip_name"] == CLIP_NAME
    assert system["clip_file_size"] == FAKE_FILE_SIZE
    assert system["clip_mtime_ns"] == FAKE_MTIME_NS
    assert system["cache_schema_version"] == 1

    refs = system["references"]
    assert [r["label"] for r in refs] == expected_labels
    assert [r["index"] for r in refs] == list(range(len(expected_labels)))  # positional
    for r in refs:
        assert r["thumbnail"] == "thumbnails/{}_{}.jpg".format(FAKE_FINGERPRINT, r["index"])
    # save_thumbnail called exactly once per supplied reference, same index
    assert [c["index"] for c in thumb_calls] == [r["index"] for r in refs]
    assert all(c["fingerprint"] == FAKE_FINGERPRINT for c in thumb_calls)


def test_g_thumbnail_failure_still_lists_reference_without_thumbnail(monkeypatch, tmp_path, caplog):
    node_module = _load_node_module()
    fake_execute = _fake_execute_setting_proxy_state(last_hit=False, last_core_cache_written=True)
    _patch_common(monkeypatch, node_module, tmp_path, fake_execute, FakeRealClip())
    save_calls = _patch_verbose(monkeypatch, node_module, load_return=None)

    def boom_on_last_frame(image, fingerprint, index, cache_dir, max_size=256):
        if index == 1:
            raise OSError("thumbnail disk full")
        return "thumbnails/{}_{}.jpg".format(fingerprint, index)

    monkeypatch.setattr(node_module, "save_thumbnail", boom_on_last_frame)

    node = node_module.MiniMaxH3CLIPCachedImageToVideo()
    with caplog.at_level(logging.WARNING):
        node.execute(
            clip_name=CLIP_NAME, vae="fake_vae", prompt="a prompt",
            width=1344, height=768, length=124,
            first_frame=torch.zeros(1, 2, 2, 3), last_frame=torch.zeros(1, 2, 2, 3),
        )

    refs = save_calls[0]["system"]["references"]
    assert [r["label"] for r in refs] == ["first_frame", "last_frame"]  # neither reference lost
    assert refs[0]["thumbnail"] == "thumbnails/{}_0.jpg".format(FAKE_FINGERPRINT)
    assert "thumbnail" not in refs[1]  # the one that failed is listed without it
    assert any(r.levelno == logging.WARNING and "THUMBNAIL WRITE FAILED" in r.getMessage()
               for r in caplog.records)


def test_h_hit_with_no_verbose_sidecar_backfills(monkeypatch, tmp_path):
    node_module = _load_node_module()
    fake_execute = _fake_execute_setting_proxy_state(last_hit=True, last_core_cache_written=None)
    _patch_common(monkeypatch, node_module, tmp_path, fake_execute, FakeRealClip())
    save_calls = _patch_verbose(monkeypatch, node_module, load_return=None)  # legacy entry
    _patch_thumbnails(monkeypatch, node_module)

    node = node_module.MiniMaxH3CLIPCachedImageToVideo()
    node.execute(
        clip_name=CLIP_NAME, vae="fake_vae", prompt="a prompt",
        width=1344, height=768, length=124, first_frame=torch.zeros(1, 2, 2, 3),
    )

    assert len(save_calls) == 1
    refs = save_calls[0]["system"]["references"]
    assert len(refs) == 1
    assert refs[0]["label"] == "first_frame"
    assert refs[0]["index"] == 0
    assert refs[0]["thumbnail"] == "thumbnails/{}_0.jpg".format(FAKE_FINGERPRINT)


def test_i_hit_with_existing_verbose_sidecar_does_not_rewrite(monkeypatch, tmp_path):
    node_module = _load_node_module()
    fake_execute = _fake_execute_setting_proxy_state(last_hit=True, last_core_cache_written=None)
    _patch_common(monkeypatch, node_module, tmp_path, fake_execute, FakeRealClip())
    save_calls = _patch_verbose(
        monkeypatch, node_module,
        load_return={"fingerprint": FAKE_FINGERPRINT, "system": {}, "user": {}},
    )

    node = node_module.MiniMaxH3CLIPCachedImageToVideo()
    node.execute(
        clip_name=CLIP_NAME, vae="fake_vae", prompt="a prompt",
        width=1344, height=768, length=124,
    )

    assert save_calls == []


def test_j_miss_with_failed_core_cache_write_does_not_write_verbose(monkeypatch, tmp_path):
    node_module = _load_node_module()
    fake_execute = _fake_execute_setting_proxy_state(last_hit=False, last_core_cache_written=False)
    _patch_common(monkeypatch, node_module, tmp_path, fake_execute, FakeRealClip())
    save_calls = _patch_verbose(monkeypatch, node_module, load_return=None)

    node = node_module.MiniMaxH3CLIPCachedImageToVideo()
    node.execute(
        clip_name=CLIP_NAME, vae="fake_vae", prompt="a prompt",
        width=1344, height=768, length=124,
    )

    assert save_calls == []


def test_k_verbose_write_failure_is_swallowed_and_warned(monkeypatch, tmp_path, caplog):
    node_module = _load_node_module()
    fake_execute = _fake_execute_setting_proxy_state(last_hit=False, last_core_cache_written=True)
    _patch_common(monkeypatch, node_module, tmp_path, fake_execute, FakeRealClip())

    def boom(fingerprint, system, cache_dir):
        raise OSError("verbose sidecar disk full")

    monkeypatch.setattr(node_module, "save_verbose", boom)
    monkeypatch.setattr(node_module, "load_verbose", lambda fingerprint, cache_dir: None)

    node = node_module.MiniMaxH3CLIPCachedImageToVideo()
    with caplog.at_level(logging.WARNING):
        result = node.execute(
            clip_name=CLIP_NAME, vae="fake_vae", prompt="a prompt",
            width=1344, height=768, length=124,
        )

    assert result == ("cond_fake", "latent_fake")
    assert any(r.levelno == logging.WARNING and "VERBOSE WRITE FAILED" in r.getMessage()
               for r in caplog.records)


def test_f_node_class_mappings_has_exactly_one_matching_key():
    spec = importlib.util.spec_from_file_location(
        "minimaxh3clipcached_package_under_test", os.path.join(REPO_ROOT, "__init__.py"))
    package = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = package
    spec.loader.exec_module(package)

    assert list(package.NODE_CLASS_MAPPINGS.keys()) == ["MiniMaxH3CLIPCachedImageToVideo"]
    assert package.NODE_CLASS_MAPPINGS["MiniMaxH3CLIPCachedImageToVideo"].__name__ == "MiniMaxH3CLIPCachedImageToVideo"
