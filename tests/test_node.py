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
import logging
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
FAKE_CTIME_NS = 333


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
                         lambda clip_name: (FAKE_FILE_SIZE, FAKE_MTIME_NS, FAKE_CTIME_NS))
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
    assert kwargs["clip_ctime_ns"] == FAKE_CTIME_NS
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


def test_e2_abi_unavailable_forces_refresh_and_passes_unavailable_marker(monkeypatch, tmp_path):
    """When get_encoder_abi_id() reports the encoder ABI identity is
    unavailable (plan audit point 1), execute() must build the proxy with
    force_refresh=True even under cache_mode="auto", and pass
    encoder_abi_id="unavailable" -- never serve or write a HIT computed under
    an unverified tokenizer implementation."""
    node_module = _load_node_module()
    real_clip = FakeRealClip()

    def fake_execute(cls, clip, vae, prompt, width, height, length,
                      first_frame=None, last_frame=None):
        tokens = clip.tokenize(prompt, images=[])
        cond = clip.encode_from_tokens_scheduled(tokens)
        return (cond, "latent_fake")

    _patch_common(monkeypatch, node_module, tmp_path, fake_execute, real_clip)
    monkeypatch.setattr(node_module, "get_encoder_abi_id", lambda: (None, False))

    SpyCachedClipProxy, construction_calls = _make_spy_cached_clip_proxy()
    monkeypatch.setattr(node_module, "CachedClipProxy", SpyCachedClipProxy)

    node = node_module.MiniMaxH3CLIPCachedFL2VA()
    node.execute(
        clip_name=CLIP_NAME, vae="fake_vae", prompt="a prompt",
        width=1344, height=768, length=124, cache_mode="auto",
    )

    assert len(construction_calls) == 1
    _, kwargs = construction_calls[0]
    assert kwargs["force_refresh"] is True
    assert kwargs["encoder_abi_id"] == "unavailable"


def test_e3_abi_available_passes_real_id_and_respects_cache_mode(monkeypatch, tmp_path):
    """With the ABI identity available, execute() passes it straight through
    to the proxy and force_refresh follows cache_mode alone."""
    node_module = _load_node_module()
    real_clip = FakeRealClip()

    def fake_execute(cls, clip, vae, prompt, width, height, length,
                      first_frame=None, last_frame=None):
        tokens = clip.tokenize(prompt, images=[])
        cond = clip.encode_from_tokens_scheduled(tokens)
        return (cond, "latent_fake")

    _patch_common(monkeypatch, node_module, tmp_path, fake_execute, real_clip)
    monkeypatch.setattr(node_module, "get_encoder_abi_id", lambda: ("0.34.2:deadbeef", True))

    SpyCachedClipProxy, construction_calls = _make_spy_cached_clip_proxy()
    monkeypatch.setattr(node_module, "CachedClipProxy", SpyCachedClipProxy)

    node = node_module.MiniMaxH3CLIPCachedFL2VA()
    node.execute(
        clip_name=CLIP_NAME, vae="fake_vae", prompt="a prompt",
        width=1344, height=768, length=124, cache_mode="auto",
    )

    _, kwargs = construction_calls[0]
    assert kwargs["force_refresh"] is False
    assert kwargs["encoder_abi_id"] == "0.34.2:deadbeef"


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


def test_i_is_changed_auto_reflects_checkpoint_file_identity(monkeypatch):
    """cache_mode="auto" must change when the checkpoint file underneath
    an unchanged clip_name changes (swapped file, same filename) --
    otherwise ComfyUI's own execution cache could skip re-running this
    node entirely and our on-disk fingerprint (which does include file
    identity) would never even get computed. Same-stat calls must stay
    stable so an unchanged graph still hits ComfyUI's own execution
    cache."""
    node_module = _load_node_module()
    cls = node_module.MiniMaxH3CLIPCachedFL2VA

    monkeypatch.setattr(node_module, "resolve_clip_stat", lambda clip_name: (111, 222, 333))
    before = cls.IS_CHANGED(cache_mode="auto", clip_name=CLIP_NAME, prompt="p")
    before_again = cls.IS_CHANGED(cache_mode="auto", clip_name=CLIP_NAME, prompt="p")
    assert before == before_again

    # Same filename, size and mtime; only ctime changes after an in-place
    # rewrite or replacement whose original mtime was restored.
    monkeypatch.setattr(node_module, "resolve_clip_stat", lambda clip_name: (111, 222, 444))
    after = cls.IS_CHANGED(cache_mode="auto", clip_name=CLIP_NAME, prompt="p")
    assert after != before


def test_i_is_changed_returns_nan_when_checkpoint_file_missing(monkeypatch):
    """If the checkpoint named in the graph is gone from disk, IS_CHANGED must
    not raise from inside ComfyUI's scheduling layer -- it returns NaN (same
    as cache_mode="refresh") so execute() runs and surfaces the real
    FileNotFoundError as this node's own error."""
    node_module = _load_node_module()
    cls = node_module.MiniMaxH3CLIPCachedFL2VA

    def _missing(clip_name):
        raise FileNotFoundError("text_encoders file {!r} not found".format(clip_name))

    monkeypatch.setattr(node_module, "resolve_clip_stat", _missing)

    result = cls.IS_CHANGED(cache_mode="auto", clip_name=CLIP_NAME, prompt="p")
    assert isinstance(result, float) and math.isnan(result)
    assert result != result  # NaN != NaN -- forces re-execution instead of raising


def test_i_is_changed_returns_nan_when_encoder_abi_unavailable(monkeypatch):
    """If the encoder ABI identity can't be determined (plan audit point 1),
    IS_CHANGED must return NaN so ComfyUI always re-executes -- caching is
    unsafe this session and execute() forces a real encode too."""
    node_module = _load_node_module()
    cls = node_module.MiniMaxH3CLIPCachedFL2VA

    monkeypatch.setattr(node_module, "get_encoder_abi_id", lambda: (None, False))

    result = cls.IS_CHANGED(cache_mode="auto", clip_name=CLIP_NAME, prompt="p")
    assert isinstance(result, float) and math.isnan(result)
    assert result != result


def test_i_is_changed_auto_folds_in_abi_id_and_stays_stable(monkeypatch):
    """cache_mode="auto" folds the encoder ABI id in as the last tuple
    element: stable across calls while the ABI is unchanged (so an unchanged
    graph still hits ComfyUI's own execution cache), and different when the
    ABI id changes (forcing re-execution after an upstream tokenizer change)."""
    node_module = _load_node_module()
    cls = node_module.MiniMaxH3CLIPCachedFL2VA

    monkeypatch.setattr(node_module, "resolve_clip_stat", lambda clip_name: (111, 222, 333))
    monkeypatch.setattr(node_module, "get_encoder_abi_id", lambda: ("0.34.2:deadbeef", True))

    first = cls.IS_CHANGED(cache_mode="auto", clip_name=CLIP_NAME, prompt="p")
    second = cls.IS_CHANGED(cache_mode="auto", clip_name=CLIP_NAME, prompt="p")
    assert first == second
    assert first[-1] == "0.34.2:deadbeef"

    monkeypatch.setattr(node_module, "get_encoder_abi_id", lambda: ("0.34.2:cafef00d", True))
    after_abi_change = cls.IS_CHANGED(cache_mode="auto", clip_name=CLIP_NAME, prompt="p")
    assert after_abi_change != first


def test_j_encoder_unloaded_before_stock_nodes_post_encode_work(monkeypatch, tmp_path):
    """The real encoder must be released as soon as encode_from_tokens_scheduled()
    returns, before any work the stock node still does afterwards (e.g. FL2VA's
    keyframe VAE encode) -- not only in nodes.py's outer finally once the whole
    stock execute() has returned."""
    node_module = _load_node_module()
    real_clip = FakeRealClip()

    def fake_execute(cls, clip, vae, prompt, width, height, length,
                      first_frame=None, last_frame=None):
        tokens = clip.tokenize(prompt, images=[])
        cond = clip.encode_from_tokens_scheduled(tokens)
        # Simulate the stock node's post-encode VAE work (e.g. FL2VA's
        # keyframe encode): the encoder must already be gone by this point.
        assert unload_calls["count"] == 1
        assert clip.real_clip is None
        return (cond, "latent_fake")

    unload_calls = _patch_common(monkeypatch, node_module, tmp_path, fake_execute, real_clip)

    node = node_module.MiniMaxH3CLIPCachedFL2VA()
    cond, latent = node.execute(
        clip_name=CLIP_NAME, vae="fake_vae", prompt="a prompt",
        width=1344, height=768, length=124,
    )

    assert real_clip.tokenize_calls == 1
    assert real_clip.encode_calls == 1
    assert unload_calls["count"] == 1
    assert unload_calls["args"][0][0] == (real_clip.patcher,)


def test_k_finally_still_unloads_when_real_encode_itself_raises(monkeypatch, tmp_path):
    """If the real encoder's own encode_from_tokens_scheduled() raises, the
    encoder must still be released exactly once -- today that happens inside
    CachedClipProxy's own try/finally (added for the process-wide encoder
    lock), with nodes.py's outer finally as a no-op safety net once
    proxy.real_clip is already None. This test only asserts the outcome
    (exactly one unload, correct patcher), not which of the two release
    points fired, so it stays valid regardless of which one does."""
    node_module = _load_node_module()

    class FailingRealClip(FakeRealClip):
        def encode_from_tokens_scheduled(self, tokens):
            self.encode_calls += 1
            raise RuntimeError("simulated encoder failure")

    real_clip = FailingRealClip()

    def fake_execute(cls, clip, vae, prompt, width, height, length,
                      first_frame=None, last_frame=None):
        tokens = clip.tokenize(prompt, images=[])
        return clip.encode_from_tokens_scheduled(tokens)

    unload_calls = _patch_common(monkeypatch, node_module, tmp_path, fake_execute, real_clip)

    node = node_module.MiniMaxH3CLIPCachedFL2VA()
    with pytest.raises(RuntimeError, match="simulated encoder failure"):
        node.execute(
            clip_name=CLIP_NAME, vae="fake_vae", prompt="a prompt",
            width=1344, height=768, length=124,
        )

    assert unload_calls["count"] == 1
    assert unload_calls["args"][0][0] == (real_clip.patcher,)


def test_q_outer_finally_failed_unload_does_not_mask_original_exception(monkeypatch, tmp_path):
    """If the stock execute() raises with the real encoder still resident AND
    nodes.py's outer safety-net unload then also raises, the exception that
    propagates out of execute() must be the original one, not the unload
    failure."""
    node_module = _load_node_module()

    class RaisingProxy(CachedClipProxy):
        def encode_from_tokens_scheduled(self, tokens):
            # Simulate a failure that leaves the real clip resident before the
            # proxy's own finally could release it, so nodes.py's outer
            # safety-net branch (proxy.real_clip is not None) is the one that
            # runs the unload.
            self.did_load_real_clip = True
            self._real_clip = FakeRealClip()
            raise RuntimeError("original failure")

    monkeypatch.setattr(node_module, "resolve_clip_stat",
                         lambda clip_name: (FAKE_FILE_SIZE, FAKE_MTIME_NS, FAKE_CTIME_NS))
    monkeypatch.setattr(node_module, "build_clip_loader_fn",
                         lambda clip_name: (lambda: FakeRealClip()))
    monkeypatch.setattr(node_module, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(node_module, "CachedClipProxy", RaisingProxy)

    def fake_execute(cls, clip, vae, prompt, width, height, length,
                      first_frame=None, last_frame=None):
        tokens = clip.tokenize(prompt, images=[])
        return clip.encode_from_tokens_scheduled(tokens)

    monkeypatch.setattr(MiniMaxH3ImageToVideo, "execute", classmethod(fake_execute))

    def exploding_unload(*args, **kwargs):
        raise RuntimeError("unload also failed")

    monkeypatch.setattr(comfy.model_management, "unload_model_and_clones", exploding_unload)

    node = node_module.MiniMaxH3CLIPCachedFL2VA()
    with pytest.raises(RuntimeError, match="original failure"):
        node.execute(
            clip_name=CLIP_NAME, vae="fake_vae", prompt="a prompt",
            width=1344, height=768, length=124,
        )


class _FakeProxy:
    """Stand-in for CachedClipProxy carrying only the three fields
    _sync_verbose_metadata() reads back after a run."""

    def __init__(self, fingerprint="a" * 64, last_hit=False, last_core_cache_written=True):
        self.last_fingerprint = fingerprint
        self.last_hit = last_hit
        self.last_core_cache_written = last_core_cache_written


def _make_core_json(cache_dir, fingerprint="a" * 64):
    """Create the core <fingerprint>.json so _sync_verbose_metadata()'s
    under-the-lock re-check ("has Delete already removed the core entry?")
    passes. Every scenario these tests simulate -- a fresh MISS whose core
    cache landed, or a HIT of an existing entry -- genuinely has this file on
    disk; only the "Delete won the race" case (test_s) deliberately omits it.
    """
    (cache_dir / "{}.json".format(fingerprint)).write_bytes(b"{}")


def test_l_sync_verbose_fresh_miss_writes_fl2va_variant(monkeypatch, tmp_path):
    """A fresh MISS whose core cache landed on disk writes a sidecar whose
    system block is tagged node_variant="fl2va" and lists the keyframes as
    positional image references with their labels."""
    node_module = _load_node_module()
    monkeypatch.setattr(node_module, "CACHE_DIR", tmp_path)
    from minimaxh3_clipcache.verbose_store import load_verbose

    _make_core_json(tmp_path)
    proxy = _FakeProxy(last_hit=False, last_core_cache_written=True)
    items = [("image", torch.zeros(1, 4, 4, 3)), ("image", torch.ones(1, 4, 4, 3))]
    node_module._sync_verbose_metadata(
        proxy, "fl2va", "a prompt", CLIP_NAME, FAKE_FILE_SIZE, FAKE_MTIME_NS,
        items, labels=["first_frame", "last_frame"])

    system = load_verbose("a" * 64, tmp_path)["system"]
    assert system["node_variant"] == "fl2va"
    assert system["prompt"] == "a prompt"
    assert [r["type"] for r in system["references"]] == ["image", "image"]
    assert [r["label"] for r in system["references"]] == ["first_frame", "last_frame"]
    assert [r["index"] for r in system["references"]] == [0, 1]


def test_m_sync_verbose_miss_without_core_cache_write_does_nothing(monkeypatch, tmp_path):
    """last_core_cache_written is False (this run's cache write failed): there
    is no entry to describe, so no sidecar is written. The core <fp>.json is
    created here so the test reaches and exercises the
    fresh_miss_written / hit_needs_backfill guard rather than short-circuiting
    on the earlier "core entry gone" re-check."""
    node_module = _load_node_module()
    monkeypatch.setattr(node_module, "CACHE_DIR", tmp_path)
    from minimaxh3_clipcache.verbose_store import load_verbose

    _make_core_json(tmp_path)
    proxy = _FakeProxy(last_hit=False, last_core_cache_written=False)
    node_module._sync_verbose_metadata(
        proxy, "fl2va", "a prompt", CLIP_NAME, FAKE_FILE_SIZE, FAKE_MTIME_NS, [])

    assert load_verbose("a" * 64, tmp_path) is None


def test_n_sync_verbose_hit_without_sidecar_backfills(monkeypatch, tmp_path):
    """A HIT of a legacy entry that has no sidecar yet backfills one."""
    node_module = _load_node_module()
    monkeypatch.setattr(node_module, "CACHE_DIR", tmp_path)
    from minimaxh3_clipcache.verbose_store import load_verbose

    _make_core_json(tmp_path)
    proxy = _FakeProxy(last_hit=True, last_core_cache_written=None)
    node_module._sync_verbose_metadata(
        proxy, "fl2va", "a prompt", CLIP_NAME, FAKE_FILE_SIZE, FAKE_MTIME_NS,
        [("image", torch.zeros(1, 4, 4, 3))], labels=["first_frame"])

    assert load_verbose("a" * 64, tmp_path)["system"]["node_variant"] == "fl2va"


def test_o_sync_verbose_hit_with_existing_sidecar_does_not_rewrite(monkeypatch, tmp_path):
    """A HIT of an entry that already has a sidecar is a no-op -- the existing
    file is left byte-for-byte untouched."""
    node_module = _load_node_module()
    monkeypatch.setattr(node_module, "CACHE_DIR", tmp_path)
    from minimaxh3_clipcache.verbose_store import save_verbose

    _make_core_json(tmp_path)
    save_verbose("a" * 64, {"prompt": "original", "node_variant": "fl2va", "references": []}, tmp_path)
    before = (tmp_path / ("a" * 64 + ".verbose.json")).read_bytes()

    proxy = _FakeProxy(last_hit=True, last_core_cache_written=None)
    node_module._sync_verbose_metadata(
        proxy, "fl2va", "a different prompt", CLIP_NAME, FAKE_FILE_SIZE, FAKE_MTIME_NS, [])

    assert (tmp_path / ("a" * 64 + ".verbose.json")).read_bytes() == before


def test_p_sync_verbose_write_failure_is_swallowed(monkeypatch, tmp_path, caplog):
    """save_verbose() raising must not propagate: the verbose layer is not the
    source of truth, the core cache result stands regardless."""
    node_module = _load_node_module()
    monkeypatch.setattr(node_module, "CACHE_DIR", tmp_path)

    def boom(fingerprint, system, cache_dir):
        raise OSError("verbose sidecar disk full")

    monkeypatch.setattr(node_module, "save_verbose", boom)

    _make_core_json(tmp_path)
    proxy = _FakeProxy(last_hit=False, last_core_cache_written=True)
    with caplog.at_level(logging.WARNING):
        node_module._sync_verbose_metadata(
            proxy, "fl2va", "a prompt", CLIP_NAME, FAKE_FILE_SIZE, FAKE_MTIME_NS, [])

    assert any("VERBOSE WRITE FAILED" in r.getMessage() for r in caplog.records)


def test_r_sync_verbose_fresh_miss_records_comfyui_version(monkeypatch, tmp_path):
    """The verbose sidecar written after a fresh MISS carries an informational
    "comfyui_version" field in its system block. It is best-effort and never
    part of the fingerprint -- it only helps diagnose why an older entry's
    encode differs after an upstream ComfyUI update."""
    node_module = _load_node_module()
    monkeypatch.setattr(node_module, "CACHE_DIR", tmp_path)
    from minimaxh3_clipcache.verbose_store import load_verbose
    import comfyui_version

    _make_core_json(tmp_path)
    proxy = _FakeProxy(last_hit=False, last_core_cache_written=True)
    node_module._sync_verbose_metadata(
        proxy, "fl2va", "a prompt", CLIP_NAME, FAKE_FILE_SIZE, FAKE_MTIME_NS, [])

    system = load_verbose("a" * 64, tmp_path)["system"]
    assert system["comfyui_version"] == comfyui_version.__version__


def test_s_sync_verbose_skips_write_when_delete_won_the_race_for_the_core_entry(monkeypatch, tmp_path):
    """A run can be a genuine fresh MISS (proxy: last_hit False,
    last_core_cache_written True) and still have its core <fp>.json deleted by
    a Cache Manager Delete before _sync_verbose_metadata() takes the
    fingerprint lock. The whole decision now runs under that lock and
    re-checks the core entry: with no <fp>.json on disk, nothing is written,
    so a phantom sidecar with no core cache behind it is never resurrected."""
    node_module = _load_node_module()
    monkeypatch.setattr(node_module, "CACHE_DIR", tmp_path)
    from minimaxh3_clipcache.verbose_store import load_verbose

    save_verbose_calls = []
    monkeypatch.setattr(node_module, "save_verbose",
                         lambda *a, **k: save_verbose_calls.append((a, k)))

    # Delete already removed the core entry -- no <fp>.json is created here.
    proxy = _FakeProxy(last_hit=False, last_core_cache_written=True)
    node_module._sync_verbose_metadata(
        proxy, "fl2va", "a prompt", CLIP_NAME, FAKE_FILE_SIZE, FAKE_MTIME_NS,
        [("image", torch.zeros(1, 4, 4, 3))], labels=["first_frame"])

    assert save_verbose_calls == []
    assert load_verbose("a" * 64, tmp_path) is None


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
