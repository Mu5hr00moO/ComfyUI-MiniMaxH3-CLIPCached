"""Unit tests for CachedClipProxy laziness: clip_loader_fn must only ever be
called on a real cache MISS, inside encode_from_tokens_scheduled(), never in
__init__ or tokenize(), and never more than once per proxy instance even
across multiple MISSes. No GPU, no ComfyUI, no real clip.
"""

import pytest
import torch

from minimaxh3_clipcache.fingerprint import CACHE_SCHEMA_VERSION, compute_fingerprint
from minimaxh3_clipcache.proxy import MINIMAX_H3_HIDDEN_DIM, CachedClipProxy
from minimaxh3_clipcache.store import load_conditioning, save_conditioning

CLIP_NAME = "fake_clip.safetensors"
CLIP_FILE_SIZE = 12345
CLIP_MTIME_NS = 67890


class FakeRealClip:
    """Stand-in for a real ComfyUI clip object -- constant return values,
    no actual encoding."""

    def tokenize(self, prompt, **kwargs):
        return ("real_tokens", prompt, kwargs)

    def encode_from_tokens_scheduled(self, tokens):
        return [[torch.zeros(1, MINIMAX_H3_HIDDEN_DIM), {"pooled_output": None}]]


def _make_counting_loader():
    calls = {"count": 0}
    real_clip = FakeRealClip()

    def loader():
        calls["count"] += 1
        return real_clip

    return loader, calls


def test_a_hit_never_calls_loader(tmp_path):
    prompt = "a cache hit prompt"
    kwargs = {"images": []}
    fingerprint = compute_fingerprint(
        prompt, kwargs, CLIP_NAME, CLIP_FILE_SIZE, CLIP_MTIME_NS, CACHE_SCHEMA_VERSION,
    )
    save_conditioning(
        fingerprint,
        [[torch.ones(1, MINIMAX_H3_HIDDEN_DIM), {"pooled_output": None}]],
        tmp_path,
    )

    loader, calls = _make_counting_loader()
    proxy = CachedClipProxy(loader, CLIP_NAME, CLIP_FILE_SIZE, CLIP_MTIME_NS, tmp_path)

    tokens = proxy.tokenize(prompt, **kwargs)
    cond = proxy.encode_from_tokens_scheduled(tokens)

    assert calls["count"] == 0
    assert torch.equal(cond[0][0], torch.ones(1, MINIMAX_H3_HIDDEN_DIM))


def test_b_miss_calls_loader_once(tmp_path):
    loader, calls = _make_counting_loader()
    proxy = CachedClipProxy(loader, CLIP_NAME, CLIP_FILE_SIZE, CLIP_MTIME_NS, tmp_path)

    tokens = proxy.tokenize("a cache miss prompt", images=[])
    proxy.encode_from_tokens_scheduled(tokens)

    assert calls["count"] == 1


def test_c_two_misses_in_same_proxy_load_once(tmp_path):
    loader, calls = _make_counting_loader()
    proxy = CachedClipProxy(loader, CLIP_NAME, CLIP_FILE_SIZE, CLIP_MTIME_NS, tmp_path)

    tokens_1 = proxy.tokenize("first miss prompt", images=[])
    proxy.encode_from_tokens_scheduled(tokens_1)

    tokens_2 = proxy.tokenize("second miss prompt", images=[])
    proxy.encode_from_tokens_scheduled(tokens_2)

    assert calls["count"] == 1


def test_d_hit_leaves_did_load_real_clip_false(tmp_path):
    prompt = "a cache hit prompt for did_load flag"
    kwargs = {"images": []}
    fingerprint = compute_fingerprint(
        prompt, kwargs, CLIP_NAME, CLIP_FILE_SIZE, CLIP_MTIME_NS, CACHE_SCHEMA_VERSION,
    )
    save_conditioning(
        fingerprint,
        [[torch.ones(1, MINIMAX_H3_HIDDEN_DIM), {"pooled_output": None}]],
        tmp_path,
    )

    loader, calls = _make_counting_loader()
    proxy = CachedClipProxy(loader, CLIP_NAME, CLIP_FILE_SIZE, CLIP_MTIME_NS, tmp_path)

    tokens = proxy.tokenize(prompt, **kwargs)
    proxy.encode_from_tokens_scheduled(tokens)

    assert proxy.did_load_real_clip is False


def test_e_miss_sets_did_load_real_clip_true(tmp_path):
    loader, calls = _make_counting_loader()
    proxy = CachedClipProxy(loader, CLIP_NAME, CLIP_FILE_SIZE, CLIP_MTIME_NS, tmp_path)

    tokens = proxy.tokenize("a cache miss prompt for did_load flag", images=[])
    proxy.encode_from_tokens_scheduled(tokens)

    assert proxy.did_load_real_clip is True


def test_f_force_refresh_calls_loader_and_overwrites_existing_entry(tmp_path):
    prompt = "a force refresh prompt"
    kwargs = {"images": []}
    fingerprint = compute_fingerprint(
        prompt, kwargs, CLIP_NAME, CLIP_FILE_SIZE, CLIP_MTIME_NS, CACHE_SCHEMA_VERSION,
    )
    old_value = torch.ones(1, MINIMAX_H3_HIDDEN_DIM)
    save_conditioning(fingerprint, [[old_value, {"pooled_output": None}]], tmp_path)

    loader, calls = _make_counting_loader()
    proxy = CachedClipProxy(
        loader, CLIP_NAME, CLIP_FILE_SIZE, CLIP_MTIME_NS, tmp_path, force_refresh=True,
    )

    tokens = proxy.tokenize(prompt, **kwargs)
    cond = proxy.encode_from_tokens_scheduled(tokens)

    assert calls["count"] == 1
    assert proxy.did_load_real_clip is True
    # FakeRealClip.encode_from_tokens_scheduled() returns zeros, distinguishable
    # from the pre-populated old_value of ones -- proves the returned
    # conditioning came from the real encode, not the stale cache entry.
    assert not torch.equal(cond[0][0], old_value)

    reloaded = load_conditioning(fingerprint, tmp_path)
    assert torch.equal(reloaded[0][0], cond[0][0])
    assert not torch.equal(reloaded[0][0], old_value)


def test_g_failed_unload_does_not_mask_the_original_encode_exception(tmp_path):
    """If the real encode raises AND the unload_fn then also raises in the
    finally, the exception that propagates must be the original one from the
    encode, not the one from the failed unload -- and the proxy must still
    drop its reference to the real clip so the outer safety net in nodes.py
    does not try to unload it a second time."""

    class ExplodingEncodeClip:
        patcher = object()

        def tokenize(self, prompt, **kwargs):
            return ("real_tokens", prompt, kwargs)

        def encode_from_tokens_scheduled(self, tokens):
            raise ValueError("the real encode blew up")

    def exploding_unload(patcher):
        raise RuntimeError("and then the unload blew up too")

    proxy = CachedClipProxy(
        lambda: ExplodingEncodeClip(), CLIP_NAME, CLIP_FILE_SIZE, CLIP_MTIME_NS,
        tmp_path, unload_fn=exploding_unload,
    )

    tokens = proxy.tokenize("a prompt whose encode and unload both fail", images=[])
    with pytest.raises(ValueError, match="the real encode blew up"):
        proxy.encode_from_tokens_scheduled(tokens)

    assert proxy.real_clip is None
