"""Unit tests for CachedClipProxy laziness: clip_loader_fn must only ever be
called on a real cache MISS, inside encode_from_tokens_scheduled(), never in
__init__ or tokenize(), and never more than once per proxy instance even
across multiple MISSes. No GPU, no ComfyUI, no real clip.
"""

import torch

from caching.fingerprint import CACHE_SCHEMA_VERSION, compute_fingerprint
from caching.proxy import CachedClipProxy
from caching.store import save_conditioning

CLIP_NAME = "fake_clip.safetensors"
CLIP_FILE_SIZE = 12345
CLIP_MTIME_NS = 67890


class FakeRealClip:
    """Stand-in for a real ComfyUI clip object -- constant return values,
    no actual encoding."""

    def tokenize(self, prompt, **kwargs):
        return ("real_tokens", prompt, kwargs)

    def encode_from_tokens_scheduled(self, tokens):
        return [[torch.zeros(1, 3), {"pooled_output": None}]]


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
    save_conditioning(fingerprint, [[torch.ones(1, 3), {"pooled_output": None}]], tmp_path)

    loader, calls = _make_counting_loader()
    proxy = CachedClipProxy(loader, CLIP_NAME, CLIP_FILE_SIZE, CLIP_MTIME_NS, tmp_path)

    tokens = proxy.tokenize(prompt, **kwargs)
    cond = proxy.encode_from_tokens_scheduled(tokens)

    assert calls["count"] == 0
    assert torch.equal(cond[0][0], torch.ones(1, 3))


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
