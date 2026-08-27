"""CachedClipProxy must reject a conditioning tensor whose hidden dim isn't
the MiniMax H3 encoder's (MINIMAX_H3_HIDDEN_DIM = 5120). This is the guard
against a clip_name pointing at the wrong checkpoint (e.g. a Gemma encoder,
hidden dim 3840) -- without it the mismatch only surfaces much later as a
cryptic matmul error inside the sampler.

The check runs on BOTH paths:
  - MISS/REFRESH: strictly before save_conditioning(), so a bad encode is
    never written to disk.
  - HIT: so an already-poisoned cache entry can't be served silently either.

No GPU, no ComfyUI, no real clip -- FakeRealClip stand-in, same pattern as
test_proxy_laziness.py.
"""

import pytest
import torch

from minimaxh3_clipcache.fingerprint import CACHE_SCHEMA_VERSION, compute_fingerprint
from minimaxh3_clipcache.proxy import MINIMAX_H3_HIDDEN_DIM, CachedClipProxy
from minimaxh3_clipcache.store import save_conditioning

CLIP_NAME = "gemma_by_mistake.safetensors"
CLIP_FILE_SIZE = 12345
CLIP_MTIME_NS = 67890

WRONG_HIDDEN_DIM = 3840  # what a Gemma text encoder would produce


class FakeClipWithHiddenDim:
    """Returns a conditioning whose main tensor has a configurable last dim,
    so a test can simulate the right encoder (5120) or the wrong one."""

    def __init__(self, hidden_dim):
        self.hidden_dim = hidden_dim
        self.encode_calls = 0

    def tokenize(self, prompt, **kwargs):
        return ("real_tokens", prompt, kwargs)

    def encode_from_tokens_scheduled(self, tokens):
        self.encode_calls += 1
        return [[torch.zeros(1, 7, self.hidden_dim), {"pooled_output": None}]]


def _fingerprint(prompt, kwargs):
    return compute_fingerprint(
        prompt, kwargs, CLIP_NAME, CLIP_FILE_SIZE, CLIP_MTIME_NS, CACHE_SCHEMA_VERSION,
    )


def _assert_message_is_helpful(exc_info):
    message = str(exc_info.value)
    assert str(WRONG_HIDDEN_DIM) in message
    assert str(MINIMAX_H3_HIDDEN_DIM) in message
    assert CLIP_NAME in message


def test_a_miss_with_wrong_hidden_dim_raises_and_writes_nothing(tmp_path):
    prompt = "a prompt encoded by the wrong checkpoint"
    kwargs = {"images": []}
    real_clip = FakeClipWithHiddenDim(WRONG_HIDDEN_DIM)
    proxy = CachedClipProxy(
        lambda: real_clip, CLIP_NAME, CLIP_FILE_SIZE, CLIP_MTIME_NS, tmp_path,
    )

    tokens = proxy.tokenize(prompt, **kwargs)
    with pytest.raises(RuntimeError) as exc_info:
        proxy.encode_from_tokens_scheduled(tokens)

    _assert_message_is_helpful(exc_info)
    # The real encode did run (this is a MISS) ...
    assert real_clip.encode_calls == 1
    # ... but the bad result must not have reached the cache.
    assert list(tmp_path.iterdir()) == []


def test_b_hit_on_a_poisoned_entry_raises(tmp_path):
    prompt = "a prompt whose cache entry is already poisoned"
    kwargs = {"images": []}
    fingerprint = _fingerprint(prompt, kwargs)
    save_conditioning(
        fingerprint,
        [[torch.zeros(1, 7, WRONG_HIDDEN_DIM), {"pooled_output": None}]],
        tmp_path,
    )

    real_clip = FakeClipWithHiddenDim(MINIMAX_H3_HIDDEN_DIM)
    proxy = CachedClipProxy(
        lambda: real_clip, CLIP_NAME, CLIP_FILE_SIZE, CLIP_MTIME_NS, tmp_path,
    )

    tokens = proxy.tokenize(prompt, **kwargs)
    with pytest.raises(RuntimeError) as exc_info:
        proxy.encode_from_tokens_scheduled(tokens)

    _assert_message_is_helpful(exc_info)
    # It was a HIT: the real encoder was never consulted.
    assert real_clip.encode_calls == 0


def test_c_correct_hidden_dim_passes_on_both_paths(tmp_path):
    prompt = "a prompt encoded by the right checkpoint"
    kwargs = {"images": []}
    real_clip = FakeClipWithHiddenDim(MINIMAX_H3_HIDDEN_DIM)

    # MISS: encodes, validates, saves, returns.
    proxy_miss = CachedClipProxy(
        lambda: real_clip, CLIP_NAME, CLIP_FILE_SIZE, CLIP_MTIME_NS, tmp_path,
    )
    tokens = proxy_miss.tokenize(prompt, **kwargs)
    cond_miss = proxy_miss.encode_from_tokens_scheduled(tokens)
    assert cond_miss[0][0].shape[-1] == MINIMAX_H3_HIDDEN_DIM
    assert real_clip.encode_calls == 1

    # HIT: served from the entry just written, still validated, no re-encode.
    proxy_hit = CachedClipProxy(
        lambda: real_clip, CLIP_NAME, CLIP_FILE_SIZE, CLIP_MTIME_NS, tmp_path,
    )
    tokens = proxy_hit.tokenize(prompt, **kwargs)
    cond_hit = proxy_hit.encode_from_tokens_scheduled(tokens)
    assert cond_hit[0][0].shape[-1] == MINIMAX_H3_HIDDEN_DIM
    assert real_clip.encode_calls == 1  # unchanged -- HIT did not load the clip
    assert proxy_hit.did_load_real_clip is False
