"""A failed cache write on a MISS must not discard the completed encode.

The encode result is expensive and already computed by the time
save_conditioning() runs, so a write failure (disk full, read-only cache
dir, ...) is logged as a warning and the conditioning is still returned.
No GPU, no ComfyUI, no real clip -- save_conditioning is monkeypatched to
raise.
"""

import logging

import torch

import minimaxh3_clipcache.proxy
from minimaxh3_clipcache.proxy import MINIMAX_H3_HIDDEN_DIM, CachedClipProxy

CLIP_NAME = "fake_clip.safetensors"
CLIP_FILE_SIZE = 12345
CLIP_MTIME_NS = 67890


class FakeRealClip:
    def tokenize(self, prompt, **kwargs):
        return ("real_tokens", prompt, kwargs)

    def encode_from_tokens_scheduled(self, tokens):
        return [[torch.zeros(1, MINIMAX_H3_HIDDEN_DIM), {"pooled_output": None}]]


def test_save_failure_still_returns_cond_and_warns(tmp_path, monkeypatch, caplog):
    def boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(minimaxh3_clipcache.proxy, "save_conditioning", boom)

    proxy = CachedClipProxy(
        lambda: FakeRealClip(), CLIP_NAME, CLIP_FILE_SIZE, CLIP_MTIME_NS, tmp_path,
    )

    tokens = proxy.tokenize("a cache miss prompt", images=[])
    with caplog.at_level(logging.WARNING):
        cond = proxy.encode_from_tokens_scheduled(tokens)

    assert torch.equal(cond[0][0], torch.zeros(1, MINIMAX_H3_HIDDEN_DIM))
    assert proxy.did_load_real_clip is True
    assert any(r.levelno == logging.WARNING and "CACHE WRITE FAILED" in r.getMessage()
               for r in caplog.records)
