"""Integration-level invalidation tests for CachedClipProxy (phase 21).

test_fingerprint.py already proves compute_fingerprint() itself is sensitive
to each of these inputs. This file proves the same MISS/HIT decisions
actually propagate end to end through CachedClipProxy.tokenize() ->
encode_from_tokens_scheduled() -> cache lookup -> (real encode on MISS),
mirroring how nodes.py builds one fresh CachedClipProxy instance per graph
execution. No GPU, no ComfyUI, no real clip -- FakeRealClip stand-in, same
pattern as test_proxy_laziness.py.
"""

import torch

from minimaxh3_clipcache.proxy import MINIMAX_H3_HIDDEN_DIM, CachedClipProxy

CLIP_NAME = "fake_clip.safetensors"
CLIP_FILE_SIZE = 12345
CLIP_MTIME_NS = 67890
CLIP_CTIME_NS = 98765


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


def _run(tmp_path, prompt, images, clip_name=CLIP_NAME, clip_file_size=CLIP_FILE_SIZE,
         clip_mtime_ns=CLIP_MTIME_NS, clip_ctime_ns=CLIP_CTIME_NS):
    """One fresh CachedClipProxy instance (own loader, own call counter) --
    same as nodes.py creating a new proxy on every node execution. Returns
    how many times this instance's loader was called (0 = HIT, 1 = MISS)."""
    loader, calls = _make_counting_loader()
    proxy = CachedClipProxy(
        loader, clip_name, clip_file_size, clip_mtime_ns, tmp_path,
        clip_ctime_ns=clip_ctime_ns,
    )
    tokens = proxy.tokenize(prompt, images=images)
    proxy.encode_from_tokens_scheduled(tokens)
    return calls["count"]


def test_invalidation_scenario_through_full_proxy(tmp_path):
    first_frame = torch.rand(1, 64, 64, 3)
    last_frame = torch.rand(1, 64, 64, 3)

    # a) populate the cache via a fresh proxy, prompt="A"
    assert _run(tmp_path, "A", [first_frame, last_frame]) == 1

    # b) new proxy instance, identical prompt+images -> HIT, its own loader
    #    (starting at 0) is never called
    assert _run(tmp_path, "A", [first_frame.clone(), last_frame.clone()]) == 0

    # c) new proxy instance, different prompt -> MISS
    assert _run(tmp_path, "B", [first_frame.clone(), last_frame.clone()]) == 1

    # d) new proxy instance, prompt="A" but a different first_frame -> MISS
    other_first_frame = torch.rand(1, 64, 64, 3)
    assert _run(tmp_path, "A", [other_first_frame, last_frame.clone()]) == 1

    # e) new proxy instance, prompt="A", same images, different clip_name -> MISS
    assert _run(tmp_path, "A", [first_frame.clone(), last_frame.clone()],
                clip_name="other_clip.safetensors") == 1

    # f) new proxy instance, prompt="A", same clip_name, different
    #    clip_file_size (simulated model file swap) -> MISS
    assert _run(tmp_path, "A", [first_frame.clone(), last_frame.clone()],
                clip_file_size=CLIP_FILE_SIZE + 1) == 1

    # g) same name/size/mtime, but metadata-change time moved after a
    # replacement that restored the original mtime -> MISS
    assert _run(tmp_path, "A", [first_frame.clone(), last_frame.clone()],
                clip_ctime_ns=CLIP_CTIME_NS + 1) == 1
