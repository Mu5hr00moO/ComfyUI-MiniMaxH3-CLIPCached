"""Two CachedClipProxy instances (as two concurrent Queue runs would create)
racing on encode_from_tokens_scheduled() must not both load the ~27 GB
encoder for the same cache fingerprint: the per-fingerprint lock plus the
in-lock cache re-check means the loser of the race is served the winner's
freshly saved result as a HIT. Distinct fingerprints must still each get
their own independent load+encode (never merged, never blocked forever),
but a process-wide encoder lock now serialises the actual load+encode
step across fingerprints too, so two real ~27 GB encoders are never
resident at the same time -- they no longer literally overlap in
wall-clock time. No GPU, no ComfyUI, no real clip.
"""

import threading
import time

import torch

from minimaxh3_clipcache.proxy import MINIMAX_H3_HIDDEN_DIM, CachedClipProxy

CLIP_NAME = "fake_clip.safetensors"
CLIP_FILE_SIZE = 12345
CLIP_MTIME_NS = 67890


class FakeRealClip:
    def __init__(self):
        self.encode_calls = 0

    def tokenize(self, prompt, **kwargs):
        return ("real_tokens", prompt, kwargs)

    def encode_from_tokens_scheduled(self, tokens):
        self.encode_calls += 1
        # widen the window so both threads are genuinely in flight
        time.sleep(0.05)
        return [[torch.zeros(1, MINIMAX_H3_HIDDEN_DIM), {"pooled_output": None}]]


def _race_two_proxies(prompts, tmp_path):
    """Start two threads, each with its own proxy but a shared loader/encoder,
    and have them call encode_from_tokens_scheduled() at the same time."""
    real_clip = FakeRealClip()
    counter_guard = threading.Lock()
    loader_calls = {"count": 0}

    def loader():
        with counter_guard:
            loader_calls["count"] += 1
        return real_clip

    barrier = threading.Barrier(len(prompts))
    errors = []

    def worker(prompt):
        try:
            proxy = CachedClipProxy(
                loader, CLIP_NAME, CLIP_FILE_SIZE, CLIP_MTIME_NS, tmp_path)
            tokens = proxy.tokenize(prompt, images=[])
            barrier.wait(timeout=10)
            proxy.encode_from_tokens_scheduled(tokens)
        except Exception as e:  # noqa: BLE001 - surface it in the assertion
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(p,)) for p in prompts]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
        assert not t.is_alive(), "worker thread did not finish within 30s (deadlock?)"

    assert not errors, "worker thread raised: {}".format(errors)
    return loader_calls["count"], real_clip.encode_calls


def test_same_fingerprint_race_loads_encoder_exactly_once(tmp_path):
    loader_count, encode_count = _race_two_proxies(
        ["identical prompt", "identical prompt"], tmp_path)
    assert loader_count == 1
    assert encode_count == 1


def test_distinct_fingerprints_race_encode_independently(tmp_path):
    loader_count, encode_count = _race_two_proxies(
        ["prompt one", "prompt two"], tmp_path)
    assert loader_count == 2
    assert encode_count == 2


def test_distinct_fingerprints_never_run_encode_concurrently(tmp_path):
    """The per-fingerprint lock alone only stops the SAME fingerprint
    loading twice; two DIFFERENT fingerprints racing a MISS must never
    have two real encoders active (loaded and encoding) at the same
    time. Tracks how many are concurrently inside encode_from_tokens_scheduled()
    and asserts it never exceeds 1."""
    guard = threading.Lock()
    state = {"active": 0, "max_active": 0}

    class TrackingRealClip:
        def __init__(self):
            self.patcher = "fake_patcher"

        def tokenize(self, prompt, **kwargs):
            return ("real_tokens", prompt, kwargs)

        def encode_from_tokens_scheduled(self, tokens):
            with guard:
                state["active"] += 1
                state["max_active"] = max(state["max_active"], state["active"])
            time.sleep(0.05)
            with guard:
                state["active"] -= 1
            return [[torch.zeros(1, MINIMAX_H3_HIDDEN_DIM), {"pooled_output": None}]]

    barrier = threading.Barrier(2)
    errors = []

    def worker(prompt):
        try:
            proxy = CachedClipProxy(
                lambda: TrackingRealClip(), CLIP_NAME, CLIP_FILE_SIZE, CLIP_MTIME_NS, tmp_path,
            )
            tokens = proxy.tokenize(prompt, images=[])
            barrier.wait(timeout=10)
            proxy.encode_from_tokens_scheduled(tokens)
        except Exception as e:  # noqa: BLE001 - surface it in the assertion
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(p,)) for p in ("prompt one", "prompt two")]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
        assert not t.is_alive(), "worker thread did not finish within 30s (deadlock?)"

    assert not errors, "worker thread raised: {}".format(errors)
    assert state["max_active"] == 1
