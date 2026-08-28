"""Two CachedClipProxy instances (as two concurrent Queue runs would create)
racing on encode_from_tokens_scheduled() must not both load the ~27 GB
encoder for the same cache fingerprint: the per-fingerprint lock plus the
in-lock cache re-check means the loser of the race is served the winner's
freshly saved result as a HIT. Distinct fingerprints must still encode
independently and in parallel. No GPU, no ComfyUI, no real clip.
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
            barrier.wait()
            proxy.encode_from_tokens_scheduled(tokens)
        except Exception as e:  # noqa: BLE001 - surface it in the assertion
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(p,)) for p in prompts]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

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
