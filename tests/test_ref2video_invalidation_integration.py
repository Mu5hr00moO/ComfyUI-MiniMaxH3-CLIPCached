"""Integration-level invalidation tests for the Ref2VA cached-CLIP path (R8).

test_fingerprint_ref2video.py already proves compute_fingerprint() itself
reacts to each relevant change on a synthetic minimax_ref_items list. This
file proves the same MISS/HIT decisions survive the real round trip:

    nodes._build_ref_slot_dicts()
        -> stock MiniMaxH3ReferenceToVideo.execute()   (does the ref-image resize)
            -> CachedClipProxy.tokenize()
            -> CachedClipProxy.encode_from_tokens_scheduled()  (fingerprint + cache)

mirroring how nodes.py wires one fresh CachedClipProxy per graph execution.
No GPU, no real encoder, no real VAE -- FakeVAE + FakeRealClip stand-ins,
same pattern as tests/test_invalidation_integration.py (the FL2VA sibling).

The ref-image resize (and therefore the effect of ref_image_size) lives in
the stock node, not in _build_ref_slot_dicts, so these tests run the stock
node for real and only substitute clip + vae.
"""

import importlib.util
import os
import sys

import torch

from comfy_extras.nodes_minimax_h3 import MiniMaxH3ReferenceToVideo
from minimaxh3_clipcache.proxy import MINIMAX_H3_HIDDEN_DIM, CachedClipProxy

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_node_module():
    spec = importlib.util.spec_from_file_location(
        "minimaxh3clipcached_nodes_r8_integration", os.path.join(REPO_ROOT, "nodes.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_build_ref_slot_dicts = _load_node_module()._build_ref_slot_dicts

CLIP_NAME = "fake_clip.safetensors"
CLIP_FILE_SIZE = 12345
CLIP_MTIME_NS = 67890

WIDTH, HEIGHT, LENGTH = 1344, 768, 124


class FakeVAE:
    """Enough of a VAE for the stock ref2va node: encode() returns a
    plausibly-shaped latent, audio_sample_rate for the (here unused) audio
    path. The returned latent only feeds minimax_refs, which is attached
    after the encode and is not part of the cache."""

    audio_sample_rate = 32000

    def encode(self, pixels):
        b, h, w = pixels.shape[0], pixels.shape[1], pixels.shape[2]
        return torch.zeros(b, 4, max(1, h // 16), max(1, w // 16))


class FakeRealClip:
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


def _img(seed, h=512, w=512):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(1, h, w, 3, generator=g)


def _slots_with(pairs):
    """9 image slots, only the given (index, tensor) pairs connected."""
    slots = [None] * 9
    for i, value in pairs:
        slots[i] = value
    return slots


def _run(tmp_path, image_slots, ref_image_size="match", prompt="a ref2va prompt"):
    """One graph execution: a fresh CachedClipProxy (its own loader + call
    counter, like nodes.py per execution), the real stock ref2va node doing
    the resize, the real fingerprint + on-disk cache in tmp_path. Returns the
    loader call count for this instance (0 = HIT, 1 = MISS)."""
    loader, calls = _make_counting_loader()
    proxy = CachedClipProxy(loader, CLIP_NAME, CLIP_FILE_SIZE, CLIP_MTIME_NS, tmp_path)

    ref_images, ref_videos, ref_video_audios, ref_audios = _build_ref_slot_dicts(
        image_slots, [None] * 3, [None] * 3, [None] * 3,
    )
    MiniMaxH3ReferenceToVideo.execute(
        clip=proxy, vae=FakeVAE(), audio_vae=FakeVAE(), prompt=prompt,
        width=WIDTH, height=HEIGHT, length=LENGTH, ref_image_size=ref_image_size,
        ref_images=ref_images, ref_videos=ref_videos,
        ref_video_audios=ref_video_audios, ref_audios=ref_audios,
    )
    return calls["count"]


def test_ref_image_content_and_slot_invalidation_through_full_ref2va_path(tmp_path):
    img_a = _img(1)
    img_b = _img(2)

    # a) populate the cache: one reference image in slot 0 -> MISS
    assert _run(tmp_path, _slots_with([(0, img_a)])) == 1

    # b) new proxy, byte-identical ref_image_0 -> HIT (loader never called)
    assert _run(tmp_path, _slots_with([(0, img_a.clone())])) == 0

    # c) new proxy, different image content in the same slot -> MISS
    assert _run(tmp_path, _slots_with([(0, img_b)])) == 1

    # d) new proxy, ref_image_1 additionally connected (was None) -> MISS
    assert _run(tmp_path, _slots_with([(0, img_a.clone()), (1, img_b.clone())])) == 1


def test_ref_image_size_large_image_match_and_max_are_distinct_entries(tmp_path):
    # A >=1600px reference forces a genuinely different resize between the two
    # modes: for a 1600x1200 source at the 1344x768 canvas, "match" scales to
    # ~1184x864 and "max" to 1600x1216 (verified arithmetically). Different
    # post-resize pixels reach tokenize(), so the two modes must not share a
    # cache entry. (A <=1024px image would resize identically -- see the next
    # test.)
    big = _img(3, h=1200, w=1600)

    assert _run(tmp_path, _slots_with([(0, big)]), ref_image_size="match") == 1
    assert _run(tmp_path, _slots_with([(0, big.clone())]), ref_image_size="max") == 1
    # the "match" entry from the first call is still there and still distinct
    assert _run(tmp_path, _slots_with([(0, big.clone())]), ref_image_size="match") == 0


def test_ref_image_size_small_image_match_and_max_share_one_entry(tmp_path):
    # INTENTIONAL, CORRECT collision -- do NOT "fix" this.
    #
    # For a reference image whose short edge is already at or below the
    # mode-specific target (here 512x512 against the 1344x768 canvas),
    # ref_image_size="match" and ref_image_size="max" compute the SAME
    # target tw/th, so the stock node hands _resize() identical arguments
    # and the post-resize tensor passed to tokenize() is byte-for-byte the
    # same. The encoder would produce an identical result either way, so
    # serving one cached entry for both modes is correct: the cache key
    # reflects what actually reaches the encoder, not the mode label. The
    # large-image test above is where the two modes genuinely diverge.
    small = _img(4, h=512, w=512)

    assert _run(tmp_path, _slots_with([(0, small)]), ref_image_size="match") == 1
    assert _run(tmp_path, _slots_with([(0, small.clone())]), ref_image_size="max") == 0


def test_lone_image_slot_index_is_invisible_and_that_is_a_hit(tmp_path):
    # INTENTIONAL HIT -- do NOT "fix" by making the absolute slot index part
    # of the cache key.
    #
    # The stock node numbers references <Picture i> by iteration order over
    # the PRESENT images, not by their slot index (_build_ref_slot_dicts
    # drops None slots). A lone reference image is <Picture 1> whether it is
    # wired to ref_image_0 or ref_image_1, so the exact same post-resize
    # tensor reaches tokenize() and the fingerprint is identical. A MISS here
    # would be a false miss: a wasted ~27 GB encoder reload to recompute a
    # byte-identical encode.
    img = _img(5)

    assert _run(tmp_path, _slots_with([(0, img)])) == 1
    assert _run(tmp_path, _slots_with([(1, img.clone())])) == 0


def test_swapped_relative_image_order_is_a_miss(tmp_path):
    # The integration-level analog of test_fingerprint_ref2video.py's
    # test_c_reversed_image_order: what the cache key is (correctly)
    # sensitive to is the RELATIVE order of the present images, because that
    # is the order they enter the tokenizer presentation as <Picture 1>,
    # <Picture 2>, ...
    img_a = _img(6)
    img_b = _img(7)

    assert _run(tmp_path, _slots_with([(0, img_a), (1, img_b)])) == 1
    assert _run(tmp_path, _slots_with([(0, img_b.clone()), (1, img_a.clone())])) == 1
