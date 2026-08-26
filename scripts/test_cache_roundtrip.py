"""Phase 12 step 2/2 test: CachedClipProxy end-to-end MISS -> HIT -> MISS.

Loads the real MiniMax H3 clip/vae once, then runs
MiniMaxH3ImageToVideo.execute() three times against a fresh empty cache_dir:
  (a) first call, variant A (no keyframes)             -> expect MISS
  (b) same prompt/inputs, a NEW CachedClipProxy instance -> expect HIT,
      and the real clip must not be touched at all (verified via a
      call-counting wrapper, not just timing)
  (c) different prompt                                  -> expect MISS again

No cache logic reimplemented here -- this only wires the already-tested
caching.proxy.CachedClipProxy into the real stock node and the real model.

Run with the comfyenv conda environment from anywhere:
    conda run -n comfyenv python scripts/test_cache_roundtrip.py
"""

import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_proxy_gate import (  # noqa: E402
    CLIP_NAME, CLIP_TYPE, VAE_NAME, PROMPT, WIDTH, HEIGHT, LENGTH, log_memory,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)  # for `caching.proxy` / `caching.comparison`
CACHE_DIR = Path(REPO_ROOT) / "cache" / "test_roundtrip"

from caching.comparison import _tensors_equal  # noqa: E402


class CallCountingClip:
    """Wraps the real clip and counts tokenize()/encode_from_tokens_scheduled()
    calls, so a HIT can be verified by call count, not just by timing."""

    def __init__(self, real_clip):
        self._real_clip = real_clip
        self.tokenize_calls = 0
        self.encode_calls = 0

    def tokenize(self, prompt, **kwargs):
        self.tokenize_calls += 1
        return self._real_clip.tokenize(prompt, **kwargs)

    def encode_from_tokens_scheduled(self, tokens):
        self.encode_calls += 1
        return self._real_clip.encode_from_tokens_scheduled(tokens)


def main():
    import nodes
    import comfy.model_management
    from comfy_extras.nodes_minimax_h3 import MiniMaxH3ImageToVideo
    from caching.proxy import CachedClipProxy

    if CACHE_DIR.exists():
        shutil.rmtree(CACHE_DIR)
    CACHE_DIR.mkdir(parents=True)
    print("cache_dir: {}".format(CACHE_DIR))

    print("=== Load real CLIP via stock nodes.CLIPLoader (once) ===")
    log_memory("before-clip-load")
    t0 = time.time()
    clip_loader = nodes.CLIPLoader()
    (clip,) = clip_loader.load_clip(CLIP_NAME, type=CLIP_TYPE)
    print("CLIP loaded in {:.1f}s".format(time.time() - t0))
    log_memory("after-clip-load")

    print("=== Load real VAE via stock nodes.VAELoader (once) ===")
    vae_loader = nodes.VAELoader()
    (vae,) = vae_loader.load_vae(VAE_NAME)
    log_memory("after-vae-load")

    clip_path = os.path.join("/home/kamil/ComfyUI", "models", "text_encoders", CLIP_NAME)
    st = os.stat(clip_path)
    clip_file_size, clip_mtime_ns = st.st_size, st.st_mtime_ns
    print("clip identity: name={} file_size={} mtime_ns={}".format(CLIP_NAME, clip_file_size, clip_mtime_ns))

    counting_clip = CallCountingClip(clip)

    print()
    print("=== (a) execute() with fresh CachedClipProxy, empty cache_dir -> expect MISS ===")
    proxy_a = CachedClipProxy(counting_clip, CLIP_NAME, clip_file_size, clip_mtime_ns, CACHE_DIR)
    t0 = time.time()
    output_a = MiniMaxH3ImageToVideo.execute(
        clip=proxy_a, vae=vae, prompt=PROMPT, width=WIDTH, height=HEIGHT, length=LENGTH,
    )
    dt_a = time.time() - t0
    print("(a) finished in {:.1f}s -- tokenize_calls={} encode_calls={}".format(
        dt_a, counting_clip.tokenize_calls, counting_clip.encode_calls))
    assert counting_clip.tokenize_calls == 1 and counting_clip.encode_calls == 1, \
        "(a) MISS should call the real clip exactly once each"

    print()
    print("=== (b) execute() with a NEW CachedClipProxy, same cache_dir/prompt -> expect HIT ===")
    proxy_b = CachedClipProxy(counting_clip, CLIP_NAME, clip_file_size, clip_mtime_ns, CACHE_DIR)
    t0 = time.time()
    output_b = MiniMaxH3ImageToVideo.execute(
        clip=proxy_b, vae=vae, prompt=PROMPT, width=WIDTH, height=HEIGHT, length=LENGTH,
    )
    dt_b = time.time() - t0
    print("(b) finished in {:.1f}s -- tokenize_calls={} encode_calls={}".format(
        dt_b, counting_clip.tokenize_calls, counting_clip.encode_calls))
    assert counting_clip.tokenize_calls == 1 and counting_clip.encode_calls == 1, \
        "(b) HIT must NOT call the real clip at all -- counts must stay at 1"

    print("--- comparing (a) vs (b) output: exact torch.equal, not allclose ---")
    _tensors_equal("output", output_a.args, output_b.args)
    print("(a) == (b): PASS (exact match, cache hit served identical conditioning)")

    print()
    print("=== (c) execute() with a different prompt, same cache_dir -> expect MISS again ===")
    proxy_c = CachedClipProxy(counting_clip, CLIP_NAME, clip_file_size, clip_mtime_ns, CACHE_DIR)
    t0 = time.time()
    output_c = MiniMaxH3ImageToVideo.execute(
        clip=proxy_c, vae=vae, prompt=PROMPT + " (different)", width=WIDTH, height=HEIGHT, length=LENGTH,
    )
    dt_c = time.time() - t0
    print("(c) finished in {:.1f}s -- tokenize_calls={} encode_calls={}".format(
        dt_c, counting_clip.tokenize_calls, counting_clip.encode_calls))
    assert counting_clip.tokenize_calls == 2 and counting_clip.encode_calls == 2, \
        "(c) MISS (different prompt) should call the real clip again"

    print()
    print("=== Timing summary ===")
    print("(a) MISS : {:.1f}s".format(dt_a))
    print("(b) HIT  : {:.1f}s".format(dt_b))
    print("(c) MISS : {:.1f}s".format(dt_c))

    print()
    print("=== Unload real clip (once, after all three runs) ===")
    log_memory("before-unload")
    comfy.model_management.unload_model_and_clones(clip.patcher)
    log_memory("after-unload")

    print()
    print("=== ROUNDTRIP RESULT: PASS ===")


if __name__ == "__main__":
    main()
