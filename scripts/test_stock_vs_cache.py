"""Phase 23: stock == cached-MISS == cached-HIT equivalence, plus a
downstream MiniMaxH3AddGuide chain, against the REAL Qwen3-VL MiniMax H3
encoder. Standalone script (no ComfyUI server -- fewer moving parts to run
unsupervised), meant to run under a hard external timeout.

Fully sequential -- ONE resident ~27GB encoder at a time, exactly like the
production node (nodes.py) does. The first attempt at this script loaded (a)
directly via nodes.CLIPLoader() and then handed (b)/(c) a lambda that reused
that same already-loaded object -- so it never actually exercised the real
MISS path (comfy.sd.load_clip() is only called by CachedClipProxy on a real
MISS, via caching.loader.build_clip_loader_fn(), never through a hand-rolled
lambda). Confirmed root cause of the resulting OOM crash (see CLAUDE.md):
comfy.sd.load_clip() does not dedupe -- an independent load call for the
same file produces a second, fully resident model object. This version calls
build_clip_loader_fn() for real in (b)/(c), exactly like nodes.py does, and
unloads immediately after each step that actually loaded a real clip, before
the next step starts:
  (a) direct nodes.CLIPLoader() load -> execute() -> unload immediately after
  (b) fresh CachedClipProxy, empty cache_dir -> MISS (real load+encode+save)
      -> unload immediately after (proxy.did_load_real_clip is True)
  (c) fresh CachedClipProxy, same cache_dir as (b) -> HIT (served from disk,
      proxy.did_load_real_clip stays False -- nothing to unload)
  (d) AddGuide chain on (a) and (c) -- does not touch clip at all, safe
      regardless of encoder residency state

Each step's output is moved to CPU right after it returns, so the later
comparisons never depend on what is or isn't still resident on the GPU.

(a)/(b)/(c) outputs compared pairwise via caching.comparison._tensors_equal
(torch.equal, not allclose -- (c) must replay the exact cached bytes, not a
fresh numerically-close encode).

try/finally is kept as a safety net only: with the sequential unload-after-
each-step structure above, finally should have nothing left to clean up on a
normal run. unload_model_and_clones() on an already-unloaded model is a
documented no-op (see CLAUDE.md), so calling it again there is harmless.

A background thread appends `free -h` + `nvidia-smi` to
/tmp/phase23_memory.log every ~15s for the duration of steps 2-4 -- for this
run's own record and as raw data for phase 24's memory investigation.

Run with the comfyenv conda environment from anywhere, under a hard timeout
since this is meant to run unsupervised:
    timeout 900 conda run -n comfyenv --no-capture-output python scripts/test_stock_vs_cache.py
"""

import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_proxy_gate import (  # noqa: E402
    CLIP_NAME, CLIP_TYPE, VAE_NAME, PROMPT, WIDTH, HEIGHT, LENGTH, log_memory,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# append, NOT insert(0): REPO_ROOT contains our own nodes.py (the public
# MiniMaxH3CLIPCachedImageToVideo node, added phase 18) -- inserting it ahead of
# COMFYUI_ROOT would make a later `import nodes` inside main() resolve to
# THIS repo's nodes.py instead of ComfyUI's own, exactly the collision
# __init__.py already documents avoiding the same way (see CLAUDE.md).
sys.path.append(REPO_ROOT)  # for `caching.proxy` / `caching.comparison` / `caching.loader`

from caching.comparison import _tensors_equal  # noqa: E402

CACHE_DIR = Path(REPO_ROOT) / "cache" / "test_stock_vs_cache"
MEMORY_LOG_PATH = Path("/tmp/phase23_memory.log")
MEMORY_LOG_INTERVAL_S = 15
GUIDE_FRAME_IDX = 60  # well inside the 124-frame latent, away from the first_frame keyframe at 0


def _run_capture(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout
    except Exception as e:
        return "[{} failed: {}]\n".format(" ".join(cmd), e)


def memory_logger(stop_event):
    with open(MEMORY_LOG_PATH, "a") as f:
        while not stop_event.is_set():
            f.write("=== {} ===\n".format(time.strftime("%Y-%m-%d %H:%M:%S")))
            f.write(_run_capture(["free", "-h"]))
            f.write(_run_capture(["nvidia-smi"]))
            f.write("\n")
            f.flush()
            stop_event.wait(MEMORY_LOG_INTERVAL_S)


def _to_cpu(obj):
    """Recursively move every tensor in a conditioning/latent structure to
    CPU, so later comparisons never depend on current GPU state (a model
    unloaded between steps, a different step's tensors still resident,
    etc). Handles the shapes MiniMaxH3ImageToVideo/AddGuide actually
    produce: nested list/tuple/dict, comfy.nested_tensor.NestedTensor
    (video+audio AV latent), plain torch.Tensor, and passthrough scalars."""
    import comfy.nested_tensor
    import torch

    if isinstance(obj, comfy.nested_tensor.NestedTensor):
        return comfy.nested_tensor.NestedTensor(tuple(t.detach().cpu() for t in obj.tensors))
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_cpu(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_to_cpu(v) for v in obj)
    return obj


def main():
    import torch
    import nodes
    import comfy.model_management
    from comfy_extras.nodes_minimax_h3 import MiniMaxH3ImageToVideo, MiniMaxH3AddGuide
    from caching.proxy import CachedClipProxy
    from caching.loader import build_clip_loader_fn, resolve_clip_stat

    if CACHE_DIR.exists():
        shutil.rmtree(CACHE_DIR)
    CACHE_DIR.mkdir(parents=True)
    print("cache_dir: {}".format(CACHE_DIR))
    print("memory log: {}".format(MEMORY_LOG_PATH))

    clip_file_size, clip_mtime_ns = resolve_clip_stat(CLIP_NAME)
    print("clip identity: name={} file_size={} mtime_ns={}".format(CLIP_NAME, clip_file_size, clip_mtime_ns))

    print("=== Load real VAE once (kept resident for the whole run -- small compared to the encoder) ===")
    log_memory("before-vae-load")
    vae_loader = nodes.VAELoader()
    (vae,) = vae_loader.load_vae(VAE_NAME)
    log_memory("after-vae-load")

    # Deterministic test images: IMAGE convention is [B, H, W, C], float32 in [0, 1].
    torch.manual_seed(0)
    test_first_frame = torch.rand(1, 400, 600, 3, dtype=torch.float32)
    torch.manual_seed(1)
    guide_image = torch.rand(1, 300, 500, 3, dtype=torch.float32)

    stop_event = threading.Event()
    logger_thread = threading.Thread(target=memory_logger, args=(stop_event,), daemon=True)
    logger_thread.start()
    print("=== Memory logger started -> {} (every {}s) ===".format(MEMORY_LOG_PATH, MEMORY_LOG_INTERVAL_S))

    # Tracks whatever real clip is currently resident, so `finally` can act
    # as a safety net if an exception interrupts the sequence before its own
    # local unload runs. unload_model_and_clones() on an already-unloaded
    # model is a documented no-op (CLAUDE.md), so re-unloading here on the
    # normal path is harmless.
    currently_resident_patcher = None

    try:
        print()
        print("=== (a) direct nodes.CLIPLoader() load -> execute() -> unload immediately ===")
        log_memory("before-a-load")
        t0 = time.time()
        clip_loader = nodes.CLIPLoader()
        (real_clip_a,) = clip_loader.load_clip(CLIP_NAME, type=CLIP_TYPE)
        currently_resident_patcher = real_clip_a.patcher
        print("(a) CLIP loaded in {:.1f}s".format(time.time() - t0))
        log_memory("after-a-load")

        t0 = time.time()
        output_a = MiniMaxH3ImageToVideo.execute(
            clip=real_clip_a, vae=vae, prompt=PROMPT, width=WIDTH, height=HEIGHT, length=LENGTH,
            first_frame=test_first_frame.clone(),
        )
        dt_a = time.time() - t0
        print("(a) execute() finished in {:.1f}s".format(dt_a))
        log_memory("after-a-execute")

        cond_a, latent_a = output_a.args
        cond_a, latent_a = _to_cpu(cond_a), _to_cpu(latent_a)

        comfy.model_management.unload_model_and_clones(real_clip_a.patcher)
        currently_resident_patcher = None
        del real_clip_a
        log_memory("after-a-unload")

        print()
        print("=== (b) fresh CachedClipProxy, empty cache_dir, REAL build_clip_loader_fn -- expect MISS ===")
        proxy_b = CachedClipProxy(
            build_clip_loader_fn(CLIP_NAME), CLIP_NAME, clip_file_size, clip_mtime_ns, CACHE_DIR,
        )
        t0 = time.time()
        output_b = MiniMaxH3ImageToVideo.execute(
            clip=proxy_b, vae=vae, prompt=PROMPT, width=WIDTH, height=HEIGHT, length=LENGTH,
            first_frame=test_first_frame.clone(),
        )
        dt_b = time.time() - t0
        print("(b) finished in {:.1f}s -- did_load_real_clip={}".format(dt_b, proxy_b.did_load_real_clip))
        assert proxy_b.did_load_real_clip is True, "(b) MISS should have loaded the real clip"
        log_memory("after-b-execute")

        cond_b, latent_b = output_b.args
        cond_b, latent_b = _to_cpu(cond_b), _to_cpu(latent_b)

        if proxy_b.did_load_real_clip:
            currently_resident_patcher = proxy_b.real_clip.patcher
            comfy.model_management.unload_model_and_clones(proxy_b.real_clip.patcher)
            currently_resident_patcher = None
        log_memory("after-b-unload")

        print()
        print("=== (c) fresh CachedClipProxy, same cache_dir as (b) -- expect HIT, nothing to unload ===")
        proxy_c = CachedClipProxy(
            build_clip_loader_fn(CLIP_NAME), CLIP_NAME, clip_file_size, clip_mtime_ns, CACHE_DIR,
        )
        t0 = time.time()
        output_c = MiniMaxH3ImageToVideo.execute(
            clip=proxy_c, vae=vae, prompt=PROMPT, width=WIDTH, height=HEIGHT, length=LENGTH,
            first_frame=test_first_frame.clone(),
        )
        dt_c = time.time() - t0
        print("(c) finished in {:.1f}s -- did_load_real_clip={}".format(dt_c, proxy_c.did_load_real_clip))
        assert proxy_c.did_load_real_clip is False, "(c) HIT must NOT load the real clip at all"
        log_memory("after-c-execute")

        cond_c, latent_c = output_c.args
        cond_c, latent_c = _to_cpu(cond_c), _to_cpu(latent_c)

        print()
        print("=== Comparing (a) stock vs (b) MISS vs (c) HIT: exact torch.equal, not allclose ===")
        _tensors_equal("(a) vs (b)", (cond_a, latent_a), (cond_b, latent_b))
        print("(a) == (b): PASS")
        _tensors_equal("(a) vs (c)", (cond_a, latent_a), (cond_c, latent_c))
        print("(a) == (c): PASS")
        _tensors_equal("(b) vs (c)", (cond_b, latent_b), (cond_c, latent_c))
        print("(b) == (c): PASS")

        print()
        print("=== (d) Chaining (a) stock and (c) HIT conditioning through the stock MiniMaxH3AddGuide ===")
        print("(AddGuide never touches clip -- safe regardless of encoder residency state)")
        # (d) is isolated in its own try/except: a/b/c are the actual cache
        # correctness result (this project's whole point) and must always be
        # reported in full even if (d) -- an extra downstream-compatibility
        # check that does a further real vae.encode() on top of three prior
        # real encode passes -- hits a VRAM ceiling unrelated to cache
        # correctness (see CLAUDE.md "Otwarte pytania - faza 24"). A crash
        # here must not swallow the a/b/c PASS lines above it into an
        # unreadable traceback.
        d_failed = False
        dt_guide_stock = dt_guide_hit = None
        try:
            t0 = time.time()
            guided_stock = MiniMaxH3AddGuide.execute(
                positive=cond_a, latent=latent_a, frame_idx=GUIDE_FRAME_IDX, vae=vae,
                image=guide_image.clone(),
            )
            dt_guide_stock = time.time() - t0
            print("AddGuide(stock) finished in {:.1f}s".format(dt_guide_stock))

            t0 = time.time()
            guided_hit = MiniMaxH3AddGuide.execute(
                positive=cond_c, latent=latent_c, frame_idx=GUIDE_FRAME_IDX, vae=vae,
                image=guide_image.clone(),
            )
            dt_guide_hit = time.time() - t0
            print("AddGuide(HIT) finished in {:.1f}s".format(dt_guide_hit))

            (guided_cond_stock,) = guided_stock.args
            (guided_cond_hit,) = guided_hit.args
            guided_cond_stock, guided_cond_hit = _to_cpu(guided_cond_stock), _to_cpu(guided_cond_hit)
            _tensors_equal("AddGuide(stock) vs AddGuide(HIT)", guided_cond_stock, guided_cond_hit)
            print("AddGuide(stock) == AddGuide(HIT): PASS")
        except Exception:
            d_failed = True
            print()
            print("=== (d) FAILED -- full traceback below, a/b/c result above stands regardless ===")
            traceback.print_exc()

        print()
        print("=== Timing summary ===")
        print("(a) stock          : {:.1f}s".format(dt_a))
        print("(b) MISS           : {:.1f}s".format(dt_b))
        print("(c) HIT            : {:.1f}s".format(dt_c))
        print("AddGuide(stock)    : {}".format("{:.1f}s".format(dt_guide_stock) if dt_guide_stock is not None else "FAILED/skipped"))
        print("AddGuide(HIT)      : {}".format("{:.1f}s".format(dt_guide_hit) if dt_guide_hit is not None else "FAILED/skipped"))

        print()
        if d_failed:
            print("=== RESULT: a/b/c PASS, (d) AddGuide FAILED (see traceback above) ===")
        else:
            print("=== RESULT: PASS ===")

    finally:
        print()
        print("=== Safety-net unload (should be a no-op on a normal run -- see module docstring) ===")
        stop_event.set()
        log_memory("before-finally-unload")
        if currently_resident_patcher is not None:
            comfy.model_management.unload_model_and_clones(currently_resident_patcher)
        log_memory("after-finally-unload")
        logger_thread.join(timeout=5)

    if d_failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
