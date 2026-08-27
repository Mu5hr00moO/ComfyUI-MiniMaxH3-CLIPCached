"""Phase 24 step 2: isolate whether the VRAM that fails to come back after
CachedClipProxy's cache MISS path is (1) unload_model_and_clones() itself
not coping with a clip loaded through a closure (clip_loader_fn) the way it
copes with a directly-loaded clip, or (2) an ordinary live Python reference
-- through the proxy object itself -- keeping the model reachable and
blocking the allocator's reclaim, fixable with plain del+gc.collect().

Two variants, 3 iterations each, real Qwen3-VL MiniMax H3 encoder, no VAE
involved at all (kept out on purpose to isolate CLIP-only behaviour):

  A) direct clip, exactly like step (a) in scripts/test_stock_vs_cache.py:
     nodes.CLIPLoader() -> one real tokenize()+encode_from_tokens_scheduled()
     to force actual GPU weight transfer -> unload_model_and_clones() ->
     log VRAM -> del real_clip; gc.collect(); torch.cuda.empty_cache() ->
     log VRAM again.

  B) through CachedClipProxy, exactly like step (b), with force_refresh=True
     so every iteration is a real MISS (load+encode), never served from a
     prior iteration's cache entry: CachedClipProxy(build_clip_loader_fn(...))
     -> proxy.tokenize()+proxy.encode_from_tokens_scheduled() -> if
     proxy.did_load_real_clip: unload_model_and_clones(proxy.real_clip.patcher)
     -> log VRAM (TEST A: right after unload, BEFORE any del) -> del proxy;
     gc.collect(); torch.cuda.empty_cache() -> log VRAM again.

Both variants log VRAM at the same two checkpoints (after-unload-before-del,
after-del-and-gc), so the two checkpoint columns are directly comparable
between A and B in the final table.

Phase 24 step 2 RETRY (previous attempt was manually SIGKILLed by the user
via nvitop after host RAM crossed 100% and started swapping -- fully
explained, not a mystery; see CLAUDE.md). Hardening added for this retry:
  - A background watchdog thread (daemon, started before ANY model load)
    polls RAM and VRAM every 2s. RAM >=75% logs a warning; RAM >=85% or
    VRAM reserved >=90% of device total calls os._exit(1) immediately --
    unconditional, uncatchable by any try/except, so a runaway can't out-run
    it the way the manual SIGKILL had to.
  - _log_step() now also logs system MemAvailable and this process's own
    VmRSS (both from /proc), not just VRAM -- the previous version had no
    RAM visibility at all, which is exactly why the RAM blowup that led to
    the manual kill left no trace in this script's own log.
  - This run is intentionally scoped down to ONLY 1 iteration of Variant A;
    Variant B is not called. See RUN_VARIANT_B below -- flip it back to
    re-enable the full run once a single load/unload/del/gc cycle has been
    observed to be safe.
  - Must be launched with PYTHONUNBUFFERED=1 (or `python -u`) so the log
    file actually contains output even if the process is killed hard again.

Run with the comfyenv conda environment from anywhere, under a hard timeout:
    timeout 300 conda run -n comfyenv --no-capture-output python -u scripts/test_clip_unload_isolation.py
"""

import gc
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_proxy_gate import CLIP_NAME, CLIP_TYPE, PROMPT, log_memory  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# append, NOT insert(0): REPO_ROOT contains our own nodes.py -- see
# scripts/test_stock_vs_cache.py's docstring / CLAUDE.md for why insert(0)
# here would shadow ComfyUI's own nodes.py on a later `import nodes`.
sys.path.append(REPO_ROOT)  # for `minimaxh3_clipcache.proxy` / `minimaxh3_clipcache.loader`

CACHE_DIR = Path(REPO_ROOT) / "cache" / "test_clip_unload_isolation"
VARIANT_A_ITERATIONS = 1  # deliberately reduced for this retry, see module docstring
VARIANT_B_ITERATIONS = 3  # unused this run -- RUN_VARIANT_B is False
RUN_VARIANT_B = False  # flip to True once one A cycle is confirmed safe

RAM_WARN_PCT = 75.0
RAM_HARD_STOP_PCT = 85.0
VRAM_HARD_STOP_FRACTION = 0.90
WATCHDOG_INTERVAL_S = 2


def _read_proc_field_kb(path, field):
    try:
        with open(path) as f:
            for line in f:
                if line.startswith(field + ":"):
                    return int(line.split()[1])  # value in kB
    except Exception:
        pass
    return None


def _ram_percent():
    try:
        import psutil
        return psutil.virtual_memory().percent
    except ImportError:
        total = _read_proc_field_kb("/proc/meminfo", "MemTotal")
        avail = _read_proc_field_kb("/proc/meminfo", "MemAvailable")
        if total and avail:
            return (1.0 - avail / total) * 100.0
        return 0.0


def _vram_reserved_fraction():
    import torch
    if not torch.cuda.is_available():
        return 0.0
    reserved = torch.cuda.memory_reserved()
    total = torch.cuda.get_device_properties(0).total_memory
    return reserved / total if total else 0.0


def watchdog_loop(stop_event):
    """Runs in its own daemon thread for the whole process lifetime, started
    before any model load. Deliberately uses os._exit(1), not sys.exit() or
    a raised exception: this must be a hard, unconditional stop that no
    try/except anywhere in the main thread's code can intercept or delay."""
    warned = False
    while not stop_event.is_set():
        ram_pct = _ram_percent()
        if ram_pct >= RAM_HARD_STOP_PCT:
            print("!!! WATCHDOG: RAM {:.1f}% >= {:.0f}% -- HARD STOP via os._exit(1) !!!".format(
                ram_pct, RAM_HARD_STOP_PCT), flush=True)
            os._exit(1)
        elif ram_pct >= RAM_WARN_PCT and not warned:
            print("--- WATCHDOG WARNING: RAM at {:.1f}% (>= {:.0f}%) ---".format(ram_pct, RAM_WARN_PCT), flush=True)
            warned = True
        elif ram_pct < RAM_WARN_PCT:
            warned = False

        vram_frac = _vram_reserved_fraction()
        if vram_frac >= VRAM_HARD_STOP_FRACTION:
            print("!!! WATCHDOG: VRAM reserved {:.1f}% >= {:.0f}% of device total -- HARD STOP via os._exit(1) !!!".format(
                vram_frac * 100.0, VRAM_HARD_STOP_FRACTION * 100.0), flush=True)
            os._exit(1)

        stop_event.wait(WATCHDOG_INTERVAL_S)


def _nvidia_smi_used_total():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        used_str, total_str = [s.strip() for s in out.split(",")]
        return int(used_str), int(total_str)
    except Exception as e:
        return None, "error: {}".format(e)


def _log_step(rows, variant, iteration, checkpoint):
    import torch

    allocated = torch.cuda.memory_allocated() / (1024 ** 3)
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)
    used_mib, total_mib = _nvidia_smi_used_total()
    mem_available_kb = _read_proc_field_kb("/proc/meminfo", "MemAvailable")
    own_rss_kb = _read_proc_field_kb("/proc/self/status", "VmRSS")
    label = "{}/iter{}/{}".format(variant, iteration, checkpoint)
    print(
        "[{}] pytorch allocated={:.2f}GiB reserved={:.2f}GiB | nvidia-smi used={}MiB total={}MiB | "
        "MemAvailable={}kB own_RSS={}kB".format(
            label, allocated, reserved, used_mib, total_mib, mem_available_kb, own_rss_kb),
        flush=True,
    )
    rows.append({
        "variant": variant, "iteration": iteration, "checkpoint": checkpoint,
        "pytorch_allocated_gib": allocated, "pytorch_reserved_gib": reserved,
        "nvidia_smi_used_mib": used_mib, "nvidia_smi_total_mib": total_mib,
        "mem_available_kb": mem_available_kb, "own_rss_kb": own_rss_kb,
    })
    return rows


def run_variant_a(rows):
    import torch
    import nodes
    import comfy.model_management

    print()
    print("=== VARIANT A: direct clip, {} iteration(s) ===".format(VARIANT_A_ITERATIONS))
    for i in range(1, VARIANT_A_ITERATIONS + 1):
        print("--- A iteration {} ---".format(i))
        t0 = time.time()
        clip_loader = nodes.CLIPLoader()
        (real_clip,) = clip_loader.load_clip(CLIP_NAME, type=CLIP_TYPE)
        tokens = real_clip.tokenize(PROMPT, images=[])
        cond = real_clip.encode_from_tokens_scheduled(tokens)
        print("A/iter{}: load+tokenize+encode finished in {:.1f}s".format(i, time.time() - t0))
        del cond, tokens

        comfy.model_management.unload_model_and_clones(real_clip.patcher)
        rows = _log_step(rows, "A", i, "after-unload-before-del")

        del real_clip
        gc.collect()
        torch.cuda.empty_cache()
        rows = _log_step(rows, "A", i, "after-del-and-gc")

    return rows


def run_variant_b(rows):
    import torch
    import comfy.model_management
    from minimaxh3_clipcache.proxy import CachedClipProxy
    from minimaxh3_clipcache.loader import build_clip_loader_fn, resolve_clip_stat

    if CACHE_DIR.exists():
        shutil.rmtree(CACHE_DIR)
    CACHE_DIR.mkdir(parents=True)

    clip_file_size, clip_mtime_ns = resolve_clip_stat(CLIP_NAME)

    print()
    print("=== VARIANT B: CachedClipProxy, force_refresh=True (real MISS every time), {} iterations ===".format(
        VARIANT_B_ITERATIONS))
    for i in range(1, VARIANT_B_ITERATIONS + 1):
        print("--- B iteration {} ---".format(i))
        t0 = time.time()
        proxy = CachedClipProxy(
            build_clip_loader_fn(CLIP_NAME), CLIP_NAME, clip_file_size, clip_mtime_ns, CACHE_DIR,
            force_refresh=True,
        )
        tokens = proxy.tokenize(PROMPT, images=[])
        cond = proxy.encode_from_tokens_scheduled(tokens)
        print("B/iter{}: tokenize+encode finished in {:.1f}s -- did_load_real_clip={}".format(
            i, time.time() - t0, proxy.did_load_real_clip))
        assert proxy.did_load_real_clip is True, "force_refresh=True must always be a real MISS"
        del cond, tokens

        comfy.model_management.unload_model_and_clones(proxy.real_clip.patcher)
        rows = _log_step(rows, "B", i, "after-unload-before-del")

        del proxy
        gc.collect()
        torch.cuda.empty_cache()
        rows = _log_step(rows, "B", i, "after-del-and-gc")

    return rows


def main():
    # Watchdog first, before anything else -- must be alive before any model load.
    stop_event = threading.Event()
    watchdog_thread = threading.Thread(target=watchdog_loop, args=(stop_event,), daemon=True)
    watchdog_thread.start()
    print("=== Watchdog started: RAM warn>={:.0f}% hard-stop>={:.0f}%, VRAM hard-stop>={:.0f}% of device total, "
          "polling every {}s ===".format(RAM_WARN_PCT, RAM_HARD_STOP_PCT, VRAM_HARD_STOP_FRACTION * 100.0,
                                          WATCHDOG_INTERVAL_S), flush=True)

    log_memory("before-anything")

    rows = []
    rows = run_variant_a(rows)
    if RUN_VARIANT_B:
        rows = run_variant_b(rows)
    else:
        print()
        print("=== RUN_VARIANT_B is False -- stopping after Variant A as instructed for this retry ===")

    print()
    print("=== Summary table ===")
    header = "{:<28} {:>18} {:>18} {:>14} {:>14} {:>16} {:>12}".format(
        "step", "pytorch_alloc_GiB", "pytorch_reserv_GiB", "nvsmi_used_MiB", "nvsmi_total_MiB",
        "MemAvailable_kB", "own_RSS_kB")
    print(header)
    print("-" * len(header))
    for r in rows:
        label = "{}/iter{}/{}".format(r["variant"], r["iteration"], r["checkpoint"])
        print("{:<28} {:>18.2f} {:>18.2f} {:>14} {:>14} {:>16} {:>12}".format(
            label, r["pytorch_allocated_gib"], r["pytorch_reserved_gib"],
            r["nvidia_smi_used_mib"], r["nvidia_smi_total_mib"],
            r["mem_available_kb"], r["own_rss_kb"]))

    print()
    print("=== RESULT: data collected, see summary table above (no pass/fail -- this is a measurement) ===")

    stop_event.set()


if __name__ == "__main__":
    main()
