"""Phase 24 step 1: isolate whether repeated real vae.encode() calls alone
(no CLIP loaded at all, ever) accumulate VRAM across calls in this
environment -- to confirm or rule out the "Otwarte pytania - faza 24"
hypothesis in CLAUDE.md that the growth seen in scripts/test_stock_vs_cache.py
comes from vae.encode(), not from CachedClipProxy or the CLIP unload path.

Loads only the real MiniMax H3 VAE (minimax_h3_video_vae_int8_convrot) via
the stock nodes.VAELoader -- no nodes.CLIPLoader() call anywhere in this
script, no CachedClipProxy, no minimaxh3_clipcache/ package involved at all. Runs
vae.encode() four times in a row on a freshly resized test image each time,
exactly the same call shape as
`kf["latent"] = vae.encode(kf.pop("image"))` in
comfy_extras.nodes_minimax_h3.MiniMaxH3ImageToVideo.execute() (image resized
to 1344x768 via the same _resize() helper, crop="disabled", matching how
first_frame is resized there) -- logging torch.cuda.memory_allocated(),
torch.cuda.memory_reserved(), and real nvidia-smi used/total after every
call, with no unload anywhere inside the loop.

Then runs three cleanup steps one at a time, logging after each, to see
which one (if any) actually releases VRAM:
  1. comfy.model_management.unload_model_and_clones(vae.patcher) -- VAE has
     a .patcher attribute assigned in comfy/sd.py's VAE.__init__, the same
     ModelPatcher-based mechanism CLIP uses, confirmed by reading comfy/sd.py
     directly rather than assumed.
  2. torch.cuda.empty_cache()
  3. gc.collect()

Run with the comfyenv conda environment from anywhere, under a hard timeout:
    timeout 600 conda run -n comfyenv --no-capture-output python scripts/test_vae_memory_isolation.py
"""

import gc
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_proxy_gate import VAE_NAME, WIDTH, HEIGHT, log_memory  # noqa: E402

ITERATIONS = 4


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


def _log_step(rows, label):
    import torch

    allocated = torch.cuda.memory_allocated() / (1024 ** 3)
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)
    used_mib, total_mib = _nvidia_smi_used_total()
    print("[{}] pytorch allocated={:.2f}GiB reserved={:.2f}GiB | nvidia-smi used={}MiB total={}MiB".format(
        label, allocated, reserved, used_mib, total_mib))
    rows.append({
        "label": label,
        "pytorch_allocated_gib": allocated,
        "pytorch_reserved_gib": reserved,
        "nvidia_smi_used_mib": used_mib,
        "nvidia_smi_total_mib": total_mib,
    })
    return rows


def main():
    import torch
    import nodes
    import comfy.model_management
    from comfy_extras.nodes_minimax_h3 import _resize

    rows = []

    print("=== Load real VAE only (no CLIP anywhere in this script) ===")
    log_memory("before-vae-load")
    t0 = time.time()
    vae_loader = nodes.VAELoader()
    (vae,) = vae_loader.load_vae(VAE_NAME)
    print("VAE loaded in {:.1f}s".format(time.time() - t0))
    log_memory("after-vae-load")
    rows = _log_step(rows, "after-vae-load")

    assert hasattr(vae, "patcher"), \
        "VAE object has no .patcher attribute -- confirmed present in comfy/sd.py VAE.__init__ for this " \
        "ComfyUI version, so this would mean a version mismatch; do not guess an alternate unload path."

    print()
    print("=== {} repeated vae.encode() calls, no unload in between ===".format(ITERATIONS))
    for i in range(1, ITERATIONS + 1):
        torch.manual_seed(i)
        test_image = torch.rand(1, 400, 600, 3, dtype=torch.float32)
        resized = _resize(test_image[:1], WIDTH, HEIGHT, "disabled")

        t0 = time.time()
        latent = vae.encode(resized)
        dt = time.time() - t0
        print("iteration {}: vae.encode() finished in {:.1f}s, latent shape={}".format(
            i, dt, tuple(latent.shape)))
        rows = _log_step(rows, "after-iteration-{}".format(i))

        del latent, resized, test_image

    print()
    print("=== Cleanup, one step at a time ===")
    comfy.model_management.unload_model_and_clones(vae.patcher)
    rows = _log_step(rows, "after-unload_model_and_clones")

    torch.cuda.empty_cache()
    rows = _log_step(rows, "after-empty_cache")

    gc.collect()
    rows = _log_step(rows, "after-gc_collect")

    print()
    print("=== Summary table ===")
    header = "{:<28} {:>18} {:>18} {:>14} {:>14}".format(
        "step", "pytorch_alloc_GiB", "pytorch_reserv_GiB", "nvsmi_used_MiB", "nvsmi_total_MiB")
    print(header)
    print("-" * len(header))
    for r in rows:
        print("{:<28} {:>18.2f} {:>18.2f} {:>14} {:>14}".format(
            r["label"], r["pytorch_allocated_gib"], r["pytorch_reserved_gib"],
            str(r["nvidia_smi_used_mib"]), str(r["nvidia_smi_total_mib"])))

    print()
    print("=== RESULT: data collected, see summary table above (no pass/fail -- this is a measurement) ===")


if __name__ == "__main__":
    main()
