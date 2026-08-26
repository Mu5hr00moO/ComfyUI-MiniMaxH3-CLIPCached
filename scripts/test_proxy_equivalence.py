"""Phase 6 test: stock CLIP vs SpyClipProxy equivalence.

Loads the real MiniMax H3 clip/vae once, then runs
comfy_extras.nodes_minimax_h3.MiniMaxH3ImageToVideo.execute() twice per input
variant - once with the real clip (run_stock) and once with a fresh
SpyClipProxy wrapping that same real clip (run_proxy) - and asserts the
outputs are numerically equivalent.

Two input variants:
  A: no first_frame/last_frame (as in the phase 4-5 gate)
  B: with first_frame, exercising the vae.encode(keyframe) path

No cache logic here. Reuses SpyClipProxy and the loader setup from
scripts/test_proxy_gate.py instead of redefining them.

Run with the comfyenv conda environment from anywhere:
    conda run -n comfyenv python scripts/test_proxy_equivalence.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_proxy_gate import (  # noqa: E402
    CLIP_NAME, CLIP_TYPE, VAE_NAME, PROMPT, WIDTH, HEIGHT, LENGTH,
    SpyClipProxy, log_memory,
)

FAILURES = []


def deep_compare(path, a, b):
    """Recursively compare two result substructures, printing PASS/FAIL per leaf.

    Tensors: torch.allclose(atol=1e-6, rtol=1e-5). Dicts: same key sets, recurse
    per key. Lists/tuples: same length, recurse per index. Everything else: ==.
    """
    import torch

    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        if type(a) is not type(b):
            print("  FAIL {}: type mismatch stock={} proxy={}".format(path, type(a).__name__, type(b).__name__))
            FAILURES.append(path)
            return
        if a.shape != b.shape:
            print("  FAIL {}: shape mismatch stock={} proxy={}".format(path, tuple(a.shape), tuple(b.shape)))
            FAILURES.append(path)
            return
        if a.dtype != b.dtype:
            print("  FAIL {}: dtype mismatch stock={} proxy={}".format(path, a.dtype, b.dtype))
            FAILURES.append(path)
            return
        ok = torch.allclose(a, b, atol=1e-6, rtol=1e-5)
        if ok:
            print("  PASS {}: tensor shape={} dtype={} allclose(atol=1e-6, rtol=1e-5)".format(
                path, tuple(a.shape), a.dtype))
        else:
            diff = (a - b).abs()
            print("  FAIL {}: tensor shape={} dtype={} NOT allclose, max_abs_diff={} mean_abs_diff={}".format(
                path, tuple(a.shape), a.dtype, diff.max().item(), diff.mean().item()))
            FAILURES.append(path)
        return

    if isinstance(a, dict) or isinstance(b, dict):
        if type(a) is not type(b):
            print("  FAIL {}: type mismatch stock={} proxy={}".format(path, type(a).__name__, type(b).__name__))
            FAILURES.append(path)
            return
        keys_a, keys_b = set(a.keys()), set(b.keys())
        if keys_a != keys_b:
            print("  FAIL {}: dict key mismatch stock={} proxy={}".format(path, sorted(keys_a), sorted(keys_b)))
            FAILURES.append(path)
            return
        print("  PASS {}: dict keys match {}".format(path, sorted(keys_a)))
        for k in sorted(keys_a):
            deep_compare("{}[{!r}]".format(path, k), a[k], b[k])
        return

    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        if type(a) is not type(b):
            print("  FAIL {}: type mismatch stock={} proxy={}".format(path, type(a).__name__, type(b).__name__))
            FAILURES.append(path)
            return
        if len(a) != len(b):
            print("  FAIL {}: length mismatch stock={} proxy={}".format(path, len(a), len(b)))
            FAILURES.append(path)
            return
        print("  PASS {}: length match ({})".format(path, len(a)))
        for i, (ea, eb) in enumerate(zip(a, b)):
            deep_compare("{}[{}]".format(path, i), ea, eb)
        return

    if a == b:
        print("  PASS {}: equal ({!r})".format(path, a))
    else:
        print("  FAIL {}: stock={!r} proxy={!r}".format(path, a, b))
        FAILURES.append(path)


def compare_outputs(variant, output_stock, output_proxy):
    print("--- comparing variant {}: cond ---".format(variant))
    cond_stock, latent_stock = output_stock.args
    cond_proxy, latent_proxy = output_proxy.args

    if type(cond_stock) is not type(cond_proxy):
        print("  FAIL cond: type mismatch stock={} proxy={}".format(type(cond_stock).__name__, type(cond_proxy).__name__))
        FAILURES.append("variant {} cond type".format(variant))
    elif len(cond_stock) != len(cond_proxy):
        print("  FAIL cond: length mismatch stock={} proxy={}".format(len(cond_stock), len(cond_proxy)))
        FAILURES.append("variant {} cond length".format(variant))
    else:
        print("  PASS cond: type={} len={}".format(type(cond_stock).__name__, len(cond_stock)))
        for i, (entry_stock, entry_proxy) in enumerate(zip(cond_stock, cond_proxy)):
            tensor_stock, extra_stock = entry_stock[0], entry_stock[1]
            tensor_proxy, extra_proxy = entry_proxy[0], entry_proxy[1]
            deep_compare("cond[{}][0] (main tensor)".format(i), tensor_stock, tensor_proxy)
            deep_compare("cond[{}][1] (extra dict)".format(i), extra_stock, extra_proxy)

    print("--- comparing variant {}: latent['samples'] ---".format(variant))
    samples_stock = latent_stock["samples"]
    samples_proxy = latent_proxy["samples"]
    if samples_stock.is_nested != samples_proxy.is_nested:
        print("  FAIL latent.samples: is_nested mismatch stock={} proxy={}".format(
            samples_stock.is_nested, samples_proxy.is_nested))
        FAILURES.append("variant {} latent.samples.is_nested".format(variant))
        return
    if len(samples_stock.tensors) != len(samples_proxy.tensors):
        print("  FAIL latent.samples: tensor count mismatch stock={} proxy={}".format(
            len(samples_stock.tensors), len(samples_proxy.tensors)))
        FAILURES.append("variant {} latent.samples length".format(variant))
        return
    names = ["video", "audio"]
    for i, (t_stock, t_proxy) in enumerate(zip(samples_stock.tensors, samples_proxy.tensors)):
        name = names[i] if i < len(names) else "tensor[{}]".format(i)
        deep_compare("latent.samples.{}".format(name), t_stock, t_proxy)


def run_variant(label, clip, proxy, vae, extra_kwargs):
    from comfy_extras.nodes_minimax_h3 import MiniMaxH3ImageToVideo

    print("=== Variant {}: run_stock (real clip) ===".format(label))
    t0 = time.time()
    output_stock = MiniMaxH3ImageToVideo.execute(
        clip=clip, vae=vae, prompt=PROMPT, width=WIDTH, height=HEIGHT, length=LENGTH,
        **extra_kwargs,
    )
    print("run_stock finished in {:.1f}s".format(time.time() - t0))

    print("=== Variant {}: run_proxy (SpyClipProxy) ===".format(label))
    t0 = time.time()
    output_proxy = MiniMaxH3ImageToVideo.execute(
        clip=proxy, vae=vae, prompt=PROMPT, width=WIDTH, height=HEIGHT, length=LENGTH,
        **extra_kwargs,
    )
    print("run_proxy finished in {:.1f}s".format(time.time() - t0))
    print("proxy.last_call kwargs keys: {}".format(list(proxy.last_call[1].keys())))

    compare_outputs(label, output_stock, output_proxy)


def main():
    import torch
    import nodes
    import comfy.model_management

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

    # Deterministic test image: IMAGE convention is [B, H, W, C], float32 in [0, 1].
    torch.manual_seed(0)
    test_first_frame = torch.rand(1, 400, 600, 3, dtype=torch.float32)

    variants = [
        ("A_no_keyframes", {}),
        ("B_with_first_frame", {"first_frame": test_first_frame.clone()}),
    ]

    for label, extra_kwargs in variants:
        proxy = SpyClipProxy(clip)
        run_variant(label, clip, proxy, vae, extra_kwargs)

    print("=== Unload real clip (once, after both variants) ===")
    log_memory("before-unload")
    comfy.model_management.unload_model_and_clones(clip.patcher)
    log_memory("after-unload")

    print()
    if FAILURES:
        print("=== EQUIVALENCE RESULT: FAIL ({} mismatching field(s)) ===".format(len(FAILURES)))
        for f in FAILURES:
            print("  - {}".format(f))
        sys.exit(1)
    else:
        print("=== EQUIVALENCE RESULT: PASS (stock == proxy for all variants/fields) ===")


if __name__ == "__main__":
    main()
