"""R3 go/no-go gate: transparent SpyClipProxy (no cache) in front of the stock
MiniMaxH3ReferenceToVideo node (ref2va).

Minimal variant: one reference image, no reference videos, no audio. This only
answers: does the stock ref2va node work unmodified when clip is replaced by a
transparent pass-through object, and what exactly does it hand to tokenize()
(the minimax_ref_items= presentation list)?

Reuses SpyClipProxy / log_memory / loader constants from test_proxy_gate.py so
the proxy is byte-for-byte the same object used in the earlier gates.

Run with the comfyenv conda environment from anywhere:
    conda run -n comfyenv python scripts/test_ref2video_gate.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_proxy_gate import (  # noqa: E402
    CLIP_NAME, CLIP_TYPE, SpyClipProxy, log_memory,
)

VAE_NAME = "minimax_h3_video_vae_int8_convrot.safetensors"
AUDIO_VAE_NAME = "minimax_h3_audio_vae_fp32.safetensors"

PROMPT = "a test prompt with <Picture 1>"
WIDTH = 1344
HEIGHT = 768
LENGTH = 124
REF_IMAGE_SIZE = "match"


def describe(obj, indent="    "):
    """One-line-per-leaf structural dump of a ref_items-style nested structure."""
    import torch

    lines = []

    def walk(path, v):
        if isinstance(v, torch.Tensor):
            lines.append("{}{}: Tensor shape={} dtype={} device={}".format(
                indent, path, tuple(v.shape), v.dtype, v.device))
        elif isinstance(v, dict):
            lines.append("{}{}: dict keys={}".format(indent, path, list(v.keys())))
            for k, sub in v.items():
                walk("{}[{!r}]".format(path, k), sub)
        elif isinstance(v, (list, tuple)):
            lines.append("{}{}: {} len={}".format(indent, path, type(v).__name__, len(v)))
            for i, sub in enumerate(v):
                walk("{}[{}]".format(path, i), sub)
        else:
            lines.append("{}{}: {} = {!r}".format(indent, path, type(v).__name__, v))

    walk("", obj)
    return "\n".join(lines)


def main():
    import torch
    import nodes
    import comfy.model_management
    from comfy_extras.nodes_minimax_h3 import MiniMaxH3ReferenceToVideo

    print("=== Step 1: load real CLIP via stock nodes.CLIPLoader ===")
    log_memory("before-clip-load")
    t0 = time.time()
    clip_loader = nodes.CLIPLoader()
    (clip,) = clip_loader.load_clip(CLIP_NAME, type=CLIP_TYPE)
    print("CLIP loaded in {:.1f}s".format(time.time() - t0))
    log_memory("after-clip-load")

    print("=== Step 2: load real VAE + audio_vae via stock nodes.VAELoader (two separate calls) ===")
    t0 = time.time()
    (vae,) = nodes.VAELoader().load_vae(VAE_NAME)
    print("VAE ({}) loaded in {:.1f}s".format(VAE_NAME, time.time() - t0))
    t0 = time.time()
    (audio_vae,) = nodes.VAELoader().load_vae(AUDIO_VAE_NAME)
    print("audio_vae ({}) loaded in {:.1f}s".format(AUDIO_VAE_NAME, time.time() - t0))
    print("vae is audio_vae? {}".format(vae is audio_vae))
    log_memory("after-vae-load")

    print("=== Step 3: wrap clip in SpyClipProxy (no cache, pure delegation) ===")
    proxy = SpyClipProxy(clip)

    print("=== Step 4: build one reference image (IMAGE convention [B,H,W,C], float32 [0,1]) ===")
    torch.manual_seed(0)
    test_image = torch.rand(1, 512, 512, 3, dtype=torch.float32)
    print("test_image: shape={} dtype={}".format(tuple(test_image.shape), test_image.dtype))

    print("=== Step 5: MiniMaxH3ReferenceToVideo.execute() with proxy standing in for clip ===")
    t0 = time.time()
    output = MiniMaxH3ReferenceToVideo.execute(
        clip=proxy,
        vae=vae,
        audio_vae=audio_vae,
        prompt=PROMPT,
        width=WIDTH,
        height=HEIGHT,
        length=LENGTH,
        ref_image_size=REF_IMAGE_SIZE,
        ref_images={"ref_image_0": test_image},
        ref_videos=None,
        ref_video_audios=None,
        ref_audios=None,
    )
    print("execute() finished in {:.1f}s".format(time.time() - t0))
    log_memory("after-execute")

    cond, latent = output.args

    print("=== Step 6: inspect returned cond / latent ===")
    print("cond: type={} len={}".format(type(cond).__name__, len(cond)))
    for i, entry in enumerate(cond):
        tensor, extra = entry[0], entry[1]
        print("  cond[{}][0]: type={} shape={} dtype={} device={}".format(
            i, type(tensor).__name__, tuple(tensor.shape), tensor.dtype, tensor.device))
        print("  cond[{}][1]: extra keys={}".format(i, list(extra.keys())))
        if "minimax_refs" in extra:
            print("  cond[{}][1]['minimax_refs']:".format(i))
            print(describe(extra["minimax_refs"], indent="      "))

    samples = latent["samples"]
    print("latent: type={} keys={}".format(type(latent).__name__, list(latent.keys())))
    print("latent['samples']: type={} is_nested={}".format(
        type(samples).__name__, getattr(samples, "is_nested", None)))
    if getattr(samples, "is_nested", False):
        names = ["video", "audio"]
        for i, t in enumerate(samples.tensors):
            name = names[i] if i < len(names) else "tensor[{}]".format(i)
            print("  samples.tensors[{}] ({}): shape={} dtype={}".format(
                i, name, tuple(t.shape), t.dtype))

    print("=== Step 7: proxy.last_call (exactly what tokenize() received) ===")
    prompt_seen, kwargs_seen = proxy.last_call
    print("prompt: {!r}".format(prompt_seen))
    print("kwargs keys: {}".format(list(kwargs_seen.keys())))
    ref_items = kwargs_seen.get("minimax_ref_items")
    print("minimax_ref_items: type={} len={}".format(
        type(ref_items).__name__, len(ref_items) if ref_items is not None else None))
    if ref_items:
        print("minimax_ref_items[0]: type={}".format(type(ref_items[0]).__name__))
        print(describe(ref_items, indent="    "))

    print("=== Step 8: targeted unload of the real clip via unload_model_and_clones ===")
    log_memory("before-unload")
    comfy.model_management.unload_model_and_clones(clip.patcher)
    log_memory("after-unload")

    print("=== GATE RESULT: PASS (no exception raised) ===")


if __name__ == "__main__":
    main()
