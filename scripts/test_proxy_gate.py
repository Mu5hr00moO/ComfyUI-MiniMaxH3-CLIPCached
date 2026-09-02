"""Phase 4-5 go/no-go gate: transparent SpyClipProxy (no cache) in front of the
stock MiniMaxH3ImageToVideo node.

Loads the real MiniMax H3 CLIP (Qwen3-VL encoder) and VAE with the exact same
loader classes ComfyUI's own CLIPLoader/VAELoader nodes use, wraps the clip in
a proxy that logs and forwards every call 1:1, and drives
comfy_extras.nodes_minimax_h3.MiniMaxH3ImageToVideo.execute() directly with
that proxy standing in for clip.

No cache logic here. This only answers: does the stock node work unmodified
when clip is replaced by a transparent pass-through object?

Run with the comfyenv conda environment from anywhere:
    conda run -n comfyenv python scripts/test_proxy_gate.py
"""

import os
import sys
import time

# <ComfyUI>/custom_nodes/<this repo>/scripts/<this file> under a normal
# install, so the ComfyUI root is four directories up from this file.
# Derived rather than hard-coded so the gate also runs from a fork checked
# out elsewhere; COMFYUI_ROOT in the environment overrides it. Mirrors
# tests/conftest.py.
_here = os.path.abspath(__file__)
COMFYUI_ROOT = os.environ.get(
    "COMFYUI_ROOT",
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_here)))),
)
sys.path.insert(0, COMFYUI_ROOT)
os.chdir(COMFYUI_ROOT)  # folder_paths / nodes assume cwd == ComfyUI root

CLIP_NAME = "qwen3vl_32b_minimax_h3_int8_convrot.safetensors"
CLIP_TYPE = "minimax"
VAE_NAME = "minimax_h3_video_vae_int8_convrot.safetensors"  # same int8_convrot family as the encoder

PROMPT = "a test prompt"
WIDTH = 1344
HEIGHT = 768
LENGTH = 124


def log_memory(label):
    import torch

    with open("/proc/meminfo") as f:
        meminfo = dict(
            (line.split(":")[0], line.split(":")[1].strip())
            for line in f if ":" in line
        )
    mem_available = meminfo.get("MemAvailable", "?")
    mem_free = meminfo.get("MemFree", "?")

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        free_cuda, total_cuda = torch.cuda.mem_get_info()
        print(
            "[MEM {}] RAM available={} free={} | "
            "VRAM allocated={:.2f}GiB reserved={:.2f}GiB "
            "device_free={:.2f}GiB device_total={:.2f}GiB".format(
                label, mem_available, mem_free,
                allocated, reserved,
                free_cuda / (1024 ** 3), total_cuda / (1024 ** 3),
            )
        )
    else:
        print("[MEM {}] RAM available={} free={} | CUDA not available".format(
            label, mem_available, mem_free))


class SpyClipProxy:
    """Transparent 1:1 delegate to a real CLIP object. No cache, no logic.

    Records the last tokenize() call (prompt + kwargs) for inspection.
    """

    def __init__(self, real_clip):
        self._real_clip = real_clip
        self.last_call = None

    def tokenize(self, prompt, **kwargs):
        self.last_call = (prompt, kwargs)
        print("[SPY] tokenize called: prompt={!r} kwargs_keys={}".format(
            prompt, list(kwargs.keys())))
        for k, v in kwargs.items():
            if k == "images":
                print("[SPY]   images: {} item(s)".format(len(v)))
                for i, img in enumerate(v):
                    print("[SPY]     image[{}]: type={} shape={}".format(
                        i, type(img).__name__, getattr(img, "shape", None)))
            else:
                print("[SPY]   {}: type={}".format(k, type(v).__name__))
        return self._real_clip.tokenize(prompt, **kwargs)

    def encode_from_tokens_scheduled(self, tokens):
        print("[SPY] encode_from_tokens_scheduled called: tokens type={}".format(
            type(tokens).__name__))
        return self._real_clip.encode_from_tokens_scheduled(tokens)


def main():
    import torch
    import nodes
    import comfy.model_management
    from comfy_extras.nodes_minimax_h3 import MiniMaxH3ImageToVideo

    print("=== Step 1: load real CLIP via stock nodes.CLIPLoader ===")
    log_memory("before-clip-load")
    t0 = time.time()
    clip_loader = nodes.CLIPLoader()
    (clip,) = clip_loader.load_clip(CLIP_NAME, type=CLIP_TYPE)
    print("CLIP loaded in {:.1f}s".format(time.time() - t0))
    log_memory("after-clip-load")

    print("=== Step 2: load real VAE via stock nodes.VAELoader ===")
    t0 = time.time()
    vae_loader = nodes.VAELoader()
    (vae,) = vae_loader.load_vae(VAE_NAME)
    print("VAE loaded in {:.1f}s".format(time.time() - t0))
    log_memory("after-vae-load")

    print("=== Step 3: wrap clip in SpyClipProxy (no cache, pure delegation) ===")
    proxy = SpyClipProxy(clip)

    print("=== Step 4: MiniMaxH3ImageToVideo.execute() with proxy standing in for clip ===")
    t0 = time.time()
    output = MiniMaxH3ImageToVideo.execute(
        clip=proxy,
        vae=vae,
        prompt=PROMPT,
        width=WIDTH,
        height=HEIGHT,
        length=LENGTH,
    )
    print("execute() finished in {:.1f}s".format(time.time() - t0))
    log_memory("after-execute")

    cond, latent = output.args

    print("=== Step 5: inspect results ===")
    print("cond: type={} len={}".format(type(cond).__name__, len(cond)))
    for i, entry in enumerate(cond):
        tensor, extra = entry[0], entry[1]
        print("  cond[{}]: tensor type={} shape={} dtype={} device={}".format(
            i, type(tensor).__name__, tuple(tensor.shape), tensor.dtype, tensor.device))
        print("  cond[{}]: extra keys={}".format(i, list(extra.keys())))

    print("latent: type={} keys={}".format(type(latent).__name__, list(latent.keys())))
    samples = latent["samples"]
    print("latent['samples']: type={} is_nested={}".format(
        type(samples).__name__, getattr(samples, "is_nested", None)))
    if getattr(samples, "is_nested", False):
        for i, t in enumerate(samples.tensors):
            print("  samples.tensors[{}]: shape={} dtype={}".format(i, tuple(t.shape), t.dtype))

    print("=== Step 6: proxy.last_call (exactly what tokenize() received) ===")
    prompt_seen, kwargs_seen = proxy.last_call
    print("prompt: {!r}".format(prompt_seen))
    print("kwargs keys: {}".format(list(kwargs_seen.keys())))
    if "images" in kwargs_seen:
        print("images: {} item(s)".format(len(kwargs_seen["images"])))
        for i, img in enumerate(kwargs_seen["images"]):
            print("  images[{}]: shape={} dtype={}".format(i, tuple(img.shape), img.dtype))

    print("=== Step 7: targeted unload of the real clip via unload_model_and_clones ===")
    log_memory("before-unload")
    comfy.model_management.unload_model_and_clones(clip.patcher)
    log_memory("after-unload")

    print("=== GATE RESULT: PASS (no exception raised) ===")


if __name__ == "__main__":
    main()
