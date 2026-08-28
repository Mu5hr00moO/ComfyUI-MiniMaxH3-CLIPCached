"""MiniMaxH3CLIPCachedFL2VA: same public contract as the stock
comfy_extras.nodes_minimax_h3.MiniMaxH3ImageToVideo (t2va / fl2va conditioning
+ AV latent), except the CLIP input is replaced by a clip_name string and a
lazy CachedClipProxy. On a cache HIT the real ~27 GB MiniMax H3 encoder is
never loaded at all; on a MISS it is loaded, used once, and unloaded again
via targeted unload_model_and_clones() before returning (CLAUDE.md phases
17-19). All the actual H3 mechanics (resize, VAE keyframe encode, AV latent,
minimax_keyframes) stay in the stock node -- this file never reimplements
them, it only substitutes what the stock node sees as "clip".
"""

import gc
import os

import nodes
import comfy.model_management
import folder_paths

from minimaxh3_clipcache.loader import build_clip_loader_fn, resolve_clip_stat
from minimaxh3_clipcache.proxy import CachedClipProxy

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(REPO_ROOT, "cache")


class MiniMaxH3CLIPCachedFL2VA:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip_name": (folder_paths.get_filename_list("text_encoders"), {
                    "tooltip": "MiniMax H3 text/vision encoder (Qwen3-VL) checkpoint from "
                               "models/text_encoders. Loaded lazily -- only on a cache miss.",
                }),
                "vae": ("VAE",),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "width": ("INT", {"default": 1344, "min": 32, "max": nodes.MAX_RESOLUTION, "step": 32}),
                "height": ("INT", {"default": 768, "min": 32, "max": nodes.MAX_RESOLUTION, "step": 32}),
                "length": ("INT", {"default": 124, "min": 5, "max": 3600, "step": 17,
                                    "tooltip": "Frame count at 24 fps, snapped up to the model's 17k+5 grid "
                                               "(124 = ~5s; trained range is ~124-362, longer is untested)"}),
            },
            "optional": {
                "first_frame": ("IMAGE",),
                "last_frame": ("IMAGE",),
                "cache_mode": (["auto", "refresh"], {"default": "auto",
                    "tooltip": "auto: reuse the cached encode for an identical prompt+first_frame+"
                               "last_frame+clip_name (checkpoint identity = filename+size+mtime) if "
                               "one exists, otherwise encode and save it. refresh: ignore any cached "
                               "encode, always re-encode and overwrite the cache.",
                }),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "latent")
    FUNCTION = "execute"
    CATEGORY = "model/conditioning/minimax/cached"

    @classmethod
    def IS_CHANGED(cls, cache_mode="auto", **kwargs):
        # ComfyUI calls IS_CHANGED with every graph input as a kwarg, so we
        # only name the one we care about and swallow the rest. Returning a
        # fresh NaN whenever cache_mode == "refresh" makes the executor's
        # signature comparison fail on every Queue (NaN != NaN), so "refresh"
        # forces a real re-execution even when the user clicks Queue again
        # with all inputs unchanged. In "auto" mode we return a stable value
        # so identical graphs still hit ComfyUI's own execution cache.
        if cache_mode == "refresh":
            return float("nan")
        return cache_mode

    def execute(self, clip_name, vae, prompt, width, height, length,
                first_frame=None, last_frame=None, cache_mode="auto"):
        file_size, mtime_ns = resolve_clip_stat(clip_name)
        loader_fn = build_clip_loader_fn(clip_name)
        proxy = CachedClipProxy(
            loader_fn, clip_name, file_size, mtime_ns,
            cache_dir=CACHE_DIR,
            force_refresh=(cache_mode == "refresh"),
        )

        from comfy_extras.nodes_minimax_h3 import MiniMaxH3ImageToVideo
        try:
            cond, latent = MiniMaxH3ImageToVideo.execute(
                clip=proxy, vae=vae, prompt=prompt, width=width, height=height,
                length=length, first_frame=first_frame, last_frame=last_frame,
            )
        finally:
            # Guarantee the ~27 GB encoder is released even if the stock node
            # raises after our proxy already loaded it (a real failure mode
            # seen in phase 23). We do NOT swallow the exception here -- per
            # CLAUDE.md's "no silent fallbacks" rule the error must propagate;
            # we only make sure it doesn't leave the encoder resident as
            # ballast. On an exception cond/latent are never assigned and the
            # function exits by propagating, so there is nothing to return.
            if proxy.did_load_real_clip:
                comfy.model_management.unload_model_and_clones(proxy.real_clip.patcher)
                del proxy
                gc.collect()
                comfy.model_management.soft_empty_cache()

        return (cond, latent)
