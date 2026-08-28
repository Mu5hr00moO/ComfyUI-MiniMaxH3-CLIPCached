"""MiniMaxH3CLIPCachedImageToVideo: same public contract as the stock
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
import logging
import os

import nodes
import comfy.model_management
import folder_paths

from minimaxh3_clipcache.fingerprint import CACHE_SCHEMA_VERSION
from minimaxh3_clipcache.loader import build_clip_loader_fn, resolve_clip_stat
from minimaxh3_clipcache.proxy import CachedClipProxy
from minimaxh3_clipcache.thumbnails import save_thumbnail
from minimaxh3_clipcache.verbose_store import load_verbose, save_verbose

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(REPO_ROOT, "cache")

logger = logging.getLogger(__name__)


class MiniMaxH3CLIPCachedImageToVideo:
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

    def _build_references(self, fingerprint, first_frame, last_frame):
        """Build the positional reference descriptors for the verbose sidecar,
        each with a best-effort JPEG thumbnail.

        Indices are positional over what was actually supplied, not a fixed
        first_frame=0 / last_frame=1: if only last_frame is given it is
        index 0. Phase 3's thumbnail filenames use the same index, so this
        has to stay consistent.

        The thumbnail write for one reference must not lose the other
        reference or abort the verbose write, so the try/except is inside the
        loop: on failure that entry is simply listed without a "thumbnail"
        key.
        """
        references = []
        for label, image in (("first_frame", first_frame), ("last_frame", last_frame)):
            if image is None:
                continue
            index = len(references)
            entry = {"index": index, "label": label}
            try:
                entry["thumbnail"] = save_thumbnail(image, fingerprint, index, CACHE_DIR)
            except Exception as e:
                logger.warning(
                    "[THUMBNAIL WRITE FAILED] %s (%s, index %d): %s - reference "
                    "will be listed without a thumbnail", fingerprint[:12], label, index, e,
                )
            references.append(entry)
        return references

    def _sync_verbose_metadata(self, proxy, prompt, clip_name, clip_file_size,
                               clip_mtime_ns, first_frame, last_frame):
        """Write or backfill this run's ``<fingerprint>.verbose.json`` for the
        Cache Manager (plan sections 7 and 8).

        Two cases warrant a write: a fresh MISS whose core cache actually
        landed on disk, and a HIT of an entry that has no verbose sidecar yet
        (a legacy entry -- backfill it from the data this HIT already knows).
        Everything else is a no-op.

        Never raises. The verbose layer is not the source of truth, so a
        failure here must not disturb the already-valid conditioning / core
        cache result.
        """
        fingerprint = proxy.last_fingerprint
        if fingerprint is None:
            return  # defensive; should not happen after a successful execute()

        fresh_miss_written = proxy.last_hit is False and proxy.last_core_cache_written is True
        hit_needs_backfill = proxy.last_hit is True and load_verbose(fingerprint, CACHE_DIR) is None
        if not (fresh_miss_written or hit_needs_backfill):
            return

        references = self._build_references(fingerprint, first_frame, last_frame)

        system = {
            "prompt": prompt,
            "clip_name": clip_name,
            "clip_file_size": clip_file_size,
            "clip_mtime_ns": clip_mtime_ns,
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "references": references,
        }
        try:
            save_verbose(fingerprint, system, CACHE_DIR)
        except Exception as e:
            logger.warning(
                "[VERBOSE WRITE FAILED] %s: could not persist Cache Manager "
                "metadata (%s) - core cache remains valid", fingerprint[:12], e,
            )

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
            # Inside the try (not the finally): only describe a run that
            # actually produced a conditioning. If the stock execute() raised,
            # there is no successful operation to record.
            self._sync_verbose_metadata(proxy, prompt, clip_name, file_size,
                                        mtime_ns, first_frame, last_frame)
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
