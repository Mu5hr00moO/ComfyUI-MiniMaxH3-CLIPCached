"""Cached-CLIP siblings of the stock MiniMax H3 conditioning nodes:

- MiniMaxH3CLIPCachedFL2VA  wraps comfy_extras.nodes_minimax_h3.MiniMaxH3ImageToVideo
  (t2va / fl2va: prompt + optional first/last keyframes)
- MiniMaxH3CLIPCachedRef2VA wraps comfy_extras.nodes_minimax_h3.MiniMaxH3ReferenceToVideo
  (ref2va: prompt + reference images / videos / audio)

Both have the same public contract as their stock counterpart, except the
CLIP input is replaced by a clip_name string and a lazy CachedClipProxy. On a
cache HIT the real ~27 GB MiniMax H3 encoder is never loaded at all; on a
MISS it is loaded, used once, and unloaded again via targeted
unload_model_and_clones() before returning (CLAUDE.md phases 17-19). All the
actual H3 mechanics (resize, VAE keyframe / reference encode, AV latent,
minimax_keyframes / minimax_refs) stay in the stock node -- this file never
reimplements them, it only substitutes what the stock node sees as "clip".
"""

import gc
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import nodes
import comfy.model_management
import folder_paths

from minimaxh3_clipcache.encoder_abi import get_encoder_abi_id
from minimaxh3_clipcache.fingerprint import CACHE_SCHEMA_VERSION
from minimaxh3_clipcache.last_used import record_last_used
from minimaxh3_clipcache.loader import build_clip_loader_fn, resolve_clip_stat
from minimaxh3_clipcache.locking import get_lock
from minimaxh3_clipcache.proxy import CachedClipProxy
from minimaxh3_clipcache.thumbnails import save_thumbnail
from minimaxh3_clipcache.verbose_store import load_verbose, save_verbose

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(REPO_ROOT, "cache")

logger = logging.getLogger(__name__)


def _build_references(fingerprint, items, labels=None):
    """Build the positional reference descriptors for the verbose sidecar,
    each with a best-effort JPEG thumbnail.

    items: list[(type: str, tensor_or_None)] in the exact order the stock
    node presents them to the encoder. labels: optional list of the same
    length (only FL2VA -- "first_frame"/"last_frame"); omitted for Ref2VA,
    whose UI derives its numbering positionally from index + type.

    The thumbnail write for one reference must not lose the others or abort
    the verbose write, so the try/except is inside the loop: on failure that
    entry is simply listed without a "thumbnail" key. An audio reference has
    no tensor (None) and is listed without a thumbnail by construction.
    """
    references = []
    for i, (item_type, tensor) in enumerate(items):
        entry = {"index": i, "type": item_type}
        if labels is not None:
            entry["label"] = labels[i]
        if tensor is not None:
            try:
                entry["thumbnail"] = save_thumbnail(tensor, fingerprint, i, CACHE_DIR)
            except Exception as e:
                logger.warning(
                    "[THUMBNAIL WRITE FAILED] %s (%s, index %d): %s - reference "
                    "will be listed without a thumbnail", fingerprint[:12], item_type, i, e,
                )
        references.append(entry)
    return references


def _resolve_created_at(existing_verbose, core_path, is_fresh_miss):
    """Decide this run's <fingerprint>.verbose.json "system.created_at"
    (Cache Manager UI, ISO-8601 UTC, seconds precision).

    - An existing sidecar's valid "system.created_at" always wins, so a HIT
      backfill and a forced refresh of an already-described entry never
      touch it.
    - Otherwise, a genuine fresh MISS (a cache entry that did not exist a
      moment ago) is stamped with the current time.
    - Otherwise (a HIT backfilling a legacy entry with no verbose sidecar at
      all, or one with a sidecar missing this field) the core cache file's
      own mtime is the best available approximation of when the entry was
      actually created.
    """
    existing_system = existing_verbose.get("system") if isinstance(existing_verbose, dict) else None
    existing_created_at = existing_system.get("created_at") if isinstance(existing_system, dict) else None
    if isinstance(existing_created_at, str) and existing_created_at:
        return existing_created_at
    if is_fresh_miss:
        return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    return datetime.fromtimestamp(
        core_path.stat().st_mtime, tz=timezone.utc,
    ).replace(microsecond=0).isoformat()


def _sync_verbose_metadata(proxy, node_variant, prompt, clip_name,
                           clip_file_size, clip_mtime_ns, items, labels=None,
                           clip_ctime_ns=None, width=None, height=None):
    """Write or backfill this run's ``<fingerprint>.verbose.json`` for the
    Cache Manager (plan sections 7 and 8), for either node variant.

    Two cases warrant a write: a fresh MISS whose core cache actually landed
    on disk, and a HIT of an entry that has no verbose sidecar yet (a legacy
    entry -- backfill it from the data this HIT already knows). Everything
    else is a no-op.

    width/height are optional and purely informational
    (system.width/.height/.megapixels) -- diagnostic context for the Cache
    Manager UI, never part of compute_fingerprint() or the HIT/MISS decision.
    Omitted entirely (not None) when not supplied.

    Never raises. The verbose layer is not the source of truth, so a failure
    here must not disturb the already-valid conditioning / core cache result.
    """
    fingerprint = proxy.last_fingerprint
    if fingerprint is None:
        return  # defensive; should not happen after a successful execute()

    try:
        # Hold the per-fingerprint lock across the ENTIRE decision + write,
        # not just the final save_verbose(). The Cache Manager Delete holds
        # this same lock while removing the core <fp>.json/.safetensors,
        # <fp>.verbose.json and thumbnails, and it can win the race between
        # "this run was a fresh MISS" (decided from proxy state) and this
        # function acquiring the lock. Deciding and writing under one lock
        # closes that gap.
        with get_lock(fingerprint):
            # Re-verify under the lock: if Delete already ran, the core
            # <fp>.json is gone and writing a verbose sidecar now would
            # resurrect a phantom entry with no core cache behind it.
            core_path = Path(CACHE_DIR) / "{}.json".format(fingerprint)
            if not core_path.exists():
                return

            existing_verbose = load_verbose(fingerprint, CACHE_DIR)
            existing_system = existing_verbose.get("system") if isinstance(existing_verbose, dict) else None
            existing_created_at = existing_system.get("created_at") if isinstance(existing_system, dict) else None
            has_created_at = isinstance(existing_created_at, str) and bool(existing_created_at)

            fresh_miss_written = proxy.last_hit is False and proxy.last_core_cache_written is True
            hit_needs_backfill = proxy.last_hit is True and (existing_verbose is None or not has_created_at)
            if not (fresh_miss_written or hit_needs_backfill):
                return

            created_at = _resolve_created_at(existing_verbose, core_path, fresh_miss_written)

            references = _build_references(fingerprint, items, labels)
            system = {
                "prompt": prompt,
                "clip_name": clip_name,
                "clip_file_size": clip_file_size,
                "clip_mtime_ns": clip_mtime_ns,
                "cache_schema_version": CACHE_SCHEMA_VERSION,
                "node_variant": node_variant,
                "created_at": created_at,
                "references": references,
            }
            if width is not None and height is not None:
                system["width"] = width
                system["height"] = height
                system["megapixels"] = round(width * height / 1_000_000, 2)
            if clip_ctime_ns is not None:
                system["clip_ctime_ns"] = clip_ctime_ns
            # Informational only: which ComfyUI version produced this cache
            # entry, to help diagnose why an older entry's encode looks
            # different after an upstream update. NOT part of
            # compute_fingerprint() -- it never affects HIT/MISS. Best-effort:
            # a missing/renamed comfyui_version module (or any other failure)
            # must never break the verbose write.
            try:
                from comfyui_version import __version__ as comfyui_version
                system["comfyui_version"] = comfyui_version
            except Exception as e:
                logger.debug("could not record comfyui_version in verbose metadata: %s", e)

            save_verbose(fingerprint, system, CACHE_DIR)
    except Exception as e:
        logger.warning(
            "[VERBOSE WRITE FAILED] %s: could not persist Cache Manager "
            "metadata (%s) - core cache remains valid", fingerprint[:12], e,
        )


def _record_last_used(proxy, node_variant):
    """Record this run's fingerprint as `node_variant`'s currently active
    entry, for the Cache Manager's "highlight the active row" feature
    (minimaxh3_clipcache.last_used). Unlike _sync_verbose_metadata(), this
    runs unconditionally on every successful execute() - HIT, MISS, or
    refresh alike - because "which entry is active right now" doesn't care
    whether this run wrote anything new.
    """
    fingerprint = proxy.last_fingerprint
    if fingerprint is None:
        return  # defensive; should not happen after a successful execute()
    record_last_used(node_variant, fingerprint)


def _is_changed_common(clip_name, cache_mode):
    """Shared body of both nodes' IS_CHANGED classmethod -- the two were
    byte-for-byte identical. Returns the value ComfyUI folds into its own
    execution-cache signature for this node.

    Returning a fresh NaN whenever cache_mode == "refresh" makes the
    executor's signature comparison fail on every Queue (NaN != NaN), so
    "refresh" forces a real re-execution even when the user clicks Queue
    again with all inputs unchanged.

    In "auto" mode we must NOT return a bare constant: ComfyUI's own
    execution cache keys on (literal inputs, IS_CHANGED result), and
    clip_name is just a filename string. If the on-disk checkpoint is
    replaced under the same filename, the literal input is unchanged, so a
    constant IS_CHANGED would let ComfyUI skip re-executing this node
    entirely -- serving a stale CONDITIONING without our own fingerprint
    (which already includes file_size/mtime_ns/ctime_ns) ever being
    computed. Folding the same stat into IS_CHANGED closes that gap: a
    swapped file changes this return value, forcing a real re-execution, at
    which point our own on-disk cache does its normal fingerprint-based
    HIT/MISS. clip_name is only absent when a test calls IS_CHANGED directly
    without it; real graphs always supply it.
    """
    if cache_mode == "refresh":
        return float("nan")
    # If the encoder ABI identity can't be determined (plan audit point 1),
    # disk caching is unsafe this session: return a fresh NaN so ComfyUI
    # always re-executes, and execute() below forces a real encode too.
    abi_id, abi_available = get_encoder_abi_id()
    if not abi_available:
        return float("nan")
    if clip_name is None:
        return cache_mode
    try:
        file_size, mtime_ns, ctime_ns = resolve_clip_stat(clip_name)
    except FileNotFoundError:
        # The checkpoint named in the graph no longer exists on disk. Return
        # NaN (same trick as cache_mode == "refresh" above) to force a real
        # execution rather than let this propagate here, in ComfyUI's own
        # scheduling layer, as a confusing failure before the node even
        # runs -- execute() will raise the same FileNotFoundError, but from
        # inside the node, where ComfyUI reports it as this node's error.
        return float("nan")
    return (cache_mode, clip_name, file_size, mtime_ns, ctime_ns, abi_id)


def _build_cached_proxy(clip_name, cache_mode):
    """Resolve the checkpoint's on-disk identity and build the
    CachedClipProxy that both nodes hand to their stock counterpart in place
    of a real CLIP. Shared verbatim between the two execute() methods.

    Returns (proxy, clip_file_size, clip_mtime_ns, clip_ctime_ns) -- the
    three stat values come back alongside the proxy because execute() also
    needs them for the Cache Manager verbose-metadata write.

    When the encoder ABI identity is unavailable (plan audit point 1), the
    proxy is built with force_refresh=True regardless of cache_mode and
    encoder_abi_id="unavailable": never serve or write a HIT computed under
    an unverified tokenizer implementation. See
    minimaxh3_clipcache.encoder_abi.
    """
    file_size, mtime_ns, ctime_ns = resolve_clip_stat(clip_name)
    loader_fn = build_clip_loader_fn(clip_name)
    abi_id, abi_available = get_encoder_abi_id()
    proxy = CachedClipProxy(
        loader_fn, clip_name, file_size, mtime_ns,
        cache_dir=CACHE_DIR,
        force_refresh=(cache_mode == "refresh") or not abi_available,
        unload_fn=lambda patcher: comfy.model_management.unload_model_and_clones(patcher),
        encoder_abi_id=abi_id if abi_available else "unavailable",
        clip_ctime_ns=ctime_ns,
    )
    return proxy, file_size, mtime_ns, ctime_ns


def _release_real_clip_safety_net(proxy):
    """nodes.py's outer safety net for releasing the real ~27 GB encoder,
    run in both execute() methods' finally. Shared verbatim between them.

    On the normal path the proxy already released the encoder itself right
    after its own real encode (CachedClipProxy.encode_from_tokens_scheduled,
    via the unload_fn passed in by _build_cached_proxy), before returning
    control to the stock node -- so for FL2VA it no longer sits resident
    through the stock node's later keyframe VAE encode, and for Ref2VA
    (whose stock node does its VAE ref-encoding BEFORE the CLIP encode)
    there is no post-encode work left to protect anyway. proxy.real_clip is
    None once the proxy has already released it, so the inner guard avoids a
    redundant double-unload.

    The explicit unload here is only the safety net for a failure INSIDE the
    real encode itself, before the proxy got a chance to unload (a real
    failure mode seen in phase 23). We do NOT swallow the stock node's
    exception -- per CLAUDE.md's "no silent fallbacks" rule the error must
    propagate; we only make sure it doesn't leave the encoder resident as
    ballast. On an exception cond/latent are never assigned and execute()
    exits by propagating, so there is nothing to return.

    Returns True when the real encoder was loaded this run, so the caller
    must then drop its own reference to the proxy and reclaim memory --
    ``del proxy; gc.collect(); comfy.model_management.soft_empty_cache()`` in
    its own finally. Returns False on a cache HIT, where the encoder was
    never loaded and there is nothing to reclaim.

    Those three cleanup statements deliberately stay in each caller's finally
    instead of running here: ``proxy`` is a *parameter* of this function,
    i.e. a second strong reference on top of the one execute() still holds in
    its own frame while it waits (in finally) for this call to return. A
    ``del proxy`` here would only drop this local binding, leaving execute()'s
    reference -- and the real ~27 GB encoder reachable through it -- alive
    across gc.collect() and soft_empty_cache(). Run in execute()'s own frame,
    ``del proxy`` drops the last reference, so the proxy (and the encoder
    under it) is torn down at once, before soft_empty_cache() -- the ordering
    this had before the finally block was extracted into this shared helper.
    """
    if not proxy.did_load_real_clip:
        return False
    if proxy.real_clip is not None:
        try:
            comfy.model_management.unload_model_and_clones(proxy.real_clip.patcher)
        except Exception as e:
            logger.warning(
                "[ENCODER UNLOAD FAILED] could not unload after execute() "
                "(safety-net path): %s", e,
            )
    return True


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
                               "last_frame+clip_name (checkpoint identity = filename+size+mtime+ctime) if "
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
    def IS_CHANGED(cls, clip_name=None, cache_mode="auto", **kwargs):
        # ComfyUI calls IS_CHANGED with every graph input as a kwarg, so we
        # only name the ones we care about and swallow the rest. The body is
        # shared with MiniMaxH3CLIPCachedRef2VA in _is_changed_common().
        return _is_changed_common(clip_name, cache_mode)

    def execute(self, clip_name, vae, prompt, width, height, length,
                first_frame=None, last_frame=None, cache_mode="auto"):
        proxy, file_size, mtime_ns, ctime_ns = _build_cached_proxy(clip_name, cache_mode)

        from comfy_extras.nodes_minimax_h3 import MiniMaxH3ImageToVideo
        try:
            cond, latent = MiniMaxH3ImageToVideo.execute(
                clip=proxy, vae=vae, prompt=prompt, width=width, height=height,
                length=length, first_frame=first_frame, last_frame=last_frame,
            )
            # Inside the try (not the finally): only describe a run that
            # actually produced a conditioning. If the stock execute() raised,
            # there is no successful operation to record.
            items, labels = [], []
            if first_frame is not None:
                items.append(("image", first_frame))
                labels.append("first_frame")
            if last_frame is not None:
                items.append(("image", last_frame))
                labels.append("last_frame")
            _sync_verbose_metadata(
                proxy, "fl2va", prompt, clip_name, file_size, mtime_ns, items,
                labels, clip_ctime_ns=ctime_ns, width=width, height=height,
            )
            _record_last_used(proxy, "fl2va")
        finally:
            # The del/gc/soft_empty_cache stay here, in execute()'s own
            # frame, on purpose -- see _release_real_clip_safety_net's
            # docstring: a `del proxy` inside that helper would not be the
            # last reference (this frame still holds `proxy` until the helper
            # returns), so the encoder would survive the reclaim.
            if _release_real_clip_safety_net(proxy):
                del proxy
                gc.collect()
                comfy.model_management.soft_empty_cache()

        return (cond, latent)


# --- Ref2VA (reference images / videos / audio) -----------------------------

# v1 keeps a fixed number of single reference slots instead of the stock
# node's io.Autogrow, mirroring the stock limits: 9 reference images, 3
# reference videos (each an IMAGE batch of frames, not a VIDEO), 3 matching
# soundtracks, 3 standalone audios.
_REF_IMAGE_COUNT = 9
_REF_VIDEO_COUNT = 3
_REF_AUDIO_COUNT = 3

# Tooltips copied verbatim from the stock MiniMaxH3ReferenceToVideo schema.
_REF_IMAGE_TOOLTIP = "Reference image (downscaled to 2048 short edge if larger, never upscaled)"
_REF_VIDEO_TOOLTIP = "Reference video frames at 24 fps (2-15s)"
_REF_VIDEO_AUDIO_TOOLTIP = "Soundtrack of the same-numbered reference video"
_REF_AUDIO_TOOLTIP = "Standalone reference audio"
_REF_IMAGE_SIZE_TOOLTIP = (
    "Reference image sizing. 'match' scales each ref (down only, keeping aspect) "
    "to the generation's pixel area; 'max' uses the reference pipeline's 2048px "
    "short edge for best identity fidelity. Reference tokens ride through every "
    "sampling step, so 'max' can be several times slower."
)


def _build_ref_slot_dicts(ref_image_slots, ref_video_slots, ref_video_audio_slots, ref_audio_slots):
    """Turn the four flat lists of fixed optional slots into the
    dict-of-named-slots shape the stock MiniMaxH3ReferenceToVideo.execute()
    expects for its ref_images / ref_videos / ref_video_audios / ref_audios
    arguments.

    Each returned dict maps "<prefix><index>" -> value for every slot that is
    actually connected, in ascending slot order; None slots are dropped. A
    group whose slots are all empty is returned as None, not {} -- that is the
    stock execute()'s own default for these arguments. The stock node treats
    None and {} identically (it does `(ref_images or {}).values()` and
    `ref_video_audios = ref_video_audios or {}`), but None matches the stock
    signature exactly and reads unambiguously.

    The "<prefix><index>" key format is load-bearing for videos: the stock
    node pairs a soundtrack to its video with
    `ref_video_audios.get("ref_video_audio_" + name.rsplit("_", 1)[-1])`, so
    ref_video_audio_<i> is only picked up as the soundtrack of ref_video_<i>
    when both keys carry the same trailing index.
    """
    def _group(slots, prefix):
        d = {prefix + str(i): v for i, v in enumerate(slots) if v is not None}
        return d or None

    return (
        _group(ref_image_slots, "ref_image_"),
        _group(ref_video_slots, "ref_video_"),
        _group(ref_video_audio_slots, "ref_video_audio_"),
        _group(ref_audio_slots, "ref_audio_"),
    )


def _build_reference_items(ref_images, ref_videos, ref_video_audios, ref_audios):
    """Reconstruct the flat, ordered reference list the stock
    MiniMaxH3ReferenceToVideo builds for the encoder, as
    list[(type: str, tensor_or_None)] for _build_references().

    Order mirrors the stock node's own assembly (see _build_ref_slot_dicts
    above): all reference images in ascending slot order, then per reference
    video in ascending slot order its matched soundtrack (if connected)
    immediately BEFORE the video itself, then the standalone audios in
    ascending slot order. Audio entries carry no tensor -- the encoder only
    ever sees an "<Audio N>" marker for them, never the waveform.
    """
    def _slot_index(key):
        return int(key.rsplit("_", 1)[-1])

    items = []
    for key in sorted(ref_images or {}, key=_slot_index):
        items.append(("image", ref_images[key]))
    for key in sorted(ref_videos or {}, key=_slot_index):
        audio_key = "ref_video_audio_" + key.rsplit("_", 1)[-1]
        if ref_video_audios and audio_key in ref_video_audios:
            items.append(("audio", None))
        items.append(("video", ref_videos[key]))
    for key in sorted(ref_audios or {}, key=_slot_index):
        items.append(("audio", None))
    return items


class MiniMaxH3CLIPCachedRef2VA:
    @classmethod
    def INPUT_TYPES(cls):
        optional = {}
        for i in range(_REF_IMAGE_COUNT):
            optional["ref_image_" + str(i)] = ("IMAGE", {"tooltip": _REF_IMAGE_TOOLTIP})
        for i in range(_REF_VIDEO_COUNT):
            optional["ref_video_" + str(i)] = ("IMAGE", {"tooltip": _REF_VIDEO_TOOLTIP})
        for i in range(_REF_VIDEO_COUNT):
            optional["ref_video_audio_" + str(i)] = ("AUDIO", {"tooltip": _REF_VIDEO_AUDIO_TOOLTIP})
        for i in range(_REF_AUDIO_COUNT):
            optional["ref_audio_" + str(i)] = ("AUDIO", {"tooltip": _REF_AUDIO_TOOLTIP})
        optional["cache_mode"] = (["auto", "refresh"], {"default": "auto",
            "tooltip": "auto: reuse the cached encode for an identical prompt + reference "
                       "images/videos/audio + clip_name (checkpoint identity = "
                       "filename+size+mtime+ctime) if one exists, otherwise encode and save it. "
                       "refresh: ignore any cached encode, always re-encode and overwrite "
                       "the cache.",
        })
        return {
            "required": {
                "clip_name": (folder_paths.get_filename_list("text_encoders"), {
                    "tooltip": "MiniMax H3 text/vision encoder (Qwen3-VL) checkpoint from "
                               "models/text_encoders. Loaded lazily -- only on a cache miss.",
                }),
                "vae": ("VAE",),
                "audio_vae": ("VAE",),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "width": ("INT", {"default": 1344, "min": 32, "max": nodes.MAX_RESOLUTION, "step": 32}),
                "height": ("INT", {"default": 768, "min": 32, "max": nodes.MAX_RESOLUTION, "step": 32}),
                "length": ("INT", {"default": 124, "min": 5, "max": 3600, "step": 17,
                                    "tooltip": "Frame count at 24 fps, (124 = ~5s, trained range is ~124-362)"}),
                "ref_image_size": (["match", "max"], {"default": "match", "tooltip": _REF_IMAGE_SIZE_TOOLTIP}),
            },
            "optional": optional,
        }

    RETURN_TYPES = ("CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "latent")
    FUNCTION = "execute"
    CATEGORY = "model/conditioning/minimax/cached"

    @classmethod
    def IS_CHANGED(cls, clip_name=None, cache_mode="auto", **kwargs):
        # ComfyUI calls IS_CHANGED with every graph input as a kwarg, so we
        # only name the ones we care about and swallow the rest. The body is
        # shared with MiniMaxH3CLIPCachedFL2VA in _is_changed_common().
        return _is_changed_common(clip_name, cache_mode)

    def execute(self, clip_name, vae, audio_vae, prompt, width, height, length,
                ref_image_size="match",
                ref_image_0=None, ref_image_1=None, ref_image_2=None, ref_image_3=None,
                ref_image_4=None, ref_image_5=None, ref_image_6=None, ref_image_7=None,
                ref_image_8=None,
                ref_video_0=None, ref_video_1=None, ref_video_2=None,
                ref_video_audio_0=None, ref_video_audio_1=None, ref_video_audio_2=None,
                ref_audio_0=None, ref_audio_1=None, ref_audio_2=None,
                cache_mode="auto"):
        proxy, file_size, mtime_ns, ctime_ns = _build_cached_proxy(clip_name, cache_mode)

        ref_images, ref_videos, ref_video_audios, ref_audios = _build_ref_slot_dicts(
            [ref_image_0, ref_image_1, ref_image_2, ref_image_3, ref_image_4,
             ref_image_5, ref_image_6, ref_image_7, ref_image_8],
            [ref_video_0, ref_video_1, ref_video_2],
            [ref_video_audio_0, ref_video_audio_1, ref_video_audio_2],
            [ref_audio_0, ref_audio_1, ref_audio_2],
        )

        from comfy_extras.nodes_minimax_h3 import MiniMaxH3ReferenceToVideo
        try:
            cond, latent = MiniMaxH3ReferenceToVideo.execute(
                clip=proxy, vae=vae, audio_vae=audio_vae, prompt=prompt,
                width=width, height=height, length=length,
                ref_image_size=ref_image_size,
                ref_images=ref_images, ref_videos=ref_videos,
                ref_video_audios=ref_video_audios, ref_audios=ref_audios,
            )
            # Inside the try (not the finally): only describe a run that
            # actually produced a conditioning. If the stock execute() raised,
            # there is no successful operation to record.
            items = _build_reference_items(ref_images, ref_videos, ref_video_audios, ref_audios)
            _sync_verbose_metadata(
                proxy, "ref2va", prompt, clip_name, file_size, mtime_ns, items,
                clip_ctime_ns=ctime_ns, width=width, height=height,
            )
            _record_last_used(proxy, "ref2va")
        finally:
            # The del/gc/soft_empty_cache stay here, in execute()'s own
            # frame, on purpose -- see _release_real_clip_safety_net's
            # docstring: a `del proxy` inside that helper would not be the
            # last reference (this frame still holds `proxy` until the helper
            # returns), so the encoder would survive the reclaim.
            if _release_real_clip_safety_net(proxy):
                del proxy
                gc.collect()
                comfy.model_management.soft_empty_cache()

        return (cond, latent)
