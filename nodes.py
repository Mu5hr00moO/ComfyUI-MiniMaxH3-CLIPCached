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
from minimaxh3_clipcache.verbose_store import add_pairing, load_verbose, save_verbose

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


def _pair_verbose_entries(fp_a, width_a, height_a, fp_b, width_b, height_b,
                          b_is_upscale_target=True):
    """Cross-link the two verbose sidecars produced by one dual-resolution
    run so the Cache Manager UI (a separate, later phase) can show just the
    primary entry with a "+ rescaled to WxH" badge instead of listing the
    same prompt twice.

    A dual-resolution node runs the full cached encode path once per target
    resolution. When the encoder input differs by resolution the two runs
    land on two distinct fingerprints -- two separate cache entries with an
    identical prompt. This records, in each entry's ``system`` block, the
    other entry's fingerprint and pixel size (``paired_fingerprint`` /
    ``paired_width`` / ``paired_height``) plus an explicit
    ``is_upscale_target`` role flag.

    The a/b split is pure filesystem symmetry -- both directions are always
    written. Which side is the upscale target is the caller's
    responsibility: ``b_is_upscale_target`` (default True) marks side b as
    the upscale-resolution entry and side a as the base-resolution entry,
    so both nodes call this with fp_a / width_a / height_a as the
    width / height side and fp_b / width_b / height_b as the
    width_upscale / height_upscale side.

    fp_a == fp_b is an immediate no-op: the two resolutions collapsed onto
    one shared cache entry, so there is nothing to pair.

    Otherwise the two directions are written under ``get_lock(fp_a)`` and
    ``get_lock(fp_b)`` taken SEPARATELY -- one acquired and released before
    the other -- never nested, so two dual-resolution runs pairing an
    overlapping set of fingerprints cannot deadlock on lock ordering. Each
    direction re-checks that its own core ``<fp>.json`` still exists under
    the lock, the same guard _sync_verbose_metadata() uses, so a Cache
    Manager Delete that removed one entry mid-run does not get a pairing
    pointer written back into a sidecar with no core cache behind it.

    Never raises. Like _sync_verbose_metadata(), the verbose layer is not the
    source of truth: a failure here must not disturb the already-valid
    conditioning / latent the dual node is about to return.
    """
    if fp_a == fp_b:
        return

    try:
        cache_dir = Path(CACHE_DIR)
        # a -> b
        with get_lock(fp_a):
            if (cache_dir / "{}.json".format(fp_a)).exists():
                add_pairing(fp_a, CACHE_DIR, fp_b, width_b, height_b,
                            is_upscale_target=not b_is_upscale_target)
        # b -> a (separate lock acquisition, not nested inside the above)
        with get_lock(fp_b):
            if (cache_dir / "{}.json".format(fp_b)).exists():
                add_pairing(fp_b, CACHE_DIR, fp_a, width_a, height_a,
                            is_upscale_target=b_is_upscale_target)
    except Exception as e:
        logger.warning(
            "[VERBOSE PAIRING FAILED] %s <-> %s: could not cross-link the "
            "dual-resolution Cache Manager entries (%s) - both core caches "
            "remain valid", fp_a[:12] if fp_a else fp_a, fp_b[:12] if fp_b else fp_b, e,
        )


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
    # a cache HIT/reuse is unsafe this session: return a fresh NaN so ComfyUI
    # always re-executes, and execute() below forces a real encode too (a
    # successful encode may still write cache files under a sentinel
    # fingerprint -- only reuse is suppressed).
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


class _ComboType(str):
    """A ``str`` that additionally compares equal to *any* ``list``.

    Used as ``MiniMaxH3CLIPName``'s single RETURN_TYPE so its output can be
    wired into the ``clip_name`` widget-turned-input of any number of
    MiniMaxH3CLIPCachedFL2VA / MiniMaxH3CLIPCachedRef2VA nodes.

    Why this is needed on ComfyUI 0.34.2: an old-style COMBO input
    (``(options_list, {...})``) is seen by the executor as the raw list of
    option strings, not the string ``"COMBO"`` -- the string conversion is
    commented out in ``comfy_execution/graph.py``. When Queue validates a
    link, ``comfy_execution/validation.py`` runs
    ``if not received_type != input_type`` first, which honours a ``__ne__``
    override on the received type. So:

    - ``RETURN_TYPES = ("COMBO",)`` (a plain str) fails that check against a
      list and Queue raises a type mismatch, even though the link draws in
      the editor.
    - ``RETURN_TYPES = (folder_paths.get_filename_list(...),)`` (a plain
      list frozen at import time) works, but drifts out of sync if
      ``models/text_encoders`` gains or loses a file mid-session.

    Overriding ``__ne__``/``__eq__`` to match any list keeps the link valid
    regardless of the live encoder list, without a restart. This rides on
    undocumented-but-stable behaviour: ``validation.py`` carries the note
    "if we ever want to break them on purpose, this can be removed" next to
    its list/COMBO special-case, so this may need revisiting on a future
    ComfyUI upgrade. The checks stay narrow -- a non-list ``other`` still
    falls through to normal string comparison, so this does not accidentally
    match unrelated string types such as ``"STRING"``.
    """

    def __ne__(self, other):
        return False if isinstance(other, list) else str.__ne__(self, other)

    def __eq__(self, other):
        return True if isinstance(other, list) else str.__eq__(self, other)

    __hash__ = str.__hash__


def _clip_name_input_spec(tooltip=None):
    """The shared ``clip_name`` INPUT_TYPES entry -- the encoder-checkpoint
    dropdown -- used by all three nodes in this package so the widget stays
    identical across them.

    Returns the ``(options_list, options_dict)`` pair, with the option list
    coming from ``folder_paths.get_filename_list("text_encoders")``. Pass
    ``tooltip`` to override the default help text for a node whose role
    differs (MiniMaxH3CLIPName); FL2VA and Ref2VA use the default, which is
    byte-for-byte the tooltip they carried inline before.
    """
    if tooltip is None:
        tooltip = ("MiniMax H3 text/vision encoder (Qwen3-VL) checkpoint from "
                   "models/text_encoders. Loaded lazily -- only on a cache miss.")
    return (folder_paths.get_filename_list("text_encoders"), {"tooltip": tooltip})


def _execute_fl2va_once(clip_name, vae, prompt, width, height, length,
                        first_frame, last_frame, cache_mode):
    """One full cached FL2VA encode at a single resolution.

    This is the entire body of MiniMaxH3CLIPCachedFL2VA.execute() from the
    proxy build onward, lifted into a module function so a second node
    (MiniMaxH3CLIPCachedFL2VADualRes) can run it twice -- once per target
    resolution -- from a single shared set of inputs. Behaviour is identical
    to handing the stock MiniMaxH3ImageToVideo.execute() a CachedClipProxy in
    place of clip: a cache HIT never loads the real encoder, a MISS loads it
    once, uses it, and releases it before returning.

    Nothing here branches on width/height. The existing fingerprint/proxy
    alone decides HIT vs MISS, so two resolutions whose encoder input happens
    to be identical (no keyframes, or keyframes that resize the same) share
    one cache entry transparently, while two that differ each encode for
    real -- exactly as two separate nodes would.
    """
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
        # Read the fingerprint out while the proxy is still alive: the finally
        # below may `del proxy` as part of reclaiming the real encoder, and
        # MiniMaxH3CLIPCachedFL2VADualRes needs it to pair the two entries.
        fingerprint = proxy.last_fingerprint
    finally:
        # The del/gc/soft_empty_cache stay here, in _execute_fl2va_once()'s
        # own frame, on purpose -- see _release_real_clip_safety_net's
        # docstring: a `del proxy` inside that helper would not be the last
        # reference (this frame still holds `proxy` until the helper returns),
        # so the encoder would survive the reclaim.
        if _release_real_clip_safety_net(proxy):
            del proxy
            gc.collect()
            comfy.model_management.soft_empty_cache()

    return (cond, latent, fingerprint)


class MiniMaxH3CLIPCachedFL2VA:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip_name": _clip_name_input_spec(),
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
        # Thin wrapper: the whole body now lives in _execute_fl2va_once() so
        # MiniMaxH3CLIPCachedFL2VADualRes can reuse it verbatim for a second
        # resolution. Single-resolution behaviour is unchanged -- the third
        # tuple element (the fingerprint) is only of use to the dual node.
        cond, latent, _fingerprint = _execute_fl2va_once(
            clip_name, vae, prompt, width, height, length,
            first_frame, last_frame, cache_mode,
        )
        return (cond, latent)


class MiniMaxH3CLIPCachedFL2VADualRes:
    """Two-resolution sibling of MiniMaxH3CLIPCachedFL2VA.

    Produces CONDITIONING + AV LATENT for two resolutions -- a base one
    (width/height) and an upscale target (width_upscale/height_upscale) --
    from a single shared set of inputs (clip_name, prompt, vae,
    first_frame, last_frame, length, cache_mode). Driving both resolutions
    off one node removes the risk of those shared values silently drifting
    apart between two separate MiniMaxH3CLIPCachedFL2VA instances in the same
    graph.

    It runs the full, unmodified cached encode path (_execute_fl2va_once)
    once per resolution and lets the existing fingerprint/proxy decide HIT vs
    MISS each time -- there is no width/height-conditional logic here. Under
    ``cache_mode="auto"``, when the encoder input is resolution-independent
    (no keyframes, or keyframes that resize identically) the second call is a
    natural cache HIT and the real encoder loads at most once; when keyframes
    make the pixels differ, both resolutions encode for real, exactly as two
    separate nodes would. ``cache_mode="refresh"`` skips the HIT path on both
    passes: each resolution re-encodes regardless of whether the fingerprints
    match, so the encoder always loads twice.

    The optional ``generate_upscale_cond`` bool (default True) gates the
    upscale-resolution encode. When it is False the second
    _execute_fl2va_once() call does not run at all -- ``positive_upscale`` /
    ``latent_upscale`` come back as ``None`` and _pair_verbose_entries() is
    skipped (there is no second fingerprint to pair). This switch is the
    ONLY way to avoid paying the upscale encode / VRAM cost, because the
    node is a single atomic Python call that returns all four outputs at
    once: ComfyUI cannot partially execute it, so bypassing the downstream
    consumer of ``positive_upscale`` / ``latent_upscale`` (an entire
    upscaler chain set to bypass, say) still forces this node to run in
    full for the base-resolution outputs, and the upscale encode happens
    regardless. Do not "fix" this by making the second encode conditional
    on something else -- the node has no visibility into what downstream
    consumes its outputs.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip_name": _clip_name_input_spec(),
                "vae": ("VAE",),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "width": ("INT", {"default": 1344, "min": 32, "max": nodes.MAX_RESOLUTION, "step": 32}),
                "height": ("INT", {"default": 768, "min": 32, "max": nodes.MAX_RESOLUTION, "step": 32}),
                "width_upscale": ("INT", {"default": 1344, "min": 32, "max": nodes.MAX_RESOLUTION, "step": 32,
                                    "tooltip": "Encoded through the same fully independent cached path as "
                                               "width -- with cache_mode auto a cache HIT when the encoder input ends up"
                                               " identical, otherwise a real encode; "
                                               "cache_mode refresh always re-encodes."}),
                "height_upscale": ("INT", {"default": 768, "min": 32, "max": nodes.MAX_RESOLUTION, "step": 32,
                                     "tooltip": "Encoded through the same fully independent cached path as "
                                                "height -- with cache_mode auto a cache HIT when the encoder input ends up"
                                                " identical, otherwise a real encode; "
                                                "cache_mode refresh always re-encodes."}),
                "length": ("INT", {"default": 124, "min": 5, "max": 3600, "step": 17,
                                    "tooltip": "Frame count at 24 fps, snapped up to the model's 17k+5 grid "
                                               "(124 = ~5s; trained range is ~124-362, longer is untested)"}),
            },
            "optional": {
                "first_frame": ("IMAGE",),
                "last_frame": ("IMAGE",),
                "generate_upscale_cond": ("BOOLEAN", {"default": True, "tooltip":
                    "When off, the second (upscale-resolution) encode is skipped entirely - "
                    "positive_upscale/latent_upscale come back as None. Turn off for a plain "
                    "generation where nothing downstream uses the upscale outputs; turn on "
                    "when you actually need them. Bypassing the upscale consumer downstream "
                    "does NOT skip this encode by itself - this is the only thing that does, "
                    "because the node runs as one atomic call."}),
                "cache_mode": (["auto", "refresh"], {"default": "auto",
                    "tooltip": "auto: reuse the cached encode for an identical prompt+first_frame+"
                               "last_frame+clip_name (checkpoint identity = filename+size+mtime+ctime) if "
                               "one exists, otherwise encode and save it. refresh: ignore any cached "
                               "encode, always re-encode and overwrite the cache. Applies to both "
                               "resolutions.",
                }),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "LATENT", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "latent", "positive_upscale", "latent_upscale")
    FUNCTION = "execute"
    CATEGORY = "model/conditioning/minimax/cached"

    @classmethod
    def IS_CHANGED(cls, clip_name=None, cache_mode="auto", **kwargs):
        # Same contract as the single-resolution node: IS_CHANGED tracks the
        # CLIP checkpoint's identity and the refresh flag, neither of which
        # depends on how many resolutions this run computes.
        return _is_changed_common(clip_name, cache_mode)

    def execute(self, clip_name, vae, prompt, width, height, width_upscale, height_upscale,
                length, first_frame=None, last_frame=None, cache_mode="auto",
                generate_upscale_cond=True):
        cond, latent, fp1 = _execute_fl2va_once(
            clip_name, vae, prompt, width, height, length,
            first_frame, last_frame, cache_mode,
        )
        if not generate_upscale_cond:
            # Upscale pass switched off: the second _execute_fl2va_once() does
            # not run at all (zero encode / VRAM cost) and there is no second
            # fingerprint, so _pair_verbose_entries() is skipped too. See the
            # class docstring for why bypassing the downstream consumer cannot
            # achieve this on its own.
            logger.info(
                "[UPSCALE COND SKIPPED] %s: generate_upscale_cond=False - "
                "positive_upscale/latent_upscale not computed", fp1[:12],
            )
            return (cond, latent, None, None)
        cond_upscale, latent_upscale, fp2 = _execute_fl2va_once(
            clip_name, vae, prompt, width_upscale, height_upscale, length,
            first_frame, last_frame, cache_mode,
        )
        # Both encodes succeeded (either call raising propagates before here),
        # so it is safe to cross-link the two Cache Manager entries now.
        # fp1 is the base-resolution side, fp2 the upscale-resolution side.
        _pair_verbose_entries(fp1, width, height, fp2, width_upscale, height_upscale,
                              b_is_upscale_target=True)
        return (cond, latent, cond_upscale, latent_upscale)


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


def _ref_slots_input_spec():
    """The fixed optional reference-slot block shared verbatim between
    MiniMaxH3CLIPCachedRef2VA.INPUT_TYPES() and
    MiniMaxH3CLIPCachedRef2VADualRes.INPUT_TYPES(): _REF_IMAGE_COUNT image
    slots, _REF_VIDEO_COUNT video slots, the same number of matching
    soundtrack slots, and _REF_AUDIO_COUNT standalone audio slots, each
    carrying the stock tooltip for its kind.

    Returned as a fresh dict on every call so each caller can add its own
    "cache_mode" entry afterward without mutating shared state. "cache_mode"
    is deliberately NOT part of this block: its tooltip differs between the
    two nodes (the dual-resolution one appends "Applies to both
    resolutions."), so it stays defined separately in each INPUT_TYPES().
    """
    optional = {}
    for i in range(_REF_IMAGE_COUNT):
        optional["ref_image_" + str(i)] = ("IMAGE", {"tooltip": _REF_IMAGE_TOOLTIP})
    for i in range(_REF_VIDEO_COUNT):
        optional["ref_video_" + str(i)] = ("IMAGE", {"tooltip": _REF_VIDEO_TOOLTIP})
    for i in range(_REF_VIDEO_COUNT):
        optional["ref_video_audio_" + str(i)] = ("AUDIO", {"tooltip": _REF_VIDEO_AUDIO_TOOLTIP})
    for i in range(_REF_AUDIO_COUNT):
        optional["ref_audio_" + str(i)] = ("AUDIO", {"tooltip": _REF_AUDIO_TOOLTIP})
    return optional


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


def _execute_ref2va_once(clip_name, vae, audio_vae, prompt, width, height, length,
                         ref_image_size, ref_images, ref_videos, ref_video_audios,
                         ref_audios, cache_mode):
    """One full cached Ref2VA encode at a single resolution.

    This is the body of MiniMaxH3CLIPCachedRef2VA.execute() from the proxy
    build onward, lifted into a module function so
    MiniMaxH3CLIPCachedRef2VADualRes can run it twice -- once per target
    resolution -- from the same references. The four ref_* arguments are the
    already-assembled {name: value} dicts (the caller runs
    _build_ref_slot_dicts on its flat optional slots); everything else
    matches the stock MiniMaxH3ReferenceToVideo.execute() contract.

    As with _execute_fl2va_once nothing here branches on width/height -- the
    existing fingerprint/proxy alone decides HIT vs MISS.
    """
    proxy, file_size, mtime_ns, ctime_ns = _build_cached_proxy(clip_name, cache_mode)

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
        # Read the fingerprint out while the proxy is still alive: the finally
        # below may `del proxy` as part of reclaiming the real encoder, and
        # MiniMaxH3CLIPCachedRef2VADualRes needs it to pair the two entries.
        fingerprint = proxy.last_fingerprint
    finally:
        # The del/gc/soft_empty_cache stay here, in _execute_ref2va_once()'s
        # own frame, on purpose -- see _release_real_clip_safety_net's
        # docstring: a `del proxy` inside that helper would not be the last
        # reference (this frame still holds `proxy` until the helper returns),
        # so the encoder would survive the reclaim.
        if _release_real_clip_safety_net(proxy):
            del proxy
            gc.collect()
            comfy.model_management.soft_empty_cache()

    return (cond, latent, fingerprint)


class MiniMaxH3CLIPCachedRef2VA:
    @classmethod
    def INPUT_TYPES(cls):
        optional = _ref_slots_input_spec()
        optional["cache_mode"] = (["auto", "refresh"], {"default": "auto",
            "tooltip": "auto: reuse the cached encode for an identical prompt + reference "
                       "images/videos/audio + clip_name (checkpoint identity = "
                       "filename+size+mtime+ctime) if one exists, otherwise encode and save it. "
                       "refresh: ignore any cached encode, always re-encode and overwrite "
                       "the cache.",
        })
        return {
            "required": {
                "clip_name": _clip_name_input_spec(),
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
        # Thin wrapper: the flat optional slots are folded into the stock
        # {name: value} dicts here, then the whole encode body runs in
        # _execute_ref2va_once() -- shared verbatim with
        # MiniMaxH3CLIPCachedRef2VADualRes. Single-resolution behaviour is
        # unchanged.
        ref_images, ref_videos, ref_video_audios, ref_audios = _build_ref_slot_dicts(
            [ref_image_0, ref_image_1, ref_image_2, ref_image_3, ref_image_4,
             ref_image_5, ref_image_6, ref_image_7, ref_image_8],
            [ref_video_0, ref_video_1, ref_video_2],
            [ref_video_audio_0, ref_video_audio_1, ref_video_audio_2],
            [ref_audio_0, ref_audio_1, ref_audio_2],
        )
        cond, latent, _fingerprint = _execute_ref2va_once(
            clip_name, vae, audio_vae, prompt, width, height, length,
            ref_image_size, ref_images, ref_videos, ref_video_audios, ref_audios,
            cache_mode,
        )
        return (cond, latent)


class MiniMaxH3CLIPCachedRef2VADualRes:
    """Two-resolution sibling of MiniMaxH3CLIPCachedRef2VA.

    Produces CONDITIONING + AV LATENT for two resolutions -- a base one
    (width/height) and an upscale target (width_upscale/height_upscale) --
    from a single shared set of inputs (clip_name, prompt, vae,
    audio_vae, ref_image_size, every ref_* slot, length, cache_mode).
    Driving both resolutions off one node removes the risk of those shared
    values silently drifting apart between two separate
    MiniMaxH3CLIPCachedRef2VA instances in the same graph.

    It runs the full, unmodified cached encode path (_execute_ref2va_once)
    once per resolution and lets the existing fingerprint/proxy decide HIT vs
    MISS each time -- there is no width/height-conditional logic here. Under
    ``cache_mode="auto"``, with no references, small references, or
    ref_image_size="max" the encoder input is resolution-independent and the
    second call is a natural cache HIT (the real encoder loads at most once);
    with large references under ref_image_size="match" the pixels handed to
    the encoder differ by resolution and both encode for real, exactly as two
    separate nodes would. ``cache_mode="refresh"`` skips the HIT path on both
    passes: each resolution re-encodes regardless of whether the fingerprints
    match, so the encoder always loads twice.

    The optional ``generate_upscale_cond`` bool (default True) gates the
    upscale-resolution encode. When it is False the second
    _execute_ref2va_once() call does not run at all -- ``positive_upscale`` /
    ``latent_upscale`` come back as ``None`` and _pair_verbose_entries() is
    skipped (there is no second fingerprint to pair). This switch is the
    ONLY way to avoid paying the upscale encode / VRAM cost, because the
    node is a single atomic Python call that returns all four outputs at
    once: ComfyUI cannot partially execute it, so bypassing the downstream
    consumer of ``positive_upscale`` / ``latent_upscale`` (an entire
    upscaler chain set to bypass, say) still forces this node to run in
    full for the base-resolution outputs, and the upscale encode happens
    regardless. Do not "fix" this by making the second encode conditional
    on something else -- the node has no visibility into what downstream
    consumes its outputs.
    """

    @classmethod
    def INPUT_TYPES(cls):
        # The optional ref_* slot block is identical to
        # MiniMaxH3CLIPCachedRef2VA's -- shared via _ref_slots_input_spec().
        # cache_mode stays defined here (not in that helper) because its
        # tooltip is tailored: "Applies to both resolutions."
        optional = _ref_slots_input_spec()
        optional["generate_upscale_cond"] = ("BOOLEAN", {"default": True, "tooltip":
            "When off, the second (upscale-resolution) encode is skipped entirely - "
            "positive_upscale/latent_upscale come back as None. Turn off for a plain "
            "generation where nothing downstream uses the upscale outputs; turn on "
            "when you actually need them. Bypassing the upscale consumer downstream "
            "does NOT skip this encode by itself - this is the only thing that does, "
            "because the node runs as one atomic call."})
        optional["cache_mode"] = (["auto", "refresh"], {"default": "auto",
            "tooltip": "auto: reuse the cached encode for an identical prompt + reference "
                       "images/videos/audio + clip_name (checkpoint identity = "
                       "filename+size+mtime+ctime) if one exists, otherwise encode and save it. "
                       "refresh: ignore any cached encode, always re-encode and overwrite "
                       "the cache. Applies to both resolutions.",
        })
        return {
            "required": {
                "clip_name": _clip_name_input_spec(),
                "vae": ("VAE",),
                "audio_vae": ("VAE",),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "width": ("INT", {"default": 1344, "min": 32, "max": nodes.MAX_RESOLUTION, "step": 32}),
                "height": ("INT", {"default": 768, "min": 32, "max": nodes.MAX_RESOLUTION, "step": 32}),
                "width_upscale": ("INT", {"default": 1344, "min": 32, "max": nodes.MAX_RESOLUTION, "step": 32,
                                    "tooltip": "Encoded through the same fully independent cached path as "
                                               "width -- with cache_mode auto a cache HIT when the encoder input ends up"
                                               " identical, otherwise a real encode; "
                                               "cache_mode refresh always re-encodes."}),
                "height_upscale": ("INT", {"default": 768, "min": 32, "max": nodes.MAX_RESOLUTION, "step": 32,
                                     "tooltip": "Encoded through the same fully independent cached path as "
                                                "height -- with cache_mode auto a cache HIT when the encoder input ends up"
                                                " identical, otherwise a real encode; "
                                                "cache_mode refresh always re-encodes."}),
                "length": ("INT", {"default": 124, "min": 5, "max": 3600, "step": 17,
                                    "tooltip": "Frame count at 24 fps, (124 = ~5s, trained range is ~124-362)"}),
                "ref_image_size": (["match", "max"], {"default": "match", "tooltip": _REF_IMAGE_SIZE_TOOLTIP}),
            },
            "optional": optional,
        }

    RETURN_TYPES = ("CONDITIONING", "LATENT", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "latent", "positive_upscale", "latent_upscale")
    FUNCTION = "execute"
    CATEGORY = "model/conditioning/minimax/cached"

    @classmethod
    def IS_CHANGED(cls, clip_name=None, cache_mode="auto", **kwargs):
        # Same contract as the single-resolution node: IS_CHANGED tracks the
        # CLIP checkpoint's identity and the refresh flag, neither of which
        # depends on how many resolutions this run computes.
        return _is_changed_common(clip_name, cache_mode)

    def execute(self, clip_name, vae, audio_vae, prompt, width, height, width_upscale, height_upscale,
                length, ref_image_size="match",
                ref_image_0=None, ref_image_1=None, ref_image_2=None, ref_image_3=None,
                ref_image_4=None, ref_image_5=None, ref_image_6=None, ref_image_7=None,
                ref_image_8=None,
                ref_video_0=None, ref_video_1=None, ref_video_2=None,
                ref_video_audio_0=None, ref_video_audio_1=None, ref_video_audio_2=None,
                ref_audio_0=None, ref_audio_1=None, ref_audio_2=None,
                cache_mode="auto", generate_upscale_cond=True):
        ref_images, ref_videos, ref_video_audios, ref_audios = _build_ref_slot_dicts(
            [ref_image_0, ref_image_1, ref_image_2, ref_image_3, ref_image_4,
             ref_image_5, ref_image_6, ref_image_7, ref_image_8],
            [ref_video_0, ref_video_1, ref_video_2],
            [ref_video_audio_0, ref_video_audio_1, ref_video_audio_2],
            [ref_audio_0, ref_audio_1, ref_audio_2],
        )
        cond, latent, fp1 = _execute_ref2va_once(
            clip_name, vae, audio_vae, prompt, width, height, length,
            ref_image_size, ref_images, ref_videos, ref_video_audios, ref_audios,
            cache_mode,
        )
        if not generate_upscale_cond:
            # Upscale pass switched off: the second _execute_ref2va_once() does
            # not run at all (zero encode / VRAM cost) and there is no second
            # fingerprint, so _pair_verbose_entries() is skipped too. See the
            # class docstring for why bypassing the downstream consumer cannot
            # achieve this on its own.
            logger.info(
                "[UPSCALE COND SKIPPED] %s: generate_upscale_cond=False - "
                "positive_upscale/latent_upscale not computed", fp1[:12],
            )
            return (cond, latent, None, None)
        cond_upscale, latent_upscale, fp2 = _execute_ref2va_once(
            clip_name, vae, audio_vae, prompt, width_upscale, height_upscale, length,
            ref_image_size, ref_images, ref_videos, ref_video_audios, ref_audios,
            cache_mode,
        )
        # Both encodes succeeded (either call raising propagates before here),
        # so it is safe to cross-link the two Cache Manager entries now.
        # fp1 is the base-resolution side, fp2 the upscale-resolution side.
        _pair_verbose_entries(fp1, width, height, fp2, width_upscale, height_upscale,
                              b_is_upscale_target=True)
        return (cond, latent, cond_upscale, latent_upscale)


# --- CLIP Name (standalone encoder picker) ----------------------------------

class MiniMaxH3CLIPName:
    """A one-widget source node for the encoder-checkpoint dropdown.

    It carries the same ``clip_name`` dropdown as MiniMaxH3CLIPCachedFL2VA /
    MiniMaxH3CLIPCachedRef2VA (via the shared ``_clip_name_input_spec()``)
    and simply re-emits the selected filename as a COMBO-typed output. The
    point is to wire that one output into the ``clip_name`` input (after
    "Convert widget to Input") of any number of FL2VA / Ref2VA nodes at
    once, so the encoder is chosen in a single place instead of N.

    See ``_ComboType`` for why the output type is a str subclass rather than
    the string ``"COMBO"`` or a frozen list.

    No IS_CHANGED: the node is deterministic in its single widget, so
    ComfyUI's default input-literal caching is already correct here.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip_name": _clip_name_input_spec(
                    "Pick the MiniMax H3 encoder checkpoint once and feed it into "
                    "multiple FL2VA / Ref2VA nodes' clip_name input (after 'Convert "
                    "widget to Input' on each)."),
            },
        }

    RETURN_TYPES = (_ComboType("COMBO"),)
    RETURN_NAMES = ("clip_name",)
    FUNCTION = "execute"
    CATEGORY = "model/conditioning/minimax/cached"

    def execute(self, clip_name):
        return (clip_name,)
