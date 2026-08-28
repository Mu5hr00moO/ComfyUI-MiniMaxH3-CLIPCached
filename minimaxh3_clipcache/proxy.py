"""CachedClipProxy: transparent clip.tokenize()/encode_from_tokens_scheduled()
replacement that fingerprints the request and serves/saves conditioning from
disk instead of always re-running the real Qwen3-VL encoder.

tokenize() is lazy (phase 4 of the project plan): it does not touch
real_clip at all, it just remembers (prompt, kwargs) and hands that back as
an opaque "tokens" placeholder. All the real work -- fingerprinting, cache
lookup, and only on a MISS the real tokenize()+encode_from_tokens_scheduled()
-- happens in encode_from_tokens_scheduled(), which is where the stock
MiniMaxH3ImageToVideo node hands the tokens straight back to us.
"""

import logging
import threading

from minimaxh3_clipcache.fingerprint import CACHE_SCHEMA_VERSION, compute_fingerprint
from minimaxh3_clipcache.store import load_conditioning, save_conditioning

logger = logging.getLogger(__name__)

# One lock per cache fingerprint, shared across every CachedClipProxy
# instance in the process. It serialises the whole lookup -> encode -> save
# path for a given entry so two racing MISS requests for the same fingerprint
# (e.g. two Queue runs, or a Cache Manager delete landing mid-write) cannot
# both load the ~27 GB encoder at once: the second thread re-checks the cache
# after the first releases the lock and finds the freshly written result.
# Distinct fingerprints get distinct locks, so unrelated encodes still run in
# parallel. The dict grows by one small Lock per unique fingerprint seen in a
# session -- negligible even for thousands of prompts.
_fingerprint_locks: dict[str, threading.Lock] = {}
_fingerprint_locks_guard = threading.Lock()


def _get_lock(fingerprint: str) -> threading.Lock:
    with _fingerprint_locks_guard:
        if fingerprint not in _fingerprint_locks:
            _fingerprint_locks[fingerprint] = threading.Lock()
        return _fingerprint_locks[fingerprint]


# A single process-wide lock around the actual "load the real ~27 GB encoder
# and run it" step, independent of fingerprint. The per-fingerprint lock
# above only prevents the SAME fingerprint from loading twice; two DIFFERENT
# fingerprints (e.g. two different prompts, or FL2VA and Ref2VA in the same
# graph) racing a MISS at the same time would otherwise each call
# clip_loader_fn() and end up with two ~27 GB encoders resident at once.
# This lock forces those onto one at a time. Held only around the real
# load+encode+unload, never around the cache lookup above, so two racing
# cache HITs (or a HIT racing a MISS) are never blocked by it.
_encoder_load_lock = threading.Lock()

MINIMAX_H3_HIDDEN_DIM = 5120  # last dim of the real Qwen3-VL/MiniMax H3
# encoder output. A mismatch here almost always means clip_name points
# at a checkpoint that isn't the MiniMax H3 text/vision encoder.


class CachedClipProxy:
    def __init__(self, clip_loader_fn, clip_name, clip_file_size, clip_mtime_ns, cache_dir,
                 force_refresh=False, unload_fn=None):
        self.clip_loader_fn = clip_loader_fn
        self.clip_name = clip_name
        self.clip_file_size = clip_file_size
        self.clip_mtime_ns = clip_mtime_ns
        self.cache_dir = cache_dir
        self.force_refresh = force_refresh
        # Optional callback(patcher) to release the real encoder as soon as
        # this proxy is done with it, called right after a successful real
        # encode -- before returning control to the stock node, which may
        # still have its own post-encode work left (e.g. FL2VA's keyframe
        # VAE encode). None (the default) means "do nothing here", which is
        # what every proxy-level test that constructs CachedClipProxy
        # directly (no ComfyUI, no real model_management) relies on; nodes.py
        # is the only caller that supplies a real one today.
        self.unload_fn = unload_fn
        self._pending = None
        self._real_clip = None
        self.did_load_real_clip = False

    @property
    def real_clip(self):
        return self._real_clip

    def tokenize(self, prompt, **kwargs):
        self._pending = (prompt, kwargs)
        return self._pending

    def encode_from_tokens_scheduled(self, tokens, unprojected=False, add_dict=None, show_pbar=True):
        # The real comfy.sd.CLIP.encode_from_tokens_scheduled() signature is
        # (self, tokens, unprojected=False, add_dict={}, show_pbar=True). Both
        # stock MiniMax H3 nodes only ever call it as (tokens), so today these
        # never arrive -- but if a caller ever passes them we must fail loudly
        # rather than let a silent TypeError happen, and we must never quietly
        # ignore them: unprojected=True in particular returns a different data
        # representation, so serving a cached (projected) result for it would
        # be silently wrong.
        if unprojected or add_dict or not show_pbar:
            raise RuntimeError(
                "CachedClipProxy.encode_from_tokens_scheduled() was called "
                "with unprojected/add_dict/show_pbar - this proxy does not "
                "support non-default values for these (they would silently "
                "invalidate cached results). If you need this, the caching "
                "logic needs to be extended to include these in the cache "
                "key first."
            )
        prompt, kwargs = tokens
        fingerprint = compute_fingerprint(
            prompt, kwargs, self.clip_name, self.clip_file_size, self.clip_mtime_ns,
            CACHE_SCHEMA_VERSION,
        )

        # Hold the per-fingerprint lock across the ENTIRE lookup -> encode ->
        # save path. A second thread that was blocked here re-runs
        # load_conditioning() below (inside the lock) before deciding to
        # encode, so if the first thread already produced and saved the
        # result it is served as a HIT and the encoder is never loaded twice.
        with _get_lock(fingerprint):
            if not self.force_refresh:
                cond = load_conditioning(fingerprint, self.cache_dir)
                if cond is not None:
                    logger.info("[CACHE HIT] %s", fingerprint[:12])
                    self._validate_output_hidden_dim(cond, fingerprint)
                    return cond
                logger.info("[CACHE MISS] %s", fingerprint[:12])
            else:
                logger.info("[CACHE REFRESH] %s", fingerprint[:12])

            with _encoder_load_lock:
                try:
                    if self._real_clip is None:
                        self._real_clip = self.clip_loader_fn()
                        self.did_load_real_clip = True
                    real_tokens = self._real_clip.tokenize(prompt, **kwargs)
                    cond = self._real_clip.encode_from_tokens_scheduled(real_tokens)
                    # Validate the shape BEFORE persisting: a wrong-checkpoint encode
                    # must never be written to the cache, and the error must surface
                    # here rather than as a cryptic matmul failure later in the sampler.
                    self._validate_output_hidden_dim(cond, fingerprint)
                    # The encode result already exists and was expensive to compute;
                    # persisting it to disk is pure optimisation, not a source of truth.
                    # This is the one place in the project where a broad `except` is
                    # deliberate: a cache-write failure must never discard a completed
                    # encode. (load_conditioning() on the read path stays strict -- see
                    # its docstring -- because there the user has not paid the encode
                    # cost yet and should learn their environment is broken.)
                    try:
                        save_conditioning(fingerprint, cond, self.cache_dir)
                    except Exception as e:
                        logger.warning(
                            "[CACHE WRITE FAILED] %s: could not persist encode result (%s) "
                            "- continuing without caching this result", fingerprint[:12], e,
                        )
                finally:
                    # Release the real encoder BEFORE releasing the lock above,
                    # not after -- otherwise a second thread queued on
                    # _encoder_load_lock could start loading its own ~27 GB
                    # encoder while ours is still resident (e.g. because
                    # _validate_output_hidden_dim() just raised), briefly
                    # recreating the exact two-encoders-at-once situation this
                    # lock exists to prevent. Guarded so a HIT-only call (which
                    # never enters this block) and a proxy built without
                    # unload_fn (every proxy-level test) are unaffected.
                    if self.unload_fn is not None and self._real_clip is not None:
                        self.unload_fn(self._real_clip.patcher)
                        self._real_clip = None
            return cond

    def _validate_output_hidden_dim(self, cond, fingerprint):
        """Fail loudly if the conditioning's hidden dim isn't the MiniMax H3
        encoder's. Runs on both the HIT and the MISS/REFRESH path, and on a
        MISS strictly before save_conditioning() so a wrong-checkpoint encode
        is never written to disk. Per CLAUDE.md's "no silent fallbacks" rule
        this exception must propagate out of the stock node's execute() and
        abort the graph -- it is not caught anywhere upstream.

        cond is [[main_tensor, {"pooled_output": ..., ...}]] (see
        minimaxh3_clipcache.serialize and the on-disk skeletons), so the main
        encoded tensor is cond[0][0].
        """
        main_tensor = cond[0][0]
        hidden_dim = main_tensor.shape[-1]
        if hidden_dim != MINIMAX_H3_HIDDEN_DIM:
            raise RuntimeError(
                "MiniMax H3 CLIP-Cached: encoded conditioning has hidden dim "
                "{}, expected {}. clip_name='{}' is very likely NOT the MiniMax "
                "H3 text/vision encoder (Qwen3-VL) - check the clip_name "
                "dropdown on this node. (fingerprint={})".format(
                    hidden_dim, MINIMAX_H3_HIDDEN_DIM, self.clip_name, fingerprint[:12],
                )
            )
