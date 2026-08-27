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

from minimaxh3_clipcache.fingerprint import CACHE_SCHEMA_VERSION, compute_fingerprint
from minimaxh3_clipcache.store import load_conditioning, save_conditioning

logger = logging.getLogger(__name__)

MINIMAX_H3_HIDDEN_DIM = 5120  # last dim of the real Qwen3-VL/MiniMax H3
# encoder output. A mismatch here almost always means clip_name points
# at a checkpoint that isn't the MiniMax H3 text/vision encoder.


class CachedClipProxy:
    def __init__(self, clip_loader_fn, clip_name, clip_file_size, clip_mtime_ns, cache_dir,
                 force_refresh=False):
        self.clip_loader_fn = clip_loader_fn
        self.clip_name = clip_name
        self.clip_file_size = clip_file_size
        self.clip_mtime_ns = clip_mtime_ns
        self.cache_dir = cache_dir
        self.force_refresh = force_refresh
        self._pending = None
        self._real_clip = None
        self.did_load_real_clip = False
        # Set by encode_from_tokens_scheduled() so nodes.py can, after the
        # stock execute() returns, tell what this run did and drive the
        # Cache Manager's verbose-metadata write/backfill (plan phase 2).
        self.last_fingerprint = None
        self.last_hit = None                 # True/False, None until anything is computed
        self.last_core_cache_written = None  # True/False/None (None = write not attempted)

    @property
    def real_clip(self):
        return self._real_clip

    def tokenize(self, prompt, **kwargs):
        self._pending = (prompt, kwargs)
        return self._pending

    def encode_from_tokens_scheduled(self, tokens):
        prompt, kwargs = tokens
        fingerprint = compute_fingerprint(
            prompt, kwargs, self.clip_name, self.clip_file_size, self.clip_mtime_ns,
            CACHE_SCHEMA_VERSION,
        )
        self.last_fingerprint = fingerprint

        if not self.force_refresh:
            cond = load_conditioning(fingerprint, self.cache_dir)
            if cond is not None:
                logger.info("[CACHE HIT] %s", fingerprint[:12])
                self._validate_output_hidden_dim(cond, fingerprint)
                self.last_hit = True
                return cond
            logger.info("[CACHE MISS] %s", fingerprint[:12])
        else:
            logger.info("[CACHE REFRESH] %s", fingerprint[:12])

        # Past the early HIT return: this run is a MISS or a REFRESH.
        self.last_hit = False

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
            self.last_core_cache_written = True
        except Exception as e:
            self.last_core_cache_written = False
            logger.warning(
                "[CACHE WRITE FAILED] %s: could not persist encode result (%s) "
                "- continuing without caching this result", fingerprint[:12], e,
            )
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
