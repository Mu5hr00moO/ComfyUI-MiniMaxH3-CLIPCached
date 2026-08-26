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

from caching.fingerprint import CACHE_SCHEMA_VERSION, compute_fingerprint
from caching.store import load_conditioning, save_conditioning

logger = logging.getLogger(__name__)


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

    def tokenize(self, prompt, **kwargs):
        self._pending = (prompt, kwargs)
        return self._pending

    def encode_from_tokens_scheduled(self, tokens):
        prompt, kwargs = tokens
        fingerprint = compute_fingerprint(
            prompt, kwargs, self.clip_name, self.clip_file_size, self.clip_mtime_ns,
            CACHE_SCHEMA_VERSION,
        )

        if not self.force_refresh:
            cond = load_conditioning(fingerprint, self.cache_dir)
            if cond is not None:
                logger.info("[CACHE HIT] %s", fingerprint[:12])
                return cond
            logger.info("[CACHE MISS] %s", fingerprint[:12])
        else:
            logger.info("[CACHE REFRESH] %s", fingerprint[:12])

        if self._real_clip is None:
            self._real_clip = self.clip_loader_fn()
            self.did_load_real_clip = True
        real_tokens = self._real_clip.tokenize(prompt, **kwargs)
        cond = self._real_clip.encode_from_tokens_scheduled(real_tokens)
        save_conditioning(fingerprint, cond, self.cache_dir)
        return cond
