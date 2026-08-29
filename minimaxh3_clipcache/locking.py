"""Per-fingerprint locks shared by every module that mutates a cache entry's
files: proxy.py (save on MISS/refresh), nodes.py (verbose backfill after a
successful execute()), and routes.py (Cache Manager /update and /delete).

Holding the SAME lock, keyed by fingerprint, across all of these is what
prevents a Cache Manager Delete from interleaving with an in-flight save, and
what prevents a concurrent /update losing a user's edit to a racing system
metadata backfill (both do read-modify-write on the same verbose.json).

One lock per fingerprint, not a single global lock: unrelated fingerprints
must stay free to save/delete/update concurrently -- this is deliberately
narrower than the separate, global _encoder_load_lock in proxy.py, which
protects VRAM (only one 27 GB encoder resident at a time), not cache-file
consistency.

The dict grows by one small Lock per unique fingerprint ever seen in the
process and locks are never removed -- negligible memory even for thousands of
entries, and removing a lock while another thread might be about to acquire it
would itself be a correctness hazard.
"""

import threading

_fingerprint_locks: dict[str, threading.Lock] = {}
_fingerprint_locks_guard = threading.Lock()


def get_lock(fingerprint: str) -> threading.Lock:
    with _fingerprint_locks_guard:
        if fingerprint not in _fingerprint_locks:
            _fingerprint_locks[fingerprint] = threading.Lock()
        return _fingerprint_locks[fingerprint]
