"""Filesystem-only scan of the cache directory for the Cache Manager's
"Check" action (plan section 11).

This module never loads tensor data. For a pair of core files it does parse
the small JSON envelope and open only the safetensors header so the Cache
Manager reports the same generation-id mismatch that store.load_conditioning()
would treat as a MISS. "Check" stays cheap enough to run on every panel
refresh; the route runs it outside the aiohttp event loop.

Classification (plan section 11.1), by core-pair consistency and sidecar
existence:

  normal  -- "<fp>.json" AND "<fp>.safetensors" AND "<fp>.verbose.json"
  legacy  -- "<fp>.json" AND "<fp>.safetensors", but no "<fp>.verbose.json"
  inconsistent -- both core files exist, but their envelope/header cannot be
                  read or their generation ids do not match

An orphan ".safetensors" (no ".json") is swept off disk at the start of
every scan (store.gc_orphaned_cache_files()), so it is neither an entry nor
part of total_size_bytes. An orphan ".verbose.json" (no core cache) and a
lone ".json" (no ".safetensors") are NOT entries either, but are only
counted toward total_size_bytes, never deleted: unlike the orphan
".safetensors" they have no single safe, unambiguous removal condition, so
the asymmetry is deliberate.

Edge case, documented on purpose: if "<fp>.verbose.json" EXISTS but is
corrupt, the entry is still classified "normal" (the file exists) while its
"verbose" value is None (load_verbose() returns None on corruption).
Verbose classification is about sidecar existence, not sidecar validity --
this is intentional, not an oversight.

total_size_bytes is the recursive sum of every file left under cache_dir
after the orphan-".safetensors" sweep -- entries, ".verbose.json"/".json"
orphans, thumbnails, stray temp files, anything -- i.e. the "size" figure
from plan section 13.6, not just the normal/legacy entries.

Each entry additionally carries its own "size_bytes": the bytes that
deleting THAT entry would free, and nothing else. The two figures are not
built from each other -- the per-entry sum covers only the files listed by
entry_file_paths(), so the orphans and stray files inside total_size_bytes
belong to no entry and the per-entry values do not add up to the total.

Fingerprints are always 64 lowercase hex chars (sha256). Files are matched
by anchored regex rather than a bare glob("*.json"), so "<fp>.verbose.json"
is never mistaken for a core "<fp>.json".
"""

import logging
import os
import re
from pathlib import Path

from minimaxh3_clipcache.locking import get_lock
from minimaxh3_clipcache.store import (
    conditioning_paths,
    gc_orphaned_cache_files,
    inspect_conditioning_pair,
)
from minimaxh3_clipcache.thumbnails import thumbnail_paths
from minimaxh3_clipcache.verbose_store import load_verbose, verbose_paths

logger = logging.getLogger(__name__)

_FP = r"[0-9a-f]{64}"
CORE_JSON_RE = re.compile(r"^({})\.json$".format(_FP))
SAFETENSORS_RE = re.compile(r"^({})\.safetensors$".format(_FP))
VERBOSE_RE = re.compile(r"^({})\.verbose\.json$".format(_FP))


def _dir_size_bytes(root: Path) -> int:
    total = 0
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            try:
                total += os.path.getsize(os.path.join(dirpath, name))
            except OSError:
                pass  # file vanished between listing and stat -- ignore
    return total


def entry_file_paths(fingerprint, cache_dir) -> list[Path]:
    """Every file that deleting this cache entry would remove.

    Composed from the three per-artifact path helpers that the corresponding
    delete steps use (routes._delete_entry_files -> store.delete_conditioning,
    verbose_store.delete_verbose, thumbnails.delete_thumbnails), so "size of
    this entry" and "what Delete frees" can never drift apart. Nothing here
    re-spells a filename suffix; add a new per-entry artifact to its own
    module's path helper and both sides pick it up.

    Paths for the core pair and the sidecar are returned whether or not the
    files exist -- a legacy entry has no ".verbose.json", which is normal, not
    an error -- so callers must tolerate a missing file.
    """
    return [
        *conditioning_paths(fingerprint, cache_dir),
        *verbose_paths(fingerprint, cache_dir),
        *thumbnail_paths(fingerprint, cache_dir),
    ]


def _entry_size_bytes(fingerprint, cache_dir) -> int:
    """On-disk bytes of one entry: the files a Delete of it would free.

    Deliberately a separate set of stat() calls rather than a slice of the
    _dir_size_bytes() walk. Reusing that walk would mean re-deriving which
    filename belongs to which entry inside this module -- a second spelling of
    the naming convention that entry_file_paths() exists to avoid -- to save a
    handful of stat calls on a directory that is already being walked.

    A file that is missing (legacy entry, no thumbnails) or vanishes mid-scan
    simply contributes nothing.
    """
    total = 0
    for path in entry_file_paths(fingerprint, cache_dir):
        try:
            total += path.stat().st_size
        except OSError:
            pass
    return total


def _stable_pair_issue(fingerprint, cache_dir):
    """Return a stable inconsistency reason, or None.

    A successful save briefly exposes new safetensors under old JSON between
    its two atomic replacements. If an optimistic inspection sees a problem,
    re-check under the writer's per-fingerprint lock. A busy lock means the
    state is actively changing, so defer the warning until the next Check
    instead of showing a false persistent-inconsistency alert.
    """
    valid, reason = inspect_conditioning_pair(fingerprint, cache_dir)
    if valid:
        return None

    lock = get_lock(fingerprint)
    if not lock.acquire(blocking=False):
        return None
    try:
        valid, reason = inspect_conditioning_pair(fingerprint, cache_dir)
        return None if valid else reason
    finally:
        lock.release()


def scan_cache(cache_dir) -> dict:
    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        return {"entries": [], "total_count": 0, "total_size_bytes": 0}

    # Sweep orphaned ".safetensors" (no matching ".json") BEFORE listing
    # files, so a just-removed file can never be picked up in this same
    # pass and so its bytes are already gone from the size total below.
    removed = gc_orphaned_cache_files(cache_dir)
    if removed:
        logger.info("[CACHE MANAGER GC] Check swept %d orphaned .safetensors file(s)", len(removed))

    core_json = set()
    safetensors = set()
    verbose = set()
    for child in cache_dir.iterdir():
        if not child.is_file():
            continue
        for regex, bucket in ((CORE_JSON_RE, core_json),
                              (SAFETENSORS_RE, safetensors),
                              (VERBOSE_RE, verbose)):
            m = regex.match(child.name)
            if m:
                bucket.add(m.group(1))
                break

    entries = []
    for fp in sorted(core_json & safetensors):
        # Every classification carries size_bytes, including "inconsistent":
        # a broken pair still occupies disk, and its row offers the same
        # Delete as any other.
        size_bytes = _entry_size_bytes(fp, cache_dir)
        issue = _stable_pair_issue(fp, cache_dir)
        if issue is not None:
            entries.append({
                "fingerprint": fp,
                "classification": "inconsistent",
                "reason": issue,
                "size_bytes": size_bytes,
                "verbose": load_verbose(fp, cache_dir) if fp in verbose else None,
            })
            continue
        if fp in verbose:
            entries.append({
                "fingerprint": fp,
                "classification": "normal",
                "size_bytes": size_bytes,
                "verbose": load_verbose(fp, cache_dir),
            })
        else:
            entries.append({
                "fingerprint": fp,
                "classification": "legacy",
                "size_bytes": size_bytes,
                "verbose": None,
            })

    return {
        "entries": entries,
        "total_count": len(entries),
        "total_size_bytes": _dir_size_bytes(cache_dir),
    }
