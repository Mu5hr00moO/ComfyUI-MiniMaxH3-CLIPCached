"""Filesystem-only scan of the cache directory for the Cache Manager's
"Check" action (plan section 11).

This module never opens a ".safetensors" file and never parses tensors:
classification is based *purely on which files exist*, exactly the way
store.load_conditioning() decides HIT vs MISS (it returns None the moment
"<fp>.json" is absent, without ever touching the tensors). "Check" must stay
cheap enough to run on every panel refresh. (It does import
store.gc_orphaned_cache_files, which transitively imports torch/safetensors
-- a no-op cost inside a running ComfyUI, where both are already loaded.)

Classification (plan section 11.1), by file existence only:

  normal  -- "<fp>.json" AND "<fp>.safetensors" AND "<fp>.verbose.json"
  legacy  -- "<fp>.json" AND "<fp>.safetensors", but no "<fp>.verbose.json"

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
Classification is about file existence, not file validity -- this is
intentional, not an oversight.

total_size_bytes is the recursive sum of every file left under cache_dir
after the orphan-".safetensors" sweep -- entries, ".verbose.json"/".json"
orphans, thumbnails, stray temp files, anything -- i.e. the "size" figure
from plan section 13.6, not just the normal/legacy entries.

Fingerprints are always 64 lowercase hex chars (sha256). Files are matched
by anchored regex rather than a bare glob("*.json"), so "<fp>.verbose.json"
is never mistaken for a core "<fp>.json".
"""

import logging
import os
import re
from pathlib import Path

from minimaxh3_clipcache.store import gc_orphaned_cache_files
from minimaxh3_clipcache.verbose_store import load_verbose

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
        if fp in verbose:
            entries.append({
                "fingerprint": fp,
                "classification": "normal",
                "verbose": load_verbose(fp, cache_dir),
            })
        else:
            entries.append({
                "fingerprint": fp,
                "classification": "legacy",
                "verbose": None,
            })

    return {
        "entries": entries,
        "total_count": len(entries),
        "total_size_bytes": _dir_size_bytes(cache_dir),
    }
