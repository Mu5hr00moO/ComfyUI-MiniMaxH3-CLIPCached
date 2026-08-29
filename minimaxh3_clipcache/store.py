"""Atomic, pickle-free disk cache for CONDITIONING results, keyed by fingerprint.

Two files per entry: "<fingerprint>.safetensors" (tensors) and
"<fingerprint>.json" (the skeleton from minimaxh3_clipcache.serialize.flatten_tensors).
Both are written to a temp file in cache_dir and moved into place with
os.replace(), which is atomic on the same filesystem.

Write order is deliberate: tensors first, then the skeleton. The skeleton is
the file whose mere presence load_conditioning() trusts as "this entry was
written"; publishing it last means a fully-visible .json always has a
matching, fully-written .safetensors behind it.

save_conditioning() runs in two phases. Phase 1 writes both files to temp
names in cache_dir; on failure it removes its own temp files and never
touches an existing entry for this fingerprint -- that is what makes a
failed cache_mode="refresh" safe. Phase 2 publishes the two temp files with
os.replace() (safetensors first). The pair cannot be made atomic across two
separate files: if either replace raises, Phase 2 removes only its own
never-consumed temp file(s) by name and re-raises -- it never undoes a
replace that already happened. A failed .json publish therefore leaves a
fresh write as an orphan .safetensors with no .json (self-healed by
load_conditioning(), or swept by gc_orphaned_cache_files()), and a refresh
with new tensors under the previous skeleton -- a narrow window that
degrades to an ordinary MISS via unflatten_tensors(), never a crash.
"""

import json
import logging
import os
import uuid
from pathlib import Path

from safetensors import SafetensorError
from safetensors.torch import load_file, save_file

from minimaxh3_clipcache.locking import get_lock
from minimaxh3_clipcache.serialize import flatten_tensors, unflatten_tensors

logger = logging.getLogger(__name__)


def _tmp_name(path: Path) -> Path:
    # PID keeps concurrent processes apart; uuid4 additionally keeps two
    # concurrent writers of the same fingerprint *within one process*
    # (e.g. two threads) from sharing a temp path and clobbering each other.
    return path.with_name("{}.tmp-{}-{}".format(path.name, os.getpid(), uuid.uuid4().hex))


def save_conditioning(fingerprint: str, cond, cache_dir: Path) -> None:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    skeleton, tensors = flatten_tensors(cond)

    safetensors_path = cache_dir / "{}.safetensors".format(fingerprint)
    json_path = cache_dir / "{}.json".format(fingerprint)
    tmp_safetensors_path = _tmp_name(safetensors_path)
    tmp_json_path = _tmp_name(json_path)

    # Write BOTH temp files completely before replacing either final path.
    # A failure up to this point never touches an existing, valid pair at
    # all -- this is what makes a failed cache_mode="refresh" of an already-
    # cached fingerprint safe: the old entry is untouched until both new
    # files are fully written and ready to publish. Cleanup here only ever
    # removes our own temp files, never a final path that predates this call.
    try:
        save_file({k: v.detach().cpu().contiguous() for k, v in tensors.items()}, str(tmp_safetensors_path))
        tmp_json_path.write_bytes(json.dumps(skeleton).encode("utf-8"))
    except BaseException:
        for path in (tmp_safetensors_path, tmp_json_path):
            try:
                path.unlink()
            except OSError:
                pass
        raise

    # Phase 2: publish. safetensors first (module write-order invariant).
    # If EITHER os.replace() raises, clean up only our own leftover temp
    # file(s) -- by name, so this can never touch safetensors_path/
    # json_path themselves, whether or not the first replace already
    # succeeded. We deliberately do NOT undo a replace that already
    # happened: for a refresh of an existing entry that would destroy it
    # outright (old skeleton, no tensors -- guaranteed broken); for a fresh
    # write it just leaves a self-healable orphan .safetensors
    # (load_conditioning()'s own self-heal, or gc_orphaned_cache_files(),
    # already clean these up). Only the never-consumed temp file(s) are
    # ours to remove here.
    try:
        os.replace(tmp_safetensors_path, safetensors_path)
        os.replace(tmp_json_path, json_path)
    except BaseException:
        for path in (tmp_safetensors_path, tmp_json_path):
            try:
                path.unlink()
            except OSError:
                pass
        raise


def delete_conditioning(fingerprint: str, cache_dir: Path) -> None:
    """Remove a cache entry's core files.

    Order is deliberate and mirrors the write order in reverse (plan
    sections 15/20): the skeleton ".json" goes first, so the entry stops
    being a HIT the instant it is gone (load_conditioning() returns None as
    soon as ".json" is absent), then the ".safetensors" payload.

    Idempotent: a missing file is silently skipped, so a Delete can be
    retried on the same fingerprint without raising.
    """
    cache_dir = Path(cache_dir)
    for suffix in (".json", ".safetensors"):
        try:
            (cache_dir / "{}{}".format(fingerprint, suffix)).unlink()
        except FileNotFoundError:
            pass


def load_conditioning(fingerprint: str, cache_dir: Path):
    cache_dir = Path(cache_dir)
    json_path = cache_dir / "{}.json".format(fingerprint)
    safetensors_path = cache_dir / "{}.safetensors".format(fingerprint)

    if not json_path.exists():
        if safetensors_path.exists():
            # A .safetensors with no matching .json can only be left behind
            # by an interrupted write (save_conditioning() writes tensors
            # first, the skeleton last -- see the module docstring): the
            # tensors aren't corrupt, they were just orphaned mid-write.
            # Self-heal it now so this fingerprint doesn't accumulate an
            # ever-growing dead file every time it MISSes and gets
            # re-encoded and re-saved. A fingerprint that is never looked up
            # again is not covered by this -- see gc_orphaned_cache_files()
            # below for a directory-wide sweep.
            logger.warning(
                "Cache MISS for fingerprint %s: found orphaned %s with no "
                "matching %s (likely an interrupted write) - removing it",
                fingerprint, safetensors_path, json_path,
            )
            try:
                safetensors_path.unlink()
            except OSError:
                pass  # best-effort cleanup, not worth failing the MISS over
        return None  # ordinary MISS, not logged

    try:
        skeleton = json.loads(json_path.read_bytes())
    except (OSError, ValueError) as e:
        # ValueError covers both json.JSONDecodeError (malformed JSON) and
        # UnicodeDecodeError (bytes that aren't even valid UTF-8, e.g. a
        # .json file overwritten with random garbage).
        logger.warning("Cache MISS for fingerprint %s: failed to read/parse %s: %s",
                        fingerprint, json_path, e)
        return None

    if not safetensors_path.exists():
        logger.warning("Cache MISS for fingerprint %s: %s exists but %s is missing",
                        fingerprint, json_path, safetensors_path)
        return None

    try:
        tensors = load_file(str(safetensors_path))
    except SafetensorError as e:
        logger.warning("Cache MISS for fingerprint %s: failed to load %s: %s",
                        fingerprint, safetensors_path, e)
        return None

    try:
        return unflatten_tensors(skeleton, tensors)
    except (KeyError, TypeError, IndexError) as e:
        # skeleton and tensors each parsed fine on their own but disagree
        # with each other (e.g. skeleton references a tensor path that
        # isn't in the safetensors file) -- same "cache is not the source
        # of truth" MISS handling as the read/parse failures above.
        logger.warning("Cache MISS for fingerprint %s: skeleton/tensors mismatch while reconstructing: %s",
                        fingerprint, e)
        return None


def gc_orphaned_cache_files(cache_dir: Path) -> list[str]:
    """Remove every "<fingerprint>.safetensors" in cache_dir that has no
    matching "<fingerprint>.json", and return the list of fingerprints
    removed. These can only be left behind by a process killed between
    the two os.replace() calls in save_conditioning() (tensors first,
    skeleton last); load_conditioning() already self-heals a
    fingerprint's own orphan the next time that exact fingerprint
    MISSes, but a fingerprint that is never looked up again would
    otherwise accumulate a dead file forever.

    Called automatically on every Cache Manager "Check" (scanner.scan_cache()),
    so this MUST be safe to run concurrently with an in-flight
    save_conditioning() for a different (or even the same) fingerprint -
    see the per-fingerprint lock below.
    """
    cache_dir = Path(cache_dir)
    if not cache_dir.is_dir():
        return []
    removed = []
    for safetensors_path in cache_dir.glob("*.safetensors"):
        fingerprint = safetensors_path.stem
        json_path = cache_dir / "{}.json".format(fingerprint)
        if json_path.exists():
            continue
        lock = get_lock(fingerprint)
        if not lock.acquire(blocking=False):
            # A writer is actively publishing this exact fingerprint right
            # now (save_conditioning() holds this same lock for its whole
            # duration, including the window between its two os.replace()
            # calls). What looks orphaned this instant may simply be
            # mid-publish. Skip it this sweep - if it is genuinely
            # orphaned it will still be here, and unlocked, on the next
            # Check.
            continue
        try:
            # Re-check under the lock: the writer may have finished and
            # published .json in the gap between our check above and
            # acquiring the lock.
            if json_path.exists():
                continue
            safetensors_path.unlink()
            removed.append(fingerprint)
        except OSError as e:
            logger.warning("Failed to remove orphaned cache file %s: %s", safetensors_path, e)
        finally:
            lock.release()
    return removed
