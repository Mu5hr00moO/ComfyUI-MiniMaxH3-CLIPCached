"""Atomic, pickle-free disk cache for CONDITIONING results, keyed by fingerprint.

Two files per entry: "<fingerprint>.safetensors" (tensors) and
"<fingerprint>.json" (the skeleton from minimaxh3_clipcache.serialize.flatten_tensors).
Both are written to a temp file in cache_dir and moved into place with
os.replace(), which is atomic on the same filesystem.

Write order is deliberate: tensors first, then the skeleton. The skeleton is
the file whose mere presence load_conditioning() trusts as "this entry was
written"; writing it last means a fully-visible .json always has a matching,
fully-written .safetensors behind it. Any state where .json exists without a
usable .safetensors could only happen from something after our write (manual
tampering, a killed process mid-.safetensors-write followed by an
independently created .json, disk corruption) -- not from a failure inside
save_conditioning() itself, since os.replace() only makes .safetensors
visible once fully written.
"""

import json
import logging
import os
import uuid
from pathlib import Path

from safetensors import SafetensorError
from safetensors.torch import load_file, save_file

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

    # Every path this call has brought into existence, in creation order: the
    # .safetensors temp, then (once os.replace() moves it into place) the
    # final .safetensors, then the .json temp, then the final .json. If any
    # step raises -- classically the .json write failing after the
    # .safetensors is already in place -- we delete all of them so a failed
    # save leaves nothing behind: no stray .tmp-*, and no .safetensors
    # without a matching .json. The exception is then re-raised untouched;
    # it is proxy.py, not store.py, that decides whether a cache-write
    # failure after a costly encode should be swallowed with a WARNING.
    created = []
    try:
        tmp_safetensors_path = _tmp_name(safetensors_path)
        save_file({k: v.detach().cpu().contiguous() for k, v in tensors.items()}, str(tmp_safetensors_path))
        created.append(tmp_safetensors_path)

        os.replace(tmp_safetensors_path, safetensors_path)
        created[-1] = safetensors_path

        tmp_json_path = _tmp_name(json_path)
        tmp_json_path.write_bytes(json.dumps(skeleton).encode("utf-8"))
        created.append(tmp_json_path)

        os.replace(tmp_json_path, json_path)
        created[-1] = json_path
    except BaseException:
        for path in created:
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
