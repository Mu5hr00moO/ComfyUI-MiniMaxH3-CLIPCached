"""Sidecar "<fingerprint>.verbose.json" metadata for the Cache Manager.

This file is the human-facing index layer over the encode cache. It holds
two logically separate blocks:

  "system" -- a description of what was actually cached (exact prompt,
              encoder identity, cache schema version, reference info). The
              manager refreshes this from verified data every time it sees
              the entry again; it is read-only from the user's point of view.
  "user"   -- the organisational fields the user edits: name, notes, tags,
              favorite (see DEFAULT_USER_METADATA).

It is deliberately NOT part of the source of truth. store.py's
"<fingerprint>.json" + "<fingerprint>.safetensors" pair is what HIT/MISS
depends on; a missing, truncated or corrupt ".verbose.json" must never look
like a core-cache failure. load_verbose() answers None for a plain missing
file (silently -- a legacy or not-yet-described entry) and None for a file
it cannot parse (with a WARNING), exactly the way store.load_conditioning()
treats a cache it cannot trust.

Why this module does NOT carry store.py's multi-file rollback logic
------------------------------------------------------------------
save_conditioning() writes two coupled artefacts (.safetensors, then .json)
and, if the second write fails after the first is already in place, must
delete what it created so a half-written entry never lingers -- hence its
"created[]" tracking. A verbose sidecar is a SINGLE file: writing one fully
formed temp file and os.replace()-ing it onto the final path is atomic on
its own, so there is nothing to coordinate and no cleanup list to keep. The
absence of that logic here is a consequence of the artefact count, not an
oversight.

Only the temp-name convention is shared with store.py (via _tmp_name():
PID + uuid4 hex, keeping concurrent writers -- across processes and within
one process -- off each other's temp path).
"""

import copy
import json
import logging
import os
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_USER_METADATA = {"name": "", "notes": "", "tags": [], "favorite": False}


def _verbose_path(fingerprint: str, cache_dir: Path) -> Path:
    return Path(cache_dir) / "{}.verbose.json".format(fingerprint)


def _tmp_name(path: Path) -> Path:
    # Same convention as minimaxh3_clipcache.store._tmp_name: the PID keeps
    # concurrent processes apart, the uuid4 additionally keeps two
    # concurrent writers of the same fingerprint within one process from
    # sharing a temp path and clobbering each other.
    return path.with_name("{}.tmp-{}-{}".format(path.name, os.getpid(), uuid.uuid4().hex))


def load_verbose(fingerprint: str, cache_dir: Path):
    """Return the parsed "<fingerprint>.verbose.json" object, or None.

    None (silent) when the file simply does not exist. None (logged at
    WARNING) when it exists but cannot be read, is not valid JSON, or is
    not a JSON object -- corruption of this sidecar is still not a
    core-cache error.
    """
    path = _verbose_path(fingerprint, cache_dir)
    if not path.exists():
        return None

    try:
        data = json.loads(path.read_bytes())
    except (OSError, ValueError) as e:
        # ValueError covers json.JSONDecodeError (malformed JSON) and
        # UnicodeDecodeError (bytes that aren't valid UTF-8). Same handling
        # store.load_conditioning() gives an unreadable ".json".
        logger.warning("Verbose metadata for fingerprint %s at %s is unreadable, ignoring it: %s",
                       fingerprint, path, e)
        return None

    if not isinstance(data, dict):
        logger.warning("Verbose metadata for fingerprint %s at %s is not a JSON object (%s), ignoring it",
                       fingerprint, path, type(data).__name__)
        return None

    return data


def save_verbose(fingerprint: str, system: dict, cache_dir: Path) -> None:
    """Write "<fingerprint>.verbose.json", refreshing "system" while
    preserving any "user" block already on disk.

    A valid existing sidecar's "user" block is carried over untouched -- the
    user owns those fields and a system backfill must never overwrite them.
    A missing, unreadable, or structurally unusable sidecar starts from a
    fresh copy of DEFAULT_USER_METADATA.

    The write is atomic: a fully formed temp file (PID + uuid4 name) is
    os.replace()-d onto the final path. Being a single file, there is no
    partially-written multi-file state to roll back if this raises -- see
    the module docstring.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    existing = load_verbose(fingerprint, cache_dir)
    if isinstance(existing, dict) and isinstance(existing.get("user"), dict):
        user = existing["user"]
    else:
        user = copy.deepcopy(DEFAULT_USER_METADATA)

    payload = {"fingerprint": fingerprint, "system": system, "user": user}

    path = _verbose_path(fingerprint, cache_dir)
    tmp_path = _tmp_name(path)
    tmp_path.write_bytes(json.dumps(payload).encode("utf-8"))
    os.replace(tmp_path, path)
