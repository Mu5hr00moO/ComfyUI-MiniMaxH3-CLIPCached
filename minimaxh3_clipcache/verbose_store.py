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

_FIELD_TYPES = {"name": str, "notes": str, "tags": list, "favorite": bool}
_MAX_TEXT_LENGTH = 500
_MAX_TAGS = 50


def _verbose_path(fingerprint: str, cache_dir: Path) -> Path:
    return Path(cache_dir) / "{}.verbose.json".format(fingerprint)


def _tmp_name(path: Path) -> Path:
    # Same convention as minimaxh3_clipcache.store._tmp_name: the PID keeps
    # concurrent processes apart, the uuid4 additionally keeps two
    # concurrent writers of the same fingerprint within one process from
    # sharing a temp path and clobbering each other.
    return path.with_name("{}.tmp-{}-{}".format(path.name, os.getpid(), uuid.uuid4().hex))


def _atomic_write_json(path: Path, payload: dict) -> None:
    # Single file: a fully formed temp file os.replace()-d onto the final
    # path is atomic on its own, so there is nothing to roll back if this
    # raises (see the module docstring for why this differs from store.py).
    tmp_path = _tmp_name(path)
    tmp_path.write_bytes(json.dumps(payload).encode("utf-8"))
    os.replace(tmp_path, path)


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
    _atomic_write_json(_verbose_path(fingerprint, cache_dir), payload)


def update_user_metadata(fingerprint: str, updates: dict, cache_dir: Path) -> dict:
    """Apply a partial update to the "user" block of an existing verbose
    sidecar and return the full updated object.

    Only keys present in DEFAULT_USER_METADATA may be updated; any other key
    is a ValueError. Each given value is also type-checked (name/notes: str,
    tags: list of str, favorite: exactly bool) and bounded (name/notes <=
    500 chars, tags <= 50 entries); a violation is a ValueError too. "system"
    is left exactly as it was -- the Cache Manager never edits it (plan
    sections 5.1/6). Raises FileNotFoundError when there is no readable
    verbose sidecar for this fingerprint: the manager only edits entries that
    already exist (plan section 16 -> 404 in the handler).
    """
    unknown = set(updates) - set(DEFAULT_USER_METADATA)
    if unknown:
        raise ValueError("unknown user metadata field(s): {}".format(sorted(unknown)))

    for key, value in updates.items():
        expected_type = _FIELD_TYPES[key]
        if not isinstance(value, expected_type) or isinstance(value, bool) and expected_type is not bool:
            raise ValueError(
                "field '{}' must be {}, got {}".format(key, expected_type.__name__, type(value).__name__))
    if "tags" in updates:
        if not all(isinstance(t, str) for t in updates["tags"]):
            raise ValueError("all tags must be strings")
        if len(updates["tags"]) > _MAX_TAGS:
            raise ValueError("too many tags (max {})".format(_MAX_TAGS))
    for key in ("name", "notes"):
        if key in updates and len(updates[key]) > _MAX_TEXT_LENGTH:
            raise ValueError("field '{}' exceeds max length {}".format(key, _MAX_TEXT_LENGTH))

    existing = load_verbose(fingerprint, cache_dir)
    if existing is None:
        raise FileNotFoundError(
            "no verbose metadata to update for fingerprint {}".format(fingerprint))

    user = existing.get("user")
    if not isinstance(user, dict):
        user = copy.deepcopy(DEFAULT_USER_METADATA)
    user.update(updates)
    existing["user"] = user

    _atomic_write_json(_verbose_path(fingerprint, Path(cache_dir)), existing)
    return existing


def delete_verbose(fingerprint: str, cache_dir: Path) -> None:
    """Remove "<fingerprint>.verbose.json" if it exists.

    Idempotent: a missing file is not an error, so a Delete can be retried
    on the same fingerprint without blowing up.
    """
    try:
        _verbose_path(fingerprint, Path(cache_dir)).unlink()
    except FileNotFoundError:
        pass
