"""Atomic, pickle-free disk cache for CONDITIONING results, keyed by fingerprint.

Two files per entry: "<fingerprint>.safetensors" (tensors) and
"<fingerprint>.json" (the skeleton from caching.serialize.flatten_tensors).
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
from pathlib import Path

from safetensors import SafetensorError
from safetensors.torch import load_file, save_file

from caching.serialize import flatten_tensors, unflatten_tensors

logger = logging.getLogger(__name__)


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    tmp_path = path.with_name("{}.tmp-{}".format(path.name, os.getpid()))
    tmp_path.write_bytes(data)
    os.replace(tmp_path, path)


def save_conditioning(fingerprint: str, cond, cache_dir: Path) -> None:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    skeleton, tensors = flatten_tensors(cond)

    safetensors_path = cache_dir / "{}.safetensors".format(fingerprint)
    tmp_safetensors_path = safetensors_path.with_name("{}.tmp-{}".format(safetensors_path.name, os.getpid()))
    save_file({k: v.detach().cpu().contiguous() for k, v in tensors.items()}, str(tmp_safetensors_path))
    os.replace(tmp_safetensors_path, safetensors_path)

    json_path = cache_dir / "{}.json".format(fingerprint)
    _atomic_write_bytes(json_path, json.dumps(skeleton).encode("utf-8"))


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
