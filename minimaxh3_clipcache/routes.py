"""REST backend for the Cache Manager (plan section 16).

Five endpoints under /h3_cache_manager, registered on
PromptServer.instance.routes at import time -- the local convention
confirmed against MiniMaxH3-Prompt-Writer/backend/routes.py and
ComfyUI-MemoryVisualization/__init__.py (see CLAUDE.md "Faza 5 - lokalna
konwencja PromptServer routes"):

  GET  /h3_cache_manager/check      -> scan_cache(CACHE_DIR) verbatim
  GET  /h3_cache_manager/get        ?fingerprint=      -> that entry's verbose, 404 if none
  POST /h3_cache_manager/update     {fingerprint, name?/notes?/tags?/favorite?}
                                     -> partial update of the "user" block only
  POST /h3_cache_manager/delete     {fingerprint}      -> delete the whole cache entry
  GET  /h3_cache_manager/thumbnail  ?fingerprint=&index= -> image/jpeg bytes, 404 if none

None of these load a CLIP, encode anything, or load tensor payload data.
"check" is a filesystem scan that reads core JSON and safetensors headers;
the rest are small metadata / file operations. All heavy lifting stays in
the per-concern modules
(scanner / verbose_store / store / thumbnails) -- routes.py never becomes a
second store.

Handlers are plain module-level async functions so tests can call them
directly with a fake request; see tests/conftest.py for the `server` stub
that lets this module import without a running ComfyUI.
"""

import asyncio
import functools
import logging
import os
import re
from pathlib import Path

from aiohttp import web
from server import PromptServer

from minimaxh3_clipcache.last_used import get_last_used
from minimaxh3_clipcache.locking import get_lock
from minimaxh3_clipcache.scanner import scan_cache
from minimaxh3_clipcache.store import delete_conditioning
from minimaxh3_clipcache.thumbnails import THUMBNAILS_SUBDIR, delete_thumbnails
from minimaxh3_clipcache.verbose_store import (
    delete_verbose,
    load_verbose,
    update_user_metadata,
)

logger = logging.getLogger(__name__)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(REPO_ROOT, "cache")

ROUTE_PREFIX = "/h3_cache_manager"

# How long /update and /delete wait for the per-fingerprint writer lock
# before giving up with 409. A running generation on a MISS holds this lock
# for its ENTIRE load + encode + save (tens of seconds to a few minutes,
# not just the final save), so 5s is not "generous" -- it is deliberately
# short. A Delete or /update that lands mid-generation will usually just
# time out and get a 409, and that is the correct, safe outcome: the client
# retries once the encode has finished and released the lock. Raising this
# timeout to "wait it out" would only freeze the Cache Manager request for
# minutes; it would not make the operation any safer.
#
# The wait and the small operation protected by it both run in one executor
# worker. threading.Lock.acquire() must not freeze aiohttp's event loop, and
# keeping acquire -> operation -> release in the same synchronous callable
# also guarantees that request cancellation can never strand an acquired
# lock in a worker with no coroutine left to release it.
_LOCK_TIMEOUT_SECONDS = 5.0

_FINGERPRINT_RE = re.compile(r"^[0-9a-f]{64}$")
_LOCK_BUSY = object()


def _is_fingerprint(value) -> bool:
    return isinstance(value, str) and _FINGERPRINT_RE.match(value) is not None


def _error(message: str, status: int) -> web.Response:
    return web.json_response({"error": message}, status=status)


def _run_under_fingerprint_lock(fingerprint, operation):
    """Acquire, run and release entirely inside one executor worker.

    An aiohttp handler can be cancelled while it awaits ``run_in_executor``.
    If only ``lock.acquire`` ran in the worker, that worker could acquire the
    lock after cancellation with no coroutine left to release it. Keeping the
    critical section in the same synchronous callable makes release
    unconditional even when the awaiting request disappears.
    """
    lock = get_lock(fingerprint)
    if not lock.acquire(True, _LOCK_TIMEOUT_SECONDS):
        return _LOCK_BUSY
    try:
        return operation()
    finally:
        lock.release()


def _delete_entry_files(fingerprint, cache_dir):
    # The three steps here define what "a cache entry" is on disk. Each one
    # removes exactly the files its module's own path helper lists, and
    # scanner.entry_file_paths() composes those same three helpers for the
    # per-entry "size_bytes" the Cache Manager displays -- so the size shown
    # next to an entry is the size this function frees.
    delete_conditioning(fingerprint, cache_dir)
    delete_verbose(fingerprint, cache_dir)
    delete_thumbnails(fingerprint, cache_dir)


async def _json_object_body(request):
    """Parsed JSON body if it is an object, else None."""
    try:
        body = await request.json()
    except (ValueError, TypeError):
        return None
    return body if isinstance(body, dict) else None


routes = PromptServer.instance.routes


@routes.get(ROUTE_PREFIX + "/check")
async def check(request) -> web.Response:
    # scan_cache() walks the whole cache directory and reads every sidecar
    # from disk -- fast for a handful of entries, but O(entries) blocking I/O
    # that would freeze the aiohttp event loop (all of ComfyUI) for a large
    # cache. Push it onto a worker thread, same pattern as the lock waits in
    # /update and /delete.
    loop = asyncio.get_running_loop()
    data = await loop.run_in_executor(None, scan_cache, CACHE_DIR)
    # In-memory, filesystem-free: which fingerprint each node variant most
    # recently produced in this ComfyUI session, for the Cache Manager's
    # "active row" highlight. Kept out of scan_cache() on purpose -- it is not
    # a disk fact.
    data["last_used"] = get_last_used()
    return web.json_response(data)


@routes.get(ROUTE_PREFIX + "/get")
async def get(request) -> web.Response:
    fingerprint = request.query.get("fingerprint")
    if not _is_fingerprint(fingerprint):
        return _error("invalid or missing fingerprint", 400)
    verbose = load_verbose(fingerprint, CACHE_DIR)
    if verbose is None:
        return _error("no verbose metadata for fingerprint {}".format(fingerprint), 404)
    return web.json_response(verbose)


@routes.post(ROUTE_PREFIX + "/update")
async def update(request) -> web.Response:
    body = await _json_object_body(request)
    if body is None:
        return _error("request body must be a JSON object", 400)

    fingerprint = body.get("fingerprint")
    if not _is_fingerprint(fingerprint):
        return _error("invalid or missing fingerprint", 400)

    updates = {k: v for k, v in body.items() if k != "fingerprint"}
    if not updates:
        return _error("no user metadata fields to update", 400)

    # Serialise against the writer's per-fingerprint lock (proxy save /
    # nodes.py backfill): update_user_metadata does a read-modify-write on
    # the same verbose.json, so a racing system write could otherwise clobber
    # this edit. Non-blocking with a timeout -- a stuck writer must surface as
    # 409, never hang the request -- and the wait runs on an executor thread
    # so it never freezes the event loop (see _LOCK_TIMEOUT_SECONDS).
    loop = asyncio.get_running_loop()
    operation = functools.partial(update_user_metadata, fingerprint, updates, CACHE_DIR)
    try:
        updated = await loop.run_in_executor(
            None, _run_under_fingerprint_lock, fingerprint, operation,
        )
    except FileNotFoundError:
        return _error("no verbose metadata for fingerprint {}".format(fingerprint), 404)
    except ValueError as e:
        return _error(str(e), 400)
    if updated is _LOCK_BUSY:
        return _error(
            "this cache entry is currently being written by a running "
            "generation - try again in a moment", 409,
        )
    return web.json_response(updated)


@routes.post(ROUTE_PREFIX + "/delete")
async def delete(request) -> web.Response:
    body = await _json_object_body(request)
    if body is None:
        return _error("request body must be a JSON object", 400)

    fingerprint = body.get("fingerprint")
    if not _is_fingerprint(fingerprint):
        return _error("invalid or missing fingerprint", 400)

    # Serialise against the writer's per-fingerprint lock so a Delete cannot
    # land in the middle of an in-flight encode save. Non-blocking with a
    # timeout -- a stuck writer surfaces as 409 rather than hanging the request
    # -- and the wait runs on an executor thread so it never freezes the event
    # loop (see _LOCK_TIMEOUT_SECONDS).
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        None, _run_under_fingerprint_lock, fingerprint,
        functools.partial(_delete_entry_files, fingerprint, CACHE_DIR),
    )
    if result is _LOCK_BUSY:
        return _error(
            "this cache entry is currently being written by a running "
            "generation - try again in a moment", 409,
        )
    # Order per plan sections 15/20: core ".json" first (entry stops being a
    # HIT immediately), then ".safetensors", then the manager-only sidecar,
    # then thumbnails. Each step is idempotent, so a partial previous delete
    # is simply completed.
    return web.json_response({"deleted": fingerprint})


@routes.get(ROUTE_PREFIX + "/thumbnail")
async def thumbnail(request) -> web.Response:
    fingerprint = request.query.get("fingerprint")
    index = request.query.get("index")
    if not _is_fingerprint(fingerprint) or not (isinstance(index, str) and index.isdigit()):
        return _error("invalid or missing fingerprint/index", 400)

    path = Path(CACHE_DIR) / THUMBNAILS_SUBDIR / "{}_{}.jpg".format(fingerprint, int(index))
    if not path.is_file():
        return _error("thumbnail not found", 404)
    try:
        body = path.read_bytes()
    except FileNotFoundError:
        # Delete won the race between is_file() and read_bytes() above -
        # the same outcome as if is_file() had returned False a moment
        # later. Not worth locking for: a stale 404 here is harmless.
        return _error("thumbnail not found", 404)
    return web.Response(body=body, content_type="image/jpeg")
