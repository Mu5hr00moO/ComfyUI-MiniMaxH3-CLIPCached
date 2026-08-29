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

None of these load a CLIP, encode anything, or open a ".safetensors" file.
"check" is a pure filesystem scan; the rest are small metadata / file
operations. All heavy lifting stays in the per-concern modules
(scanner / verbose_store / store / thumbnails) -- routes.py never becomes a
second store.

Handlers are plain module-level async functions so tests can call them
directly with a fake request; see tests/conftest.py for the `server` stub
that lets this module import without a running ComfyUI.
"""

import asyncio
import logging
import os
import re
from pathlib import Path

from aiohttp import web
from server import PromptServer

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
# before giving up with 409. A running generation only holds it for the
# duration of one verbose/core save (well under a second), so 5s is
# generous; the point is to never block the request indefinitely behind a
# stuck writer.
#
# The wait itself is pushed onto an executor thread (run_in_executor):
# threading.Lock.acquire() is a blocking call, and running it directly in
# an `async def` would freeze the whole aiohttp event loop (i.e. all of
# ComfyUI, not just this request) for the duration of the wait, timeout or
# not. run_in_executor moves only the waiting onto a worker thread, leaving
# the event loop free to serve other requests.
_LOCK_TIMEOUT_SECONDS = 5.0

_FINGERPRINT_RE = re.compile(r"^[0-9a-f]{64}$")


def _is_fingerprint(value) -> bool:
    return isinstance(value, str) and _FINGERPRINT_RE.match(value) is not None


def _error(message: str, status: int) -> web.Response:
    return web.json_response({"error": message}, status=status)


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
    return web.json_response(scan_cache(CACHE_DIR))


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
    lock = get_lock(fingerprint)
    loop = asyncio.get_running_loop()
    acquired = await loop.run_in_executor(None, lock.acquire, True, _LOCK_TIMEOUT_SECONDS)
    if not acquired:
        return _error(
            "this cache entry is currently being written by a running "
            "generation - try again in a moment", 409,
        )
    try:
        updated = update_user_metadata(fingerprint, updates, CACHE_DIR)
    except FileNotFoundError:
        return _error("no verbose metadata for fingerprint {}".format(fingerprint), 404)
    except ValueError as e:
        return _error(str(e), 400)
    finally:
        lock.release()
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
    lock = get_lock(fingerprint)
    loop = asyncio.get_running_loop()
    acquired = await loop.run_in_executor(None, lock.acquire, True, _LOCK_TIMEOUT_SECONDS)
    if not acquired:
        return _error(
            "this cache entry is currently being written by a running "
            "generation - try again in a moment", 409,
        )
    # Order per plan sections 15/20: core ".json" first (entry stops being a
    # HIT immediately), then ".safetensors", then the manager-only sidecar,
    # then thumbnails. Each step is idempotent, so a partial previous delete
    # is simply completed.
    try:
        delete_conditioning(fingerprint, CACHE_DIR)
        delete_verbose(fingerprint, CACHE_DIR)
        delete_thumbnails(fingerprint, CACHE_DIR)
    finally:
        lock.release()
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
