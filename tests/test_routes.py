"""Unit tests for minimaxh3_clipcache.routes (Phase 5 REST backend).

The `server` module is stubbed in conftest.py with a pass-through route
table, so importing minimaxh3_clipcache.routes leaves every handler as a
plain async module function. These tests call the handlers directly with a
fake request object and inspect the returned aiohttp Response -- no ComfyUI
server, no aiohttp routing layer.

What is NOT covered here (needs a live ComfyUI -- see CLAUDE.md "Faza 5"):
real @routes.get/@routes.post registration on PromptServer, aiohttp path
and query-string routing, and HTTP-level transfer of the thumbnail bytes.
"""

import asyncio
import json

import pytest

from minimaxh3_clipcache import routes as routes_module
from minimaxh3_clipcache.locking import get_lock
from minimaxh3_clipcache.routes import check, delete, get, thumbnail, update
from minimaxh3_clipcache.verbose_store import load_verbose, save_verbose

FP = "a" * 64
FP2 = "b" * 64
BAD_FP = "not-a-fingerprint"


class _Req:
    def __init__(self, query=None, json_body=None, json_raises=False):
        self.query = query or {}
        self._json_body = json_body
        self._json_raises = json_raises

    async def json(self):
        if self._json_raises:
            raise ValueError("body is not JSON")
        return self._json_body


def _run(coro):
    return asyncio.run(coro)


def _body(response):
    return json.loads(response.text)


@pytest.fixture(autouse=True)
def _cache_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(routes_module, "CACHE_DIR", tmp_path)
    return tmp_path


def _system(prompt="a prompt"):
    return {"prompt": prompt, "clip_name": "x.safetensors", "clip_file_size": 1,
            "clip_mtime_ns": 2, "cache_schema_version": 1, "references": []}


def _make_core_entry(cache_dir, fp):
    (cache_dir / "{}.json".format(fp)).write_bytes(b"{}")
    (cache_dir / "{}.safetensors".format(fp)).write_bytes(b"tensorbytes")


def _make_thumbnail(cache_dir, fp, index, data=b"\xff\xd8fakejpeg\xff\xd9"):
    thumbs = cache_dir / "thumbnails"
    thumbs.mkdir(exist_ok=True)
    (thumbs / "{}_{}.jpg".format(fp, index)).write_bytes(data)
    return data


# --- check ---

def test_check_empty_cache(_cache_dir):
    response = _run(check(_Req()))
    assert response.status == 200
    assert _body(response) == {"entries": [], "total_count": 0, "total_size_bytes": 0}


def test_check_lists_a_normal_entry(_cache_dir):
    _make_core_entry(_cache_dir, FP)
    save_verbose(FP, _system(), _cache_dir)

    response = _run(check(_Req()))
    payload = _body(response)
    assert payload["total_count"] == 1
    assert payload["entries"][0]["fingerprint"] == FP
    assert payload["entries"][0]["classification"] == "normal"


# --- get ---

def test_get_returns_verbose(_cache_dir):
    save_verbose(FP, _system(prompt="hello"), _cache_dir)
    response = _run(get(_Req(query={"fingerprint": FP})))
    assert response.status == 200
    assert _body(response)["system"]["prompt"] == "hello"


def test_get_missing_verbose_is_404(_cache_dir):
    response = _run(get(_Req(query={"fingerprint": FP})))
    assert response.status == 404


def test_get_bad_fingerprint_is_400(_cache_dir):
    assert _run(get(_Req(query={"fingerprint": BAD_FP}))).status == 400
    assert _run(get(_Req(query={}))).status == 400


# --- update ---

def test_update_partial_user_fields(_cache_dir):
    system = _system()
    save_verbose(FP, system, _cache_dir)

    response = _run(update(_Req(json_body={"fingerprint": FP, "name": "Keeper", "favorite": True})))
    assert response.status == 200

    on_disk = load_verbose(FP, _cache_dir)
    assert on_disk["user"] == {"name": "Keeper", "notes": "", "tags": [], "favorite": True}
    assert on_disk["system"] == system  # system untouched
    assert _body(response) == on_disk


def test_update_unknown_field_is_400(_cache_dir):
    save_verbose(FP, _system(), _cache_dir)
    response = _run(update(_Req(json_body={"fingerprint": FP, "prompt": "hijack"})))
    assert response.status == 400


def test_update_missing_sidecar_is_404(_cache_dir):
    response = _run(update(_Req(json_body={"fingerprint": FP, "name": "x"})))
    assert response.status == 404


def test_update_non_object_body_is_400(_cache_dir):
    assert _run(update(_Req(json_raises=True))).status == 400
    assert _run(update(_Req(json_body=["not", "an", "object"]))).status == 400


def test_update_no_fields_to_change_is_400(_cache_dir):
    save_verbose(FP, _system(), _cache_dir)
    assert _run(update(_Req(json_body={"fingerprint": FP}))).status == 400


def test_update_bad_fingerprint_is_400(_cache_dir):
    assert _run(update(_Req(json_body={"fingerprint": BAD_FP, "name": "x"}))).status == 400


# --- delete ---

def test_delete_removes_all_artifacts(_cache_dir):
    _make_core_entry(_cache_dir, FP)
    save_verbose(FP, _system(), _cache_dir)
    _make_thumbnail(_cache_dir, FP, 0)
    _make_thumbnail(_cache_dir, FP, 1)

    response = _run(delete(_Req(json_body={"fingerprint": FP})))
    assert response.status == 200
    assert _body(response) == {"deleted": FP}

    assert not (_cache_dir / "{}.json".format(FP)).exists()
    assert not (_cache_dir / "{}.safetensors".format(FP)).exists()
    assert not (_cache_dir / "{}.verbose.json".format(FP)).exists()
    assert list((_cache_dir / "thumbnails").glob("{}_*.jpg".format(FP))) == []


def test_delete_only_touches_the_named_fingerprint(_cache_dir):
    _make_core_entry(_cache_dir, FP)
    _make_core_entry(_cache_dir, FP2)
    save_verbose(FP2, _system(), _cache_dir)

    _run(delete(_Req(json_body={"fingerprint": FP})))

    assert (_cache_dir / "{}.json".format(FP2)).exists()
    assert (_cache_dir / "{}.verbose.json".format(FP2)).exists()


def test_delete_is_idempotent(_cache_dir):
    _make_core_entry(_cache_dir, FP)
    assert _run(delete(_Req(json_body={"fingerprint": FP}))).status == 200
    assert _run(delete(_Req(json_body={"fingerprint": FP}))).status == 200  # nothing left


def test_delete_bad_input(_cache_dir):
    assert _run(delete(_Req(json_body={"fingerprint": BAD_FP}))).status == 400
    assert _run(delete(_Req(json_raises=True))).status == 400


# --- delete / update vs. the writer's per-fingerprint lock ---
#
# get_lock() hands out a process-wide lock per fingerprint; acquiring it by
# hand here stands in for a running generation that is mid-save. Each test
# releases it in a finally so the shared lock never leaks into another test.

def test_delete_is_409_and_a_no_op_while_the_entry_is_being_written(_cache_dir, monkeypatch):
    monkeypatch.setattr(routes_module, "_LOCK_TIMEOUT_SECONDS", 0.05)
    _make_core_entry(_cache_dir, FP)
    save_verbose(FP, _system(), _cache_dir)
    _make_thumbnail(_cache_dir, FP, 0)

    lock = get_lock(FP)
    assert lock.acquire(timeout=1)
    try:
        response = _run(delete(_Req(json_body={"fingerprint": FP})))
        assert response.status == 409
        # every artefact is still on disk -- delete did not run at all
        assert (_cache_dir / "{}.json".format(FP)).exists()
        assert (_cache_dir / "{}.safetensors".format(FP)).exists()
        assert (_cache_dir / "{}.verbose.json".format(FP)).exists()
        assert (_cache_dir / "thumbnails" / "{}_0.jpg".format(FP)).exists()
    finally:
        lock.release()


def test_update_is_409_and_a_no_op_while_the_entry_is_being_written(_cache_dir, monkeypatch):
    monkeypatch.setattr(routes_module, "_LOCK_TIMEOUT_SECONDS", 0.05)
    save_verbose(FP, _system(), _cache_dir)
    user_before = load_verbose(FP, _cache_dir)["user"]

    lock = get_lock(FP)
    assert lock.acquire(timeout=1)
    try:
        response = _run(update(_Req(json_body={"fingerprint": FP, "name": "Keeper"})))
        assert response.status == 409
        assert load_verbose(FP, _cache_dir)["user"] == user_before
    finally:
        lock.release()


def test_delete_and_update_work_and_release_the_lock_when_it_is_free(_cache_dir):
    _make_core_entry(_cache_dir, FP)
    save_verbose(FP, _system(), _cache_dir)

    assert _run(update(_Req(json_body={"fingerprint": FP, "name": "x"}))).status == 200
    assert _run(delete(_Req(json_body={"fingerprint": FP}))).status == 200

    lock = get_lock(FP)
    assert lock.acquire(blocking=False), "a handler left the fingerprint lock held"
    lock.release()


# --- thumbnail ---

def test_thumbnail_serves_bytes(_cache_dir):
    data = _make_thumbnail(_cache_dir, FP, 0)
    response = _run(thumbnail(_Req(query={"fingerprint": FP, "index": "0"})))
    assert response.status == 200
    assert response.content_type == "image/jpeg"
    assert response.body == data


def test_thumbnail_missing_is_404(_cache_dir):
    response = _run(thumbnail(_Req(query={"fingerprint": FP, "index": "0"})))
    assert response.status == 404


def test_thumbnail_bad_input_is_400(_cache_dir):
    assert _run(thumbnail(_Req(query={"fingerprint": BAD_FP, "index": "0"}))).status == 400
    assert _run(thumbnail(_Req(query={"fingerprint": FP, "index": "notint"}))).status == 400
    assert _run(thumbnail(_Req(query={"fingerprint": FP}))).status == 400
