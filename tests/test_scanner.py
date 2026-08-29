"""Unit tests for minimaxh3_clipcache.scanner.scan_cache."""

import json

import torch

from minimaxh3_clipcache.locking import get_lock
from minimaxh3_clipcache.scanner import scan_cache
from minimaxh3_clipcache.store import save_conditioning

FP1 = "1" * 64
FP2 = "2" * 64
FP3 = "3" * 64
FP4 = "4" * 64


def _write(path, nbytes):
    path.write_bytes(b"x" * nbytes)


def _core(tmp_path, fp):
    save_conditioning(fp, [torch.zeros(1)], tmp_path)


def _verbose(tmp_path, fp, obj=None, raw=None):
    path = tmp_path / "{}.verbose.json".format(fp)
    if raw is not None:
        path.write_bytes(raw)
    else:
        payload = obj if obj is not None else {"fingerprint": fp, "system": {}, "user": {}}
        path.write_bytes(json.dumps(payload).encode("utf-8"))


def _recursive_size(tmp_path):
    return sum(p.stat().st_size for p in tmp_path.rglob("*") if p.is_file())


def test_a_empty_cache_dir_returns_zeros(tmp_path):
    assert scan_cache(tmp_path) == {"entries": [], "total_count": 0, "total_size_bytes": 0}


def test_a_missing_cache_dir_returns_zeros(tmp_path):
    assert scan_cache(tmp_path / "does-not-exist") == {
        "entries": [], "total_count": 0, "total_size_bytes": 0}


def test_b_normal_entry_reports_verbose_content(tmp_path):
    _core(tmp_path, FP1)
    verbose_obj = {
        "fingerprint": FP1,
        "system": {"prompt": "a prompt", "references": []},
        "user": {"name": "keeper", "notes": "", "tags": ["night"], "favorite": True},
    }
    _verbose(tmp_path, FP1, obj=verbose_obj)

    result = scan_cache(tmp_path)

    assert result["total_count"] == 1
    (entry,) = result["entries"]
    assert entry["fingerprint"] == FP1
    assert entry["classification"] == "normal"
    assert entry["verbose"] == verbose_obj


def test_c_legacy_entry_has_no_verbose(tmp_path):
    _core(tmp_path, FP1)

    (entry,) = scan_cache(tmp_path)["entries"]

    assert entry["classification"] == "legacy"
    assert entry["verbose"] is None


def test_d_orphan_safetensors_is_swept_off_disk_by_scan(tmp_path):
    orphan = tmp_path / "{}.safetensors".format(FP1)
    _write(orphan, 500)

    result = scan_cache(tmp_path)

    assert not orphan.exists()  # gc_orphaned_cache_files() removed it
    assert result["entries"] == []
    assert result["total_count"] == 0
    assert result["total_size_bytes"] == 0  # its bytes are gone, not counted


def test_d_orphan_safetensors_sweep_leaves_a_real_entry_untouched(tmp_path):
    _core(tmp_path, FP1)
    _verbose(tmp_path, FP1)
    orphan = tmp_path / "{}.safetensors".format(FP2)
    _write(orphan, 500)

    result = scan_cache(tmp_path)

    assert not orphan.exists()
    (entry,) = result["entries"]
    assert entry["fingerprint"] == FP1
    assert entry["classification"] == "normal"
    assert (tmp_path / "{}.json".format(FP1)).exists()
    assert (tmp_path / "{}.safetensors".format(FP1)).exists()
    assert result["total_size_bytes"] == _recursive_size(tmp_path)


def test_e_orphan_verbose_not_an_entry_but_counted_in_size(tmp_path):
    _verbose(tmp_path, FP1)
    size = (tmp_path / "{}.verbose.json".format(FP1)).stat().st_size

    result = scan_cache(tmp_path)

    assert result["entries"] == []
    assert result["total_size_bytes"] == size


def test_e_lone_core_json_without_safetensors_is_not_an_entry(tmp_path):
    _write(tmp_path / "{}.json".format(FP1), 42)

    result = scan_cache(tmp_path)

    assert result["entries"] == []
    assert result["total_size_bytes"] == 42


def test_f_thumbnail_bytes_counted_in_size(tmp_path):
    _core(tmp_path, FP1)
    _verbose(tmp_path, FP1)
    thumbs = tmp_path / "thumbnails"
    thumbs.mkdir()
    _write(thumbs / "{}_0.jpg".format(FP1), 321)

    result = scan_cache(tmp_path)

    assert result["total_count"] == 1
    assert result["total_size_bytes"] == _recursive_size(tmp_path)
    assert result["total_size_bytes"] >= 321


def test_g_normal_with_corrupt_verbose_stays_normal_but_verbose_none(tmp_path):
    _core(tmp_path, FP1)
    _verbose(tmp_path, FP1, raw=b"\xff\xfe not json at all \x00")

    (entry,) = scan_cache(tmp_path)["entries"]

    assert entry["classification"] == "normal"  # the file exists
    assert entry["verbose"] is None             # ...but it does not parse


def test_h_mixed_directory_classifies_and_sizes_correctly(tmp_path):
    _core(tmp_path, FP1)
    _verbose(tmp_path, FP1)              # FP1 normal
    _core(tmp_path, FP2)                 # FP2 legacy
    _write(tmp_path / "{}.safetensors".format(FP3), 700)  # FP3 orphan tensor
    _verbose(tmp_path, FP4)              # FP4 orphan verbose
    thumbs = tmp_path / "thumbnails"
    thumbs.mkdir()
    _write(thumbs / "{}_0.jpg".format(FP1), 55)
    _write(tmp_path / "some-unrelated-file.txt", 9)

    result = scan_cache(tmp_path)

    by_fp = {e["fingerprint"]: e["classification"] for e in result["entries"]}
    assert by_fp == {FP1: "normal", FP2: "legacy"}
    assert result["total_count"] == 2
    assert result["total_size_bytes"] == _recursive_size(tmp_path)


def test_i_generation_mismatch_is_reported_as_inconsistent(tmp_path):
    _core(tmp_path, FP1)
    _verbose(tmp_path, FP1)
    json_path = tmp_path / "{}.json".format(FP1)
    payload = json.loads(json_path.read_bytes())
    payload["generation_id"] = "torn-refresh-generation"
    json_path.write_bytes(json.dumps(payload).encode("utf-8"))

    (entry,) = scan_cache(tmp_path)["entries"]

    assert entry["classification"] == "inconsistent"
    assert entry["reason"] == "generation_mismatch"
    assert entry["verbose"] is not None


def test_i_unreadable_core_json_is_reported_as_inconsistent(tmp_path):
    _core(tmp_path, FP1)
    (tmp_path / "{}.json".format(FP1)).write_bytes(b"not json")

    (entry,) = scan_cache(tmp_path)["entries"]

    assert entry["classification"] == "inconsistent"
    assert entry["reason"] == "json_unreadable"


def test_i_transient_mismatch_while_writer_holds_lock_is_not_reported(tmp_path):
    _core(tmp_path, FP1)
    _verbose(tmp_path, FP1)
    json_path = tmp_path / "{}.json".format(FP1)
    payload = json.loads(json_path.read_bytes())
    payload["generation_id"] = "temporarily-old-json"
    json_path.write_bytes(json.dumps(payload).encode("utf-8"))

    lock = get_lock(FP1)
    assert lock.acquire(blocking=False)
    try:
        (entry,) = scan_cache(tmp_path)["entries"]
        assert entry["classification"] == "normal"
    finally:
        lock.release()

    (entry,) = scan_cache(tmp_path)["entries"]
    assert entry["classification"] == "inconsistent"
