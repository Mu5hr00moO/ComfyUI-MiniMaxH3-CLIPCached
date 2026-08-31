"""Unit tests for nodes._pair_verbose_entries -- the helper that cross-links
the two Cache Manager verbose sidecars produced by a single dual-resolution
node run (same prompt, two resolutions, two fingerprints).

Pure filesystem round-trips: no GPU, no real encoder, no ComfyUI startup.
nodes.py is loaded via importlib with a private module name, the same way
test_node.py does it, so a bare ``import nodes`` never collides with
ComfyUI's own top-level nodes.py.
"""

import importlib.util
import logging
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_node_module():
    spec = importlib.util.spec_from_file_location(
        "minimaxh3clipcached_nodes_pairing_under_test", os.path.join(REPO_ROOT, "nodes.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _make_core_json(cache_dir, fingerprint):
    """Create the core <fingerprint>.json so _pair_verbose_entries()'s
    under-the-lock re-check ("has Delete already removed the core entry?")
    passes for that direction."""
    (cache_dir / "{}.json".format(fingerprint)).write_bytes(b"{}")


FP1 = "a" * 64
FP2 = "b" * 64


def test_cross_links_both_sidecars(monkeypatch, tmp_path):
    """Two distinct fingerprints from one dual-resolution run: each sidecar
    ends up carrying the other's fingerprint and pixel size."""
    node_module = _load_node_module()
    monkeypatch.setattr(node_module, "CACHE_DIR", tmp_path)
    from minimaxh3_clipcache.verbose_store import load_verbose, save_verbose

    _make_core_json(tmp_path, FP1)
    _make_core_json(tmp_path, FP2)
    save_verbose(FP1, {"prompt": "p", "references": []}, tmp_path)
    save_verbose(FP2, {"prompt": "p", "references": []}, tmp_path)

    node_module._pair_verbose_entries(FP1, 1344, 768, FP2, 1920, 1088)

    s1 = load_verbose(FP1, tmp_path)["system"]
    s2 = load_verbose(FP2, tmp_path)["system"]
    assert s1["paired_fingerprint"] == FP2
    assert (s1["paired_width"], s1["paired_height"]) == (1920, 1088)
    assert s2["paired_fingerprint"] == FP1
    assert (s2["paired_width"], s2["paired_height"]) == (1344, 768)


def test_noop_when_fingerprints_equal(monkeypatch, tmp_path):
    """fp1 == fp2 means the two resolutions collapsed onto one shared cache
    entry -- there is nothing to pair, so no paired_fingerprint is written."""
    node_module = _load_node_module()
    monkeypatch.setattr(node_module, "CACHE_DIR", tmp_path)
    from minimaxh3_clipcache.verbose_store import load_verbose, save_verbose

    _make_core_json(tmp_path, FP1)
    save_verbose(FP1, {"prompt": "p", "references": []}, tmp_path)

    node_module._pair_verbose_entries(FP1, 1344, 768, FP1, 1920, 1088)

    assert "paired_fingerprint" not in load_verbose(FP1, tmp_path)["system"]


def test_add_pairing_failure_is_swallowed(monkeypatch, tmp_path, caplog):
    """add_pairing() raising must not propagate out of _pair_verbose_entries():
    the verbose layer is not the source of truth and the dual node has
    already produced valid conditioning for both resolutions."""
    node_module = _load_node_module()
    monkeypatch.setattr(node_module, "CACHE_DIR", tmp_path)

    _make_core_json(tmp_path, FP1)
    _make_core_json(tmp_path, FP2)

    def boom(*a, **k):
        raise OSError("pairing write blew up")

    monkeypatch.setattr(node_module, "add_pairing", boom)

    with caplog.at_level(logging.WARNING):
        node_module._pair_verbose_entries(FP1, 1344, 768, FP2, 1920, 1088)  # must not raise

    assert any("VERBOSE PAIRING FAILED" in r.getMessage() for r in caplog.records)


def test_skips_direction_whose_core_entry_is_gone(monkeypatch, tmp_path):
    """A Cache Manager Delete can remove one entry's core <fp>.json between the
    encode and the pairing. That direction is skipped (same under-the-lock
    re-check as _sync_verbose_metadata) so no pairing pointer is written into
    a sidecar with no core cache behind it; the other direction still runs."""
    node_module = _load_node_module()
    monkeypatch.setattr(node_module, "CACHE_DIR", tmp_path)
    from minimaxh3_clipcache.verbose_store import load_verbose, save_verbose

    _make_core_json(tmp_path, FP1)          # FP1 core present
    # FP2 core deliberately absent -- Delete won the race
    save_verbose(FP1, {"prompt": "p", "references": []}, tmp_path)
    save_verbose(FP2, {"prompt": "p", "references": []}, tmp_path)

    node_module._pair_verbose_entries(FP1, 1344, 768, FP2, 1920, 1088)

    assert load_verbose(FP1, tmp_path)["system"]["paired_fingerprint"] == FP2
    assert "paired_fingerprint" not in load_verbose(FP2, tmp_path)["system"]
