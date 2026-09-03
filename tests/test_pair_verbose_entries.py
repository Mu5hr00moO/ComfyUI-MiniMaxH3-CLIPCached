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
    ends up carrying the other's fingerprint and pixel size. The a side
    (fp_a / width_a / height_a) is the base-resolution entry and the b side
    the upscale target, so is_upscale_target lands False on fp_a and True
    on fp_b."""
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
    assert s1["is_upscale_target"] is False
    assert s2["paired_fingerprint"] == FP1
    assert (s2["paired_width"], s2["paired_height"]) == (1344, 768)
    assert s2["is_upscale_target"] is True


def test_b_is_upscale_target_false_swaps_the_role_flags(monkeypatch, tmp_path):
    """b_is_upscale_target is the caller's declaration of which side is the
    upscale target; passing False makes fp_a the upscale target instead."""
    node_module = _load_node_module()
    monkeypatch.setattr(node_module, "CACHE_DIR", tmp_path)
    from minimaxh3_clipcache.verbose_store import load_verbose, save_verbose

    _make_core_json(tmp_path, FP1)
    _make_core_json(tmp_path, FP2)
    save_verbose(FP1, {"prompt": "p", "references": []}, tmp_path)
    save_verbose(FP2, {"prompt": "p", "references": []}, tmp_path)

    node_module._pair_verbose_entries(FP1, 1920, 1088, FP2, 1344, 768,
                                      b_is_upscale_target=False)

    assert load_verbose(FP1, tmp_path)["system"]["is_upscale_target"] is True
    assert load_verbose(FP2, tmp_path)["system"]["is_upscale_target"] is False


def test_noop_when_fingerprints_equal_finalizes_to_base_resolution(monkeypatch, tmp_path):
    """fp1 == fp2 means the two resolutions collapsed onto one shared cache
    entry -- there is nothing to pair, so no paired_fingerprint is written.

    The one thing this branch does do is finalize the informational
    generation-size trio to the BASE side (width_a / height_a): the upscale
    pass was a cache HIT that _sync_verbose_metadata() may have moved the
    trio forward to the upscale resolution, and this run's canonical size is
    the base one. Everything else in "system" is preserved."""
    node_module = _load_node_module()
    monkeypatch.setattr(node_module, "CACHE_DIR", tmp_path)
    from minimaxh3_clipcache.verbose_store import load_verbose, save_verbose

    _make_core_json(tmp_path, FP1)
    # Sidecar left at the upscale size, as a HIT-refresh would have done.
    save_verbose(FP1, {"prompt": "p", "references": [],
                       "width": 1920, "height": 1088, "megapixels": 2.09}, tmp_path)

    node_module._pair_verbose_entries(FP1, 1344, 768, FP1, 1920, 1088)

    system = load_verbose(FP1, tmp_path)["system"]
    assert "paired_fingerprint" not in system
    assert (system["width"], system["height"]) == (1344, 768)
    assert system["megapixels"] == 1.03
    assert system["prompt"] == "p"


def test_shared_fingerprint_finalize_skipped_when_core_entry_gone(monkeypatch, tmp_path):
    """A Cache Manager Delete can remove the shared entry's core <fp>.json
    between the encode and the pairing call. On the fp1 == fp2 branch the
    under-the-lock re-check then skips the finalize, so no verbose sidecar is
    resurrected behind a deleted core entry."""
    node_module = _load_node_module()
    monkeypatch.setattr(node_module, "CACHE_DIR", tmp_path)
    from minimaxh3_clipcache.verbose_store import load_verbose, save_verbose

    # Core <fp>.json deliberately absent -- Delete won the race.
    save_verbose(FP1, {"prompt": "p", "references": [],
                       "width": 1920, "height": 1088, "megapixels": 2.09}, tmp_path)

    node_module._pair_verbose_entries(FP1, 1344, 768, FP1, 1920, 1088)  # must not raise

    system = load_verbose(FP1, tmp_path)["system"]
    assert (system["width"], system["height"]) == (1920, 1088)  # untouched


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

    s1 = load_verbose(FP1, tmp_path)["system"]
    assert s1["paired_fingerprint"] == FP2
    assert s1["is_upscale_target"] is False
    assert "paired_fingerprint" not in load_verbose(FP2, tmp_path)["system"]
