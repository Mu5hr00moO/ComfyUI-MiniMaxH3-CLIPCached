"""Per-process record of which cache fingerprint each node variant
(fl2va / ref2va) most recently produced, HIT or MISS or refresh alike -
used only to highlight the "active" row in the Cache Manager UI.

Deliberately NOT persisted to disk, and this is by design rather than an
oversight. The value being tracked answers "which entry is the one this
running ComfyUI session is currently working with", not "which entry was
touched at some point in the past". Those are different questions: a
fingerprint written to disk and reloaded after a restart would highlight
a row that nothing in the fresh process has actually used, which is
misleading for a feature whose entire purpose is to point at the
currently active entry. So on every ComfyUI restart both variants
correctly start over at None, and the highlight only appears once the
node has genuinely run in this session.

Thread-safe: record_last_used() and get_last_used() both take a module
lock, so a request thread writing while the Cache Manager route reads
cannot observe a torn dict.
"""

import threading

_lock = threading.Lock()
_last_used = {"fl2va": None, "ref2va": None}


def record_last_used(node_variant: str, fingerprint: str) -> None:
    """Record that `node_variant` just produced `fingerprint`.

    No try/except: this is a plain assignment into a fixed dict under the
    lock, with no real failure mode.
    """
    with _lock:
        _last_used[node_variant] = fingerprint


def get_last_used() -> dict:
    """Return a copy of the {variant: fingerprint | None} mapping.

    A copy so the caller cannot mutate the module's internal state by
    holding onto the returned dict.
    """
    with _lock:
        return dict(_last_used)


def _reset_for_tests() -> None:
    """Test-only: restore the module-level state to its fresh value so each
    test starts from a known baseline."""
    global _last_used
    with _lock:
        _last_used = {"fl2va": None, "ref2va": None}
