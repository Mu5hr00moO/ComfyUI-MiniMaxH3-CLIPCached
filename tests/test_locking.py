"""minimaxh3_clipcache.locking hands out one lock per fingerprint, stable for
the life of the process, so proxy.py / nodes.py / routes.py all serialise
against the same object for a given cache entry."""

from minimaxh3_clipcache.locking import get_lock

FP = "a" * 64
FP2 = "b" * 64


def test_same_fingerprint_returns_the_same_lock():
    assert get_lock(FP) is get_lock(FP)


def test_distinct_fingerprints_return_distinct_locks():
    assert get_lock(FP) is not get_lock(FP2)


def test_returned_lock_is_usable_as_a_context_manager():
    lock = get_lock("c" * 64)
    with lock:
        assert lock.locked()
    assert not lock.locked()
