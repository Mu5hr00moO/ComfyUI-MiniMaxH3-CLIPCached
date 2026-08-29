"""Unit tests for minimaxh3_clipcache.last_used.

Pure in-memory state: no ComfyUI, no GPU, no disk. The module-level dict
is process-wide, so an autouse fixture resets it before and after every
test the same way test_encoder_abi.py does for its process-wide cache.
"""

import pytest

from minimaxh3_clipcache import last_used
from minimaxh3_clipcache.last_used import get_last_used, record_last_used

FP_A = "a" * 64
FP_B = "b" * 64
FP_C = "c" * 64


@pytest.fixture(autouse=True)
def _fresh_state():
    """last_used holds module-level state; reset before and after every test
    so tests never see each other's recorded fingerprints."""
    last_used._reset_for_tests()
    yield
    last_used._reset_for_tests()


def test_a_fresh_state_is_all_none():
    assert get_last_used() == {"fl2va": None, "ref2va": None}


def test_b_record_sets_only_that_variant():
    record_last_used("fl2va", FP_A)

    state = get_last_used()
    assert state["fl2va"] == FP_A
    assert state["ref2va"] is None


def test_c_second_record_overwrites_previous():
    record_last_used("fl2va", FP_A)
    record_last_used("fl2va", FP_B)

    assert get_last_used()["fl2va"] == FP_B


def test_d_variants_are_independent():
    record_last_used("fl2va", FP_A)
    record_last_used("ref2va", FP_B)

    assert get_last_used() == {"fl2va": FP_A, "ref2va": FP_B}

    # Overwriting one leaves the other untouched.
    record_last_used("fl2va", FP_C)
    assert get_last_used() == {"fl2va": FP_C, "ref2va": FP_B}


def test_e_get_last_used_returns_a_copy():
    record_last_used("fl2va", FP_A)

    returned = get_last_used()
    returned["fl2va"] = "mutated"
    returned["ref2va"] = "mutated"

    # The internal state is unaffected by mutating the returned dict.
    assert get_last_used() == {"fl2va": FP_A, "ref2va": None}
