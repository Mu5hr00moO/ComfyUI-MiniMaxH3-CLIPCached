"""Regression: live-server teardown must never signal a recycled PID.

The ``scripts/test_*server*.py`` orchestrators launch ``python main.py`` as a
``subprocess.Popen`` child and later stop it. Doing that through a bare
``os.kill(server_pid, ...)`` plus ``psutil.pid_exists(server_pid)`` on the
remembered integer has a narrow race: once the child exits, the OS is free to
hand its PID to an unrelated new process, and a late SIGTERM/SIGKILL
escalation then lands on that process. A ``psutil.pid_exists()`` wait loop
also never reaps the zombie, so it spins until its own deadline on every
clean shutdown.

``scripts/_live_server.py`` routes every signal and liveness check through the
``Popen`` object, which CPython makes PID-reuse-safe (bpo-38630 / bpo-40550).
These tests exercise that helper with a fake ``Popen`` and a monkeypatched
``psutil`` -- no real server and no real OS-level race.
"""

import signal
import subprocess
from pathlib import Path

import pytest

from scripts._live_server import forward_termination, stop_live_server

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
SERVER_SCRIPTS = [
    "test_ref2video_server_e2e.py",
    "test_ref2video_server_hit.py",
    "test_server_memory_trend_phase17.py",
    "test_ref2video_memory_trend.py",
]


class FakePopen:
    """Minimal ``subprocess.Popen`` stand-in that mimics the PID-reuse guard.

    ``dies_on`` is the first received signal that makes the fake child
    "exit". Once it has exited, ``send_signal()`` is a no-op -- exactly what
    CPython's ``Popen.send_signal()`` does after the exit status is
    collected, and that is the property that makes a recycled PID safe.
    """

    def __init__(self, pid=999_999, dies_on=signal.SIGINT, exit_code=0):
        self.pid = pid
        self.returncode = None
        self._dies_on = dies_on
        self._exit_code = exit_code
        self.received = []

    def poll(self):
        return self.returncode

    def send_signal(self, sig):
        self.received.append(sig)
        if self.returncode is not None:
            return  # already exited: never touch the (possibly recycled) PID
        if sig == self._dies_on:
            self.returncode = self._exit_code

    def terminate(self):
        self.send_signal(signal.SIGTERM)

    def kill(self):
        self.send_signal(signal.SIGKILL)

    def wait(self, timeout=None):
        if self.returncode is None:
            raise subprocess.TimeoutExpired(cmd="main.py", timeout=timeout)
        return self.returncode


@pytest.fixture(autouse=True)
def _forbid_raw_pid_apis(monkeypatch):
    """Fail loudly if the helper reaches for a raw-PID API instead of Popen."""
    import os
    import psutil

    monkeypatch.setattr(
        os, "kill",
        lambda *a, **k: pytest.fail("teardown used os.kill() on a raw PID"))
    monkeypatch.setattr(
        psutil, "pid_exists",
        lambda *a, **k: pytest.fail("teardown used psutil.pid_exists() on a raw PID"))


def test_graceful_sigint_never_escalates():
    proc = FakePopen(dies_on=signal.SIGINT)
    rc = stop_live_server(proc, sigint_grace_s=5, sigterm_grace_s=5)
    assert rc == 0
    # Core regression: once the child exits on SIGINT, no SIGTERM/SIGKILL is
    # sent -- either would land on whatever now holds the recycled PID.
    assert proc.received == [signal.SIGINT]


def test_escalates_when_sigint_and_sigterm_are_ignored():
    proc = FakePopen(dies_on=signal.SIGKILL)
    rc = stop_live_server(proc, sigint_grace_s=0.01, sigterm_grace_s=0.01)
    assert rc == 0
    assert proc.received == [signal.SIGINT, signal.SIGTERM, signal.SIGKILL]


def test_skip_sigint_goes_straight_to_escalation():
    proc = FakePopen(dies_on=signal.SIGTERM)
    rc = stop_live_server(proc, skip_sigint=True,
                          sigint_grace_s=0.01, sigterm_grace_s=0.01)
    assert rc == 0
    assert signal.SIGINT not in proc.received
    assert proc.received == [signal.SIGTERM]


def test_already_exited_child_is_not_signalled_at_all():
    proc = FakePopen()
    proc.returncode = 0  # exited before teardown ran; PID may be recycled now
    rc = stop_live_server(proc)
    assert rc == 0
    assert proc.received == []


def test_forward_termination_skips_an_exited_child():
    proc = FakePopen()
    proc.returncode = 0
    forward_termination(proc)
    assert proc.received == []


def test_forward_termination_signals_a_live_child():
    proc = FakePopen()
    forward_termination(proc)
    assert proc.received == [signal.SIGTERM]


def test_forward_termination_tolerates_a_never_launched_server():
    forward_termination(None)  # must not raise


@pytest.mark.parametrize("script_name", SERVER_SCRIPTS)
def test_orchestrators_delegate_teardown_to_the_popen_safe_helper(script_name):
    src = (SCRIPTS_DIR / script_name).read_text()
    assert "from _live_server import" in src
    assert "stop_live_server(" in src
    # No bare-PID signalling or liveness checks left in the orchestrators.
    assert "os.kill(" not in src
    assert "psutil.pid_exists(" not in src
    assert "_server_pid_for_cleanup" not in src
