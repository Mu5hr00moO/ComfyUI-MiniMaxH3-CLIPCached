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

A SIGINT/SIGTERM to the orchestrator process itself used to be handled by a
signal handler that fired one best-effort SIGTERM at the child and then
``os._exit(1)`` -- skipping the escalation and never reaping a child that
ignored that SIGTERM. ``install_shutdown_signal_handler()`` replaces that
with a handler that only raises ``OrchestratorShutdownSignal``; the
orchestrator's ``main()`` catches it and its ``finally:`` runs the same
``stop_live_server()`` escalation as an ordinary shutdown. The tests below
cover that the handler does nothing but raise, and that a signal-driven
teardown still escalates past an ignored SIGTERM to SIGKILL and reaps the
child.
"""

import signal
import subprocess
from pathlib import Path

import pytest

from scripts._live_server import (
    OrchestratorShutdownSignal,
    install_shutdown_signal_handler,
    stop_live_server,
)

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
    """Fail loudly if the helper reaches for a raw-PID API or a bare exit."""
    import os
    import psutil

    monkeypatch.setattr(
        os, "kill",
        lambda *a, **k: pytest.fail("teardown used os.kill() on a raw PID"))
    monkeypatch.setattr(
        psutil, "pid_exists",
        lambda *a, **k: pytest.fail("teardown used psutil.pid_exists() on a raw PID"))
    monkeypatch.setattr(
        os, "_exit",
        lambda *a, **k: pytest.fail("teardown / signal handler used os._exit()"))


@pytest.fixture
def _restore_signal_handlers():
    """Save and restore SIGINT/SIGTERM disposition around a test that installs
    the real orchestrator handler, so pytest's own handling is untouched."""
    saved = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        yield
    finally:
        for s, handler in saved.items():
            signal.signal(s, handler)


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


def test_shutdown_signal_handler_only_raises(_restore_signal_handlers):
    install_shutdown_signal_handler()
    for sig in (signal.SIGINT, signal.SIGTERM):
        handler = signal.getsignal(sig)
        assert callable(handler)
        with pytest.raises(OrchestratorShutdownSignal) as excinfo:
            handler(sig, None)
        # It carries the signal number and did nothing else: the autouse
        # fixture's os._exit / os.kill stubs would have failed the test.
        assert excinfo.value.signum == sig


def test_orchestrator_shutdown_signal_is_not_swallowed_by_except_exception():
    # Subclasses BaseException (like KeyboardInterrupt) so a mid-run
    # `try/except Exception` cannot eat a shutdown request.
    assert issubclass(OrchestratorShutdownSignal, BaseException)
    assert not issubclass(OrchestratorShutdownSignal, Exception)


def test_signal_driven_shutdown_still_escalates_and_reaps_a_stubborn_child(
        _restore_signal_handlers):
    """Regression for Codex LOW #3: a signal mid-run must go through the full
    stop_live_server() escalation, not a bare os._exit() that abandons a
    child which ignores the first SIGTERM."""
    install_shutdown_signal_handler()
    proc = FakePopen(dies_on=signal.SIGKILL)  # ignores SIGINT and SIGTERM

    shutdown_signum = None
    try:
        signal.raise_signal(signal.SIGTERM)  # handler runs, raises
        pytest.fail("handler did not raise OrchestratorShutdownSignal")
    except OrchestratorShutdownSignal as sig:
        shutdown_signum = sig.signum
    finally:
        rc = stop_live_server(proc, sigint_grace_s=0.01, sigterm_grace_s=0.01)

    assert shutdown_signum == signal.SIGTERM
    # Teardown ran from the finally: it escalated past the ignored
    # SIGINT/SIGTERM to SIGKILL and reaped the child.
    assert proc.received == [signal.SIGINT, signal.SIGTERM, signal.SIGKILL]
    assert rc == 0
    assert proc.poll() is not None  # child collected, not left an orphan


@pytest.mark.parametrize("script_name", SERVER_SCRIPTS)
def test_orchestrators_delegate_teardown_to_the_popen_safe_helper(script_name):
    src = (SCRIPTS_DIR / script_name).read_text()
    assert "from _live_server import" in src
    assert "stop_live_server(" in src
    # No bare-PID signalling or liveness checks left in the orchestrators.
    assert "os.kill(" not in src
    assert "psutil.pid_exists(" not in src
    assert "_server_pid_for_cleanup" not in src
    # The signal handler goes through the same teardown, not a parallel exit.
    assert "os._exit(" not in src
    assert "forward_termination" not in src
    assert "_server_proc_for_cleanup" not in src
    assert "install_shutdown_signal_handler()" in src
    assert "except OrchestratorShutdownSignal" in src


def test_live_server_helper_never_bare_exits_or_signals_a_raw_pid():
    """The shared helper must drive teardown through Popen only -- no
    os._exit(), no os.kill() (both appear in its docstring as prose; this
    walks the AST so only real call sites count)."""
    import ast

    tree = ast.parse((SCRIPTS_DIR / "_live_server.py").read_text())
    forbidden = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            dotted = ast.dump(node.func)
            if "'_exit'" in dotted or ("'kill'" in dotted and "'os'" in dotted):
                forbidden.add(ast.unparse(node.func))
    assert not forbidden, "raw-PID / bare-exit calls in _live_server.py: {}".format(forbidden)
