"""Shared teardown for the live-server dev scripts (not a test itself).

Every one of the `python main.py` orchestrator scripts in this directory
launches ComfyUI as a `subprocess.Popen` child and later has to stop it.
Doing that through a bare `os.kill(server_pid, ...)` on the remembered
integer plus `psutil.pid_exists(server_pid)` has a narrow but real bug: once
the child exits, the OS is free to hand its PID to an unrelated new process,
so a late SIGTERM/SIGKILL escalation can hit the wrong process. A
`psutil.pid_exists()` wait loop also never reaps the zombie, so it spins
until its own deadline on every clean shutdown.

`subprocess.Popen` already solves both problems: `send_signal()` polls first
and refuses to signal a process whose exit status has been collected
(CPython bpo-38630 / bpo-40550), and `wait(timeout=...)` reaps the child.
`stop_live_server()` routes the whole SIGINT -> SIGTERM -> SIGKILL
escalation through the Popen object so the call sites never touch a raw PID,
and `install_shutdown_signal_handler()` makes a SIGINT/SIGTERM delivered to
the orchestrator itself unwind into that same escalation (via
`OrchestratorShutdownSignal`) rather than a bare `os._exit()` that would
abandon a slow or stubborn child.

Residual, accepted limitation -- deliberately NOT closed with pidfd.
CPython's `Popen.send_signal()` still ends in a plain `os.kill(self.pid,
sig)` *after* its `poll()` guard (see the final paragraph of the comment in
CPython's own subprocess.py: "The race condition can still happen if the
race condition described above happens between the returncode test and the
kill() call"). If the child exits AND the OS recycles its exact PID inside
the sub-millisecond gap between that guard and the `os.kill()`, the signal
can land on the new holder of the PID. Every waitpid in these scripts goes
through a Popen method, all serialised on `Popen._waitpid_lock` and all
re-checking `returncode` under it, so the only actor that can even reach
this gap is a concurrent `poll()`/`wait()` racing one `send_signal()`; and
Linux allocates PIDs sequentially up to `pid_max` (~4M), so re-hitting the
same number inside a microsecond window is not a realistic event on a dev
box. Closing it anyway would mean bypassing `Popen.send_signal()` and
driving a `pidfd` we open and own -- and `os.pidfd_open` is not even exposed
in this project's interpreter (Python 3.14 / comfyenv). Reimplementing
around a proven stdlib mechanism for a developer-only script (own machine,
PID reuse, microsecond timing) is not worth it; treated as documented
residual risk, to be revisited only if the window is ever shown to be
practically reachable.
"""

import signal
import subprocess


class OrchestratorShutdownSignal(BaseException):
    """Raised in an orchestrator's main thread when SIGINT/SIGTERM arrives.

    Subclasses `BaseException`, not `Exception`, for the same reason
    `KeyboardInterrupt` does: a `try/except Exception` somewhere in the
    middle of the run must not be able to swallow a shutdown request. The
    orchestrator's `main()` catches it explicitly and lets its `finally:`
    run `stop_live_server()`, so the launched ComfyUI child gets the full
    SIGINT -> SIGTERM -> SIGKILL escalation and is reaped -- never left as an
    orphan by a bare `os._exit()` from inside the signal handler.
    """

    def __init__(self, signum):
        super().__init__(signum)
        self.signum = signum


def install_shutdown_signal_handler():
    """Route SIGINT and SIGTERM into `OrchestratorShutdownSignal`.

    The handler does the absolute minimum the CPython signal machinery
    allows: it raises, and nothing else. No printing, no I/O, no subprocess
    calls, no blocking wait -- all of that is the main thread's job, in the
    `finally:` that runs `stop_live_server()`.

    A repeat signal (an impatient second Ctrl-C) that arrives while teardown
    is already running unwinds `stop_live_server()` and propagates out of
    `main()`; by then the child has already had at least one SIGINT, so that
    is an acceptable "I really mean it" exit, not a fresh orphan risk.

    Must be called from the main thread (like `signal.signal` itself), which
    is where every orchestrator invokes it, at the top of `main()`.
    """

    def _raise_shutdown(signum, frame):
        raise OrchestratorShutdownSignal(signum)

    signal.signal(signal.SIGINT, _raise_shutdown)
    signal.signal(signal.SIGTERM, _raise_shutdown)


def stop_live_server(proc, *, skip_sigint=False, sigint_grace_s=45.0,
                     sigterm_grace_s=5.0):
    """Stop a `python main.py` subprocess, escalating SIGINT -> SIGTERM -> SIGKILL.

    Args:
        proc: the `subprocess.Popen` returned when the server was launched.
        skip_sigint: skip the initial clean SIGINT (used when a watchdog has
            already signalled the server) and go straight to waiting, then
            SIGTERM, then SIGKILL.
        sigint_grace_s: seconds to wait after SIGINT before escalating.
        sigterm_grace_s: seconds to wait after SIGTERM before escalating, and
            after SIGKILL before giving up.

    Every signal and liveness check goes through `proc`, so a PID the OS may
    have recycled after the child exited is never signalled.

    Returns the child's exit code, or None if it somehow outlived SIGKILL.
    """
    if proc.poll() is not None:
        print("=== Server PID {} already exited (rc={}) ===".format(
            proc.pid, proc.returncode), flush=True)
        return proc.returncode

    if not skip_sigint:
        print("=== Stopping server cleanly (SIGINT), PID={} ===".format(proc.pid),
              flush=True)
        try:
            proc.send_signal(signal.SIGINT)
        except Exception:
            pass

    for grace_s, escalate, label in (
        (sigint_grace_s, proc.terminate, "SIGTERM"),
        (sigterm_grace_s, proc.kill, "SIGKILL"),
    ):
        try:
            proc.wait(timeout=grace_s)
            break
        except subprocess.TimeoutExpired:
            print("!!! server PID {} still alive after {:.0f}s -- escalating to "
                  "{} !!!".format(proc.pid, grace_s, label), flush=True)
            try:
                escalate()
            except Exception:
                pass
    else:
        # SIGKILL has been sent; give the kernel a moment to reap it.
        try:
            proc.wait(timeout=sigterm_grace_s)
        except subprocess.TimeoutExpired:
            pass

    exited = proc.poll() is not None
    print("=== Server PID {} exited: {} (rc={}) ===".format(
        proc.pid, exited, proc.returncode), flush=True)
    return proc.returncode if exited else None
