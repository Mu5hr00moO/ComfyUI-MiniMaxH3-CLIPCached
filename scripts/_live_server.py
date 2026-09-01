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
These helpers route the whole escalation through the Popen object so the
call sites never touch a raw PID.
"""

import signal
import subprocess


def forward_termination(proc):
    """Best-effort SIGTERM to the launched server, safe from a signal handler.

    `proc` is the `subprocess.Popen` for `python main.py` (or None if the
    server was never launched). Safe to call more than once and after the
    child has already exited: `Popen.send_signal()` collects the exit status
    first and skips signalling a dead — possibly PID-recycled — process.
    """
    if proc is None:
        return
    try:
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
    except Exception:
        pass


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
