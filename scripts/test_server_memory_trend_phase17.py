"""Phase 17 close-out: RAM/RSS trend across 10 real /prompt API requests
against a REAL ComfyUI server (python main.py), monitored by an EXTERNAL
watchdog -- a thread in this orchestrator process, not code running inside
the server.

Background: an external review (Grok) flagged that nodes.py only called
comfy.model_management.unload_model_and_clones() after the stock node's
execute(), without the explicit `del proxy` / `gc.collect()` /
soft_empty_cache() that CLAUDE.md phase 17 promised. nodes.py has now been
brought in line with the plan. The phase 18 end-to-end run only exercised
3 iterations -- not enough to rule out a slow accumulation. This script
extends the phase 24 step 3c harness (test_server_memory_trend.py) from
3 to 10 distinct prompts, each a guaranteed real cache MISS, and logs the
server RSS after every one so a slow upward trend would be visible.

Flow (identical to test_server_memory_trend.py except for the prompt count):
  1. Launch `python main.py` (same interpreter as this orchestrator, so the
     already-activated comfyenv environment carries over) as a subprocess,
     log to /tmp/phase17_server.log, poll GET /system_stats until it
     answers, cross-check the PID bound to port 8188.
  2. Start an external watchdog thread: every 2s, log server RSS and
     psutil.virtual_memory().percent to /tmp/phase17_server_watchdog.log.
     If RAM > 70%, SIGTERM the server PID, log why, stop submitting prompts.
  3. Submit 10 prompts to /prompt sequentially (never in parallel), each
     with a different prompt string (forces a real cache MISS every time --
     identical width/height/length/clip_name/cache_mode="auto" otherwise).
     Workflow: VAELoader -> our cached node -> PreviewAny. Poll
     /history/{prompt_id} after each submission until it's non-empty
     before submitting the next.
  4. After each completed prompt: log server VmRSS (/proc/<pid>/status) and
     MemAvailable (/proc/meminfo) -- 10 data points.
  5. After 10 prompts (or a watchdog SIGTERM), stop the server cleanly with
     SIGINT and wait; escalate SIGINT -> SIGTERM -> SIGKILL if it doesn't
     exit within a generous grace period. Skipped if the watchdog already
     sent SIGTERM.

Run with the comfyenv conda environment from anywhere, under a hard timeout
(10 * ~20s real MISS + startup + margin):
    timeout 1200 conda run -n comfyenv --no-capture-output python -u scripts/test_server_memory_trend_phase17.py
"""

import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil
import requests

from _live_server import (
    OrchestratorShutdownSignal,
    install_shutdown_signal_handler,
    stop_live_server,
)

# <ComfyUI>/custom_nodes/<this repo>/scripts/<this file> under a normal
# install, so the ComfyUI root (the cwd `python main.py` is launched from
# below) is four directories up from this file. Derived rather than
# hard-coded so the script also runs from a fork checked out elsewhere;
# COMFYUI_ROOT in the environment overrides it. Mirrors tests/conftest.py.
#
# os.path.abspath, not Path.resolve / os.path.realpath: a common install
# keeps this repo outside custom_nodes/ and symlinks it in, so the
# invocation path is <ComfyUI>/custom_nodes/<symlink>/scripts/<file>.
# Walking that non-resolved path keeps every step inside the symlinked
# layout and lands on the real ComfyUI root; resolving the symlink first
# would climb four levels from the repo's true location and overshoot.
# Do not "simplify" this back to Path.resolve().
_here = os.path.abspath(__file__)
COMFYUI_ROOT = os.environ.get(
    "COMFYUI_ROOT",
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_here)))),
)

HOST = "127.0.0.1"
PORT = 8188
BASE_URL = "http://{}:{}".format(HOST, PORT)

SERVER_LOG_PATH = "/tmp/phase17_server.log"
WATCHDOG_LOG_PATH = "/tmp/phase17_server_watchdog.log"
SERVER_STARTUP_TIMEOUT_S = 240
SERVER_STARTUP_POLL_INTERVAL_S = 3

RAM_WATCHDOG_INTERVAL_S = 2
RAM_HARD_STOP_PCT = 70.0  # conservative -- big safety margin

PROMPT_POLL_INTERVAL_S = 3
PROMPT_MAX_WAIT_S = 500

CLIP_NAME = "qwen3vl_32b_minimax_h3_int8_convrot.safetensors"
VAE_NAME = "minimax_h3_video_vae_int8_convrot.safetensors"
WIDTH = 1344
HEIGHT = 768
LENGTH = 124

PROMPTS = [
    "phase17 closeout server memory trend, iteration one, alpha",
    "phase17 closeout server memory trend, iteration two, bravo",
    "phase17 closeout server memory trend, iteration three, charlie",
    "phase17 closeout server memory trend, iteration four, delta",
    "phase17 closeout server memory trend, iteration five, echo",
    "phase17 closeout server memory trend, iteration six, foxtrot",
    "phase17 closeout server memory trend, iteration seven, golf",
    "phase17 closeout server memory trend, iteration eight, hotel",
    "phase17 closeout server memory trend, iteration nine, india",
    "phase17 closeout server memory trend, iteration ten, juliett",
]


def _read_proc_field_kb(path, field):
    try:
        with open(path) as f:
            for line in f:
                if line.startswith(field + ":"):
                    return int(line.split()[1])
    except Exception:
        pass
    return None


class Watchdog:
    def __init__(self, server_proc):
        self.server_proc = server_proc
        # Captured now, so a PID the OS later recycles fails the identity
        # check in psutil.Process.is_running() instead of being read/signalled.
        self._ps = psutil.Process(server_proc.pid)
        self.stop_event = threading.Event()
        self.triggered = False
        self.reason = None
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()

    def _loop(self):
        with open(WATCHDOG_LOG_PATH, "a") as f:
            while not self.stop_event.is_set():
                ram_pct = psutil.virtual_memory().percent
                try:
                    running = self.server_proc.poll() is None and self._ps.is_running()
                    server_rss = self._ps.memory_info().rss if running else None
                except psutil.Error:
                    running, server_rss = False, None

                f.write("{} ram_pct={:.1f} server_rss_bytes={}\n".format(
                    time.strftime("%Y-%m-%d %H:%M:%S"), ram_pct, server_rss))
                f.flush()

                if ram_pct > RAM_HARD_STOP_PCT and running:
                    self.reason = "RAM {:.1f}% > {:.0f}% -- sending SIGTERM to server PID {}".format(
                        ram_pct, RAM_HARD_STOP_PCT, self.server_proc.pid)
                    print("!!! WATCHDOG: {} !!!".format(self.reason), flush=True)
                    try:
                        self.server_proc.send_signal(signal.SIGTERM)
                    except Exception:
                        pass
                    self.triggered = True
                    self.stop_event.set()
                    return

                self.stop_event.wait(RAM_WATCHDOG_INTERVAL_S)


def wait_for_server_ready():
    deadline = time.time() + SERVER_STARTUP_TIMEOUT_S
    while time.time() < deadline:
        try:
            r = requests.get(BASE_URL + "/system_stats", timeout=5)
            if r.status_code == 200:
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(SERVER_STARTUP_POLL_INTERVAL_S)
    return False


def submit_prompt(prompt_text):
    workflow = {
        "1": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": VAE_NAME},
        },
        "2": {
            "class_type": "MiniMaxH3CLIPCachedFL2VA",
            "inputs": {
                "clip_name": CLIP_NAME,
                "vae": ["1", 0],
                "prompt": prompt_text,
                "width": WIDTH,
                "height": HEIGHT,
                "length": LENGTH,
                "cache_mode": "auto",
            },
        },
        "3": {
            "class_type": "PreviewAny",
            "inputs": {"source": ["2", 0]},
        },
    }
    r = requests.post(BASE_URL + "/prompt", json={"prompt": workflow}, timeout=30)
    r.raise_for_status()
    data = r.json()
    if "error" in data:
        raise RuntimeError("prompt submission rejected: {}".format(data))
    return data["prompt_id"]


def wait_for_prompt_completion(prompt_id):
    deadline = time.time() + PROMPT_MAX_WAIT_S
    while time.time() < deadline:
        r = requests.get(BASE_URL + "/history/{}".format(prompt_id), timeout=10)
        r.raise_for_status()
        data = r.json()
        if prompt_id in data:
            entry = data[prompt_id]
            status_str = entry.get("status", {}).get("status_str")
            return status_str, entry
        time.sleep(PROMPT_POLL_INTERVAL_S)
    return "timeout", None


def main():
    install_shutdown_signal_handler()

    Path(SERVER_LOG_PATH).write_text("")
    Path(WATCHDOG_LOG_PATH).write_text("")

    print("=== Launching python main.py (cwd={}) ===".format(COMFYUI_ROOT), flush=True)
    server_log_f = open(SERVER_LOG_PATH, "w")
    server_proc = subprocess.Popen(
        [sys.executable, "main.py"], cwd=COMFYUI_ROOT,
        stdout=server_log_f, stderr=subprocess.STDOUT,
    )
    server_pid = server_proc.pid
    print("=== Server subprocess launched, PID={} ===".format(server_pid), flush=True)

    rows = []
    watchdog = None
    stopped_by_watchdog = False
    shutdown_signum = None

    try:
        print("=== Waiting for server readiness (polling {}/system_stats) ===".format(BASE_URL), flush=True)
        if not wait_for_server_ready():
            raise RuntimeError("Server did not become ready within {}s -- see {}".format(
                SERVER_STARTUP_TIMEOUT_S, SERVER_LOG_PATH))
        print("=== Server ready ===", flush=True)

        try:
            conns = [c for c in psutil.net_connections(kind="inet")
                     if c.laddr and c.laddr.port == PORT and c.status == psutil.CONN_LISTEN]
            bound_pids = {c.pid for c in conns if c.pid}
        except Exception as e:
            raise RuntimeError(
                "Could not verify ownership of port {} for launched PID {}; "
                "refusing to run against or stop an unverified server".format(PORT, server_pid)
            ) from e
        if server_proc.poll() is not None or server_pid not in bound_pids:
            raise RuntimeError(
                "Port {} is owned by PID(s) {}, not the launched server PID {}. "
                "Another ComfyUI may already be running; refusing to adopt or stop it.".format(
                    PORT, sorted(bound_pids), server_pid)
            )

        watchdog = Watchdog(server_proc)
        watchdog.start()
        print("=== External watchdog started: RAM hard-stop > {:.0f}%, polling every {}s, "
              "monitoring server PID {} ===".format(RAM_HARD_STOP_PCT, RAM_WATCHDOG_INTERVAL_S, server_pid),
              flush=True)

        for i, prompt_text in enumerate(PROMPTS, start=1):
            if watchdog.triggered:
                print("=== Watchdog already triggered -- stopping before iteration {} ===".format(i), flush=True)
                stopped_by_watchdog = True
                break

            print()
            print("=== Iteration {}: submitting prompt {!r} ===".format(i, prompt_text), flush=True)
            t0 = time.time()
            prompt_id = submit_prompt(prompt_text)
            print("iteration {}: prompt_id={}".format(i, prompt_id), flush=True)

            status_str, entry = wait_for_prompt_completion(prompt_id)
            dt = time.time() - t0
            print("iteration {}: status={} finished in {:.1f}s".format(i, status_str, dt), flush=True)

            if status_str != "success":
                print("!!! iteration {}: NOT success (status={}) !!!".format(i, status_str), flush=True)
                if entry:
                    print(json.dumps(entry, indent=2)[:2000], flush=True)

            server_rss_kb = _read_proc_field_kb("/proc/{}/status".format(server_pid), "VmRSS")
            mem_available_kb = _read_proc_field_kb("/proc/meminfo", "MemAvailable")
            print("iteration {}: server_RSS={}kB MemAvailable={}kB".format(
                i, server_rss_kb, mem_available_kb), flush=True)

            rows.append({
                "iteration": i, "status": status_str, "duration_s": dt,
                "server_rss_kb": server_rss_kb, "mem_available_kb": mem_available_kb,
            })

            if watchdog.triggered:
                stopped_by_watchdog = True
                break

    except OrchestratorShutdownSignal as sig:
        shutdown_signum = sig.signum
        print("!!! Orchestrator received signal {} -- stopping the server through the "
              "normal teardown (SIGINT->SIGTERM->SIGKILL), not a bare exit !!!".format(
                  sig.signum), flush=True)
    finally:
        if watchdog is not None:
            watchdog.stop()

        print(flush=True)
        stop_live_server(server_proc, skip_sigint=stopped_by_watchdog,
                         sigint_grace_s=60)

        server_log_f.close()

    if shutdown_signum is not None:
        print("=== Run interrupted by signal {}; server stopped via normal teardown. ===".format(
            shutdown_signum), flush=True)
        sys.exit(128 + shutdown_signum)

    print()
    print("=== Summary table ===")
    header = "{:>10} {:>10} {:>12} {:>16} {:>18}".format(
        "iteration", "status", "duration_s", "server_RSS_kB", "MemAvailable_kB")
    print(header)
    print("-" * len(header))
    for r in rows:
        print("{:>10} {:>10} {:>12.1f} {:>16} {:>18}".format(
            r["iteration"], r["status"], r["duration_s"], r["server_rss_kb"], r["mem_available_kb"]))

    print()
    print("=== Watchdog triggered: {} ===".format(watchdog.triggered if watchdog else "N/A (never started)"))
    if watchdog and watchdog.triggered:
        print("=== Watchdog reason: {} ===".format(watchdog.reason))

    print()
    print("=== RESULT: data collected, see summary table above ===")


if __name__ == "__main__":
    main()
