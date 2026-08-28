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

COMFYUI_ROOT = "/home/kamil/ComfyUI"

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

_server_pid_for_cleanup = None


def _forwarding_signal_handler(signum, frame):
    print("!!! Orchestrator received signal {} (likely external timeout/Ctrl-C) -- forwarding SIGTERM to "
          "server PID {} before exit, so it is never left orphaned !!!".format(
              signum, _server_pid_for_cleanup), flush=True)
    if _server_pid_for_cleanup is not None:
        try:
            os.kill(_server_pid_for_cleanup, signal.SIGTERM)
        except ProcessLookupError:
            pass
    os._exit(1)


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
    def __init__(self, server_pid):
        self.server_pid = server_pid
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
                    server_rss = psutil.Process(self.server_pid).memory_info().rss
                except psutil.NoSuchProcess:
                    server_rss = None

                f.write("{} ram_pct={:.1f} server_rss_bytes={}\n".format(
                    time.strftime("%Y-%m-%d %H:%M:%S"), ram_pct, server_rss))
                f.flush()

                if ram_pct > RAM_HARD_STOP_PCT:
                    self.reason = "RAM {:.1f}% > {:.0f}% -- sending SIGTERM to server PID {}".format(
                        ram_pct, RAM_HARD_STOP_PCT, self.server_pid)
                    print("!!! WATCHDOG: {} !!!".format(self.reason), flush=True)
                    try:
                        os.kill(self.server_pid, signal.SIGTERM)
                    except ProcessLookupError:
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
    global _server_pid_for_cleanup
    signal.signal(signal.SIGTERM, _forwarding_signal_handler)
    signal.signal(signal.SIGINT, _forwarding_signal_handler)

    Path(SERVER_LOG_PATH).write_text("")
    Path(WATCHDOG_LOG_PATH).write_text("")

    print("=== Launching python main.py (cwd={}) ===".format(COMFYUI_ROOT), flush=True)
    server_log_f = open(SERVER_LOG_PATH, "w")
    server_proc = subprocess.Popen(
        [sys.executable, "main.py"], cwd=COMFYUI_ROOT,
        stdout=server_log_f, stderr=subprocess.STDOUT,
    )
    server_pid = server_proc.pid
    _server_pid_for_cleanup = server_pid
    print("=== Server subprocess launched, PID={} ===".format(server_pid), flush=True)

    rows = []
    watchdog = None
    stopped_by_watchdog = False

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
            if bound_pids and server_pid not in bound_pids:
                print("!!! WARNING: port {} is bound by PID(s) {} but we launched PID {} -- "
                      "using the bound PID instead !!!".format(PORT, bound_pids, server_pid), flush=True)
                server_pid = next(iter(bound_pids))
                _server_pid_for_cleanup = server_pid
        except Exception as e:
            print("--- Could not cross-check port->PID binding ({}), trusting launched PID {} ---".format(
                e, server_pid), flush=True)

        watchdog = Watchdog(server_pid)
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

    finally:
        if watchdog is not None:
            watchdog.stop()

        if stopped_by_watchdog:
            print()
            print("=== Watchdog already sent SIGTERM -- skipping the clean-stop step ===", flush=True)
        else:
            print()
            print("=== Stopping server cleanly (SIGINT), PID={} ===".format(server_pid), flush=True)
            try:
                os.kill(server_pid, signal.SIGINT)
            except ProcessLookupError:
                pass

        deadline = time.time() + 60
        while time.time() < deadline and psutil.pid_exists(server_pid):
            time.sleep(1)
        if psutil.pid_exists(server_pid):
            print("!!! Server PID {} still alive after 60s -- escalating to SIGTERM !!!".format(server_pid),
                  flush=True)
            try:
                os.kill(server_pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            time.sleep(5)
        if psutil.pid_exists(server_pid):
            print("!!! Server PID {} still alive after SIGTERM -- escalating to SIGKILL !!!".format(server_pid),
                  flush=True)
            try:
                os.kill(server_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        else:
            print("=== Server PID {} confirmed exited ===".format(server_pid), flush=True)

        server_log_f.close()

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
