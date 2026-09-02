"""R9: short RAM/RSS trend check for the MiniMaxH3CLIPCachedRef2VA node across
5 real /prompt API requests against a LIVE ComfyUI server (python main.py, so
aimdo / DynamicVRAM is actually active -- see CLAUDE.md phase 24), monitored
by an EXTERNAL watchdog thread in this orchestrator process (not code running
inside the server).

This is the Ref2VA counterpart of test_server_memory_trend_phase17.py (which
did 10 iterations for the FL2VA node). Ref2VA runs the stock
MiniMaxH3ReferenceToVideo through the same CachedClipProxy and the same
targeted unload, but additionally holds a second VAE (audio_vae). The point
of this run is to confirm the memory trend stays flat here too -- i.e. the
aimdo mechanism behaves identically for both nodes because it is the same
proxy and the same unload path.

Flow (mirrors phase17, adapted for Ref2VA):
  1. Launch `python main.py` (same interpreter as this orchestrator, so the
     already-activated comfyenv carries over) as a subprocess, log to
     /tmp/r9_server.log, poll GET /system_stats until it answers, cross-check
     the PID bound to port 8188.
  2. Start an external watchdog thread: every 2s, log server RSS and
     psutil.virtual_memory().percent to /tmp/r9_server_watchdog.log. If RAM
     > 70%, SIGTERM the server PID, log why, stop submitting prompts. The
     watchdog firing is itself a meaningful result, not a test failure.
  3. Submit 5 prompts to /prompt sequentially (never in parallel), each with a
     DIFFERENT prompt string (forces a real cache MISS every time) and ONE
     ref_image_0 (a 512x512 EmptyImage -- a safe mid-range size, well clear of
     the >=1600px sizes used in the fingerprint-collision tests). Everything
     else -- width/height/length/clip_name/vae/audio_vae/ref_image_size/
     cache_mode="auto" -- is identical across the 5. Poll /history/{prompt_id}
     after each submission until it is non-empty before submitting the next.
  4. After each completed prompt: log server VmRSS (/proc/<pid>/status) and
     MemAvailable (/proc/meminfo) -- 5 data points.
  5. After 5 prompts (or a watchdog SIGTERM), stop the server: SIGINT,
     escalating SIGINT -> SIGTERM -> SIGKILL on a grace timer so a slow
     graceful shutdown never hangs the run. SIGINT is skipped if the watchdog
     already sent SIGTERM.
  6. Print the summary table (iteration | status | duration | server RSS |
     MemAvailable -- same format as phase 24 / phase 17), then a final
     `ps aux | grep` for stray processes and `nvidia-smi`.

Run with the comfyenv conda environment from anywhere, under a hard timeout
(server startup + 5 * Ref2VA MISS with a cold encoder load + margin):
    timeout 720 conda run -n comfyenv --no-capture-output python -u \
        custom_nodes/ComfyUI-MiniMaxH3-CLIPCached/scripts/test_ref2video_memory_trend.py
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
COMFYUI_ROOT = os.environ.get(
    "COMFYUI_ROOT", str(Path(__file__).resolve().parents[3]))

HOST = "127.0.0.1"
PORT = 8188
BASE_URL = "http://{}:{}".format(HOST, PORT)

SERVER_LOG_PATH = "/tmp/r9_server.log"
WATCHDOG_LOG_PATH = "/tmp/r9_server_watchdog.log"
SERVER_STARTUP_TIMEOUT_S = 300
SERVER_STARTUP_POLL_INTERVAL_S = 3

RAM_WATCHDOG_INTERVAL_S = 2
RAM_HARD_STOP_PCT = 70.0  # conservative -- big safety margin

PROMPT_POLL_INTERVAL_S = 3
PROMPT_MAX_WAIT_S = 500

CLIP_NAME = "qwen3vl_32b_minimax_h3_int8_convrot.safetensors"
VAE_NAME = "minimax_h3_video_vae_int8_convrot.safetensors"
AUDIO_VAE_NAME = "minimax_h3_audio_vae_fp32.safetensors"
WIDTH = 1344
HEIGHT = 768
LENGTH = 124
REF_IMAGE_SIZE = "match"
# 512x512: a safe mid-range reference size. The fingerprint-collision tests
# deliberately used >=1600px; this run is about the memory trend, not
# fingerprint divergence, so a modest square keeps the VAE activation
# footprint small and predictable.
REF_IMAGE_HW = 512

# 5 distinct prompt strings -> a guaranteed real cache MISS each time. Each
# keeps the <Picture 1> tag so the single ref_image_0 is referenced the way
# the stock node expects; only the surrounding text varies.
PROMPTS = [
    "r9 ref2va memory trend, iteration one, alpha, with <Picture 1>",
    "r9 ref2va memory trend, iteration two, bravo, with <Picture 1>",
    "r9 ref2va memory trend, iteration three, charlie, with <Picture 1>",
    "r9 ref2va memory trend, iteration four, delta, with <Picture 1>",
    "r9 ref2va memory trend, iteration five, echo, with <Picture 1>",
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


def nvidia_smi():
    try:
        return subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=30).stdout
    except Exception as e:
        return "(nvidia-smi failed: {})".format(e)


def ps_grep():
    try:
        out = subprocess.run(["ps", "aux"], capture_output=True, text=True, timeout=30).stdout
        keep = [ln for ln in out.splitlines()
                if ("main.py" in ln or "ComfyUI" in ln or "comfyenv" in ln)
                and "grep" not in ln and "ps aux" not in ln]
        return "\n".join(keep) if keep else "(no matching processes)"
    except Exception as e:
        return "(ps failed: {})".format(e)


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


def build_workflow(prompt_text):
    return {
        "1": {
            "class_type": "EmptyImage",
            "inputs": {"width": REF_IMAGE_HW, "height": REF_IMAGE_HW,
                       "batch_size": 1, "color": 0x7F7F7F},
        },
        "2": {
            "class_type": "MiniMaxH3CLIPCachedRef2VA",
            "inputs": {
                "clip_name": CLIP_NAME,
                "vae": ["4", 0],
                "audio_vae": ["5", 0],
                "prompt": prompt_text,
                "width": WIDTH,
                "height": HEIGHT,
                "length": LENGTH,
                "ref_image_size": REF_IMAGE_SIZE,
                "ref_image_0": ["1", 0],
                "cache_mode": "auto",
            },
        },
        "3": {
            "class_type": "PreviewAny",
            "inputs": {"source": ["2", 0]},
        },
        "4": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": VAE_NAME},
        },
        "5": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": AUDIO_VAE_NAME},
        },
    }


def submit_prompt(prompt_text):
    r = requests.post(BASE_URL + "/prompt", json={"prompt": build_workflow(prompt_text)}, timeout=30)
    r.raise_for_status()
    data = r.json()
    if "error" in data:
        raise RuntimeError("prompt submission rejected: {}".format(json.dumps(data, indent=2)))
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


def server_cache_lines():
    text = Path(SERVER_LOG_PATH).read_text(errors="replace")
    return [ln for ln in text.splitlines() if "[CACHE " in ln]


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
            cache_before = server_cache_lines()
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

            new_cache = server_cache_lines()[len(cache_before):]
            server_rss_kb = _read_proc_field_kb("/proc/{}/status".format(server_pid), "VmRSS")
            mem_available_kb = _read_proc_field_kb("/proc/meminfo", "MemAvailable")
            print("iteration {}: new [CACHE ...] lines: {}".format(i, new_cache or "(NONE)"), flush=True)
            print("iteration {}: server_RSS={}kB MemAvailable={}kB".format(
                i, server_rss_kb, mem_available_kb), flush=True)

            rows.append({
                "iteration": i, "status": status_str, "duration_s": dt,
                "server_rss_kb": server_rss_kb, "mem_available_kb": mem_available_kb,
                "cache_lines": new_cache,
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
            r["iteration"], r["status"], r["duration_s"],
            "n/a" if r["server_rss_kb"] is None else r["server_rss_kb"],
            "n/a" if r["mem_available_kb"] is None else r["mem_available_kb"]))

    print()
    print("=== [CACHE ...] line per iteration (each should be a MISS) ===")
    for r in rows:
        print("  iteration {}: {}".format(r["iteration"], r["cache_lines"] or "(NONE)"))

    if rows:
        first_rss = rows[0]["server_rss_kb"] or 0
        last_rss = rows[-1]["server_rss_kb"] or 0
        print()
        print("=== RSS delta first -> last iteration: {:+d} kB ({:+.1f} MB) over {} iterations ===".format(
            last_rss - first_rss, (last_rss - first_rss) / 1024.0, len(rows)))

    print()
    print("=== Watchdog triggered: {} ===".format(watchdog.triggered if watchdog else "N/A (never started)"))
    if watchdog and watchdog.triggered:
        print("=== Watchdog reason: {} ===".format(watchdog.reason))

    print()
    print("=== ps aux (main.py / ComfyUI / comfyenv) ===")
    print(ps_grep())
    print()
    print("=== nvidia-smi ===")
    print(nvidia_smi())

    print()
    print("=== RESULT: data collected, see summary table above ===")


if __name__ == "__main__":
    main()
