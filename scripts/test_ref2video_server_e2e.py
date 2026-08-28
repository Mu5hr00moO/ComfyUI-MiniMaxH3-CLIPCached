"""R7: end-to-end test of the new MiniMaxH3CLIPCachedRef2VA node through a
LIVE ComfyUI server (python main.py, so aimdo / DynamicVRAM is actually
active -- see CLAUDE.md phase 24), monitored by an EXTERNAL watchdog thread
in this orchestrator process.

Flow:
  1. Launch `python main.py` (same interpreter as this orchestrator, so the
     already-activated comfyenv carries over) as a subprocess, log to
     /tmp/r7_server.log, poll GET /system_stats until it answers, then
     cross-check the PID bound to port 8188.
  2. Verify BOTH cached nodes registered: grep the startup log for our
     module + any traceback, and GET /object_info/<node_id> for each.
  3. External watchdog thread: every 2s log server RSS + RAM %; if RAM > 70%
     SIGTERM the server and stop.
  4. Iteration 1 -- submit a minimal Ref2VA graph
     (EmptyImage -> MiniMaxH3CLIPCachedRef2VA -> PreviewAny) with one
     ref_image_0, cache_mode="auto". This is a guaranteed cache MISS: expect
     a real Qwen3-VL encode (~20s+, maybe more with the second VAE loaded)
     and a [CACHE MISS] line in the server log.
  5. Iteration 2 -- submit the byte-identical graph again. Expect [CACHE HIT]
     and a near-zero execution time (the ~27 GB encoder is never loaded).
  6. Stop the server cleanly: SIGINT, escalating SIGINT -> SIGTERM -> SIGKILL
     on a grace timer so a slow graceful shutdown never hangs the run.
  7. Print: startup-log fragment (both nodes), the [CACHE MISS] fragment +
     timing, the [CACHE HIT] fragment + timing, and final nvidia-smi.

Run with the comfyenv conda environment from anywhere, under a hard timeout:
    timeout 1200 conda run -n comfyenv --no-capture-output python -u \
        custom_nodes/ComfyUI-MiniMaxH3-CLIPCached/scripts/test_ref2video_server_e2e.py
"""

import json
import os
import re
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

SERVER_LOG_PATH = "/tmp/r7_server.log"
WATCHDOG_LOG_PATH = "/tmp/r7_server_watchdog.log"
SERVER_STARTUP_TIMEOUT_S = 300
SERVER_STARTUP_POLL_INTERVAL_S = 3

RAM_WATCHDOG_INTERVAL_S = 2
RAM_HARD_STOP_PCT = 70.0

PROMPT_POLL_INTERVAL_S = 3
PROMPT_MAX_WAIT_S = 600

CLIP_NAME = "qwen3vl_32b_minimax_h3_int8_convrot.safetensors"
VAE_NAME = "minimax_h3_video_vae_int8_convrot.safetensors"
AUDIO_VAE_NAME = "minimax_h3_audio_vae_fp32.safetensors"

PROMPT_TEXT = "a test prompt with <Picture 1>"
WIDTH = 1344
HEIGHT = 768
LENGTH = 124
REF_IMAGE_SIZE = "match"
# Small square reference image -- R4 established 256px keeps the VAE
# activation footprint clear of the encoder; it does not change the CLIP
# path. A live server with aimdo has more headroom than the isolated script,
# so this is conservative.
REF_IMAGE_HW = 256

OUR_NODE_IDS = ["MiniMaxH3CLIPCachedFL2VA", "MiniMaxH3CLIPCachedRef2VA"]

_server_pid_for_cleanup = None


def _forwarding_signal_handler(signum, frame):
    print("!!! Orchestrator received signal {} -- forwarding SIGTERM to server PID {} before exit !!!".format(
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


def nvidia_smi():
    try:
        return subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=30).stdout
    except Exception as e:
        return "(nvidia-smi failed: {})".format(e)


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
                    self.reason = "RAM {:.1f}% > {:.0f}% -- SIGTERM to server PID {}".format(
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


def build_workflow():
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
                "prompt": PROMPT_TEXT,
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


def submit_prompt():
    r = requests.post(BASE_URL + "/prompt", json={"prompt": build_workflow()}, timeout=30)
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
            return entry.get("status", {}).get("status_str"), entry
        time.sleep(PROMPT_POLL_INTERVAL_S)
    return "timeout", None


def check_nodes_registered():
    print("=== Startup log: our module + any traceback ===", flush=True)
    log_text = Path(SERVER_LOG_PATH).read_text(errors="replace")
    relevant = []
    lines = log_text.splitlines()
    for i, line in enumerate(lines):
        low = line.lower()
        if ("minimaxh3" in low and "clipcache" in low) or "minimaxh3clipcached" in low \
                or "clip-cached" in low or "traceback" in low \
                or ("error" in low and "minimax" in low):
            relevant.append("  L{}: {}".format(i + 1, line))
            if "traceback" in low:
                relevant.extend("  L{}: {}".format(i + 2 + j, lines[i + 1 + j])
                                for j in range(min(15, len(lines) - i - 1)))
    print("\n".join(relevant) if relevant else "  (no matching lines)", flush=True)

    print("=== /object_info for both cached nodes ===", flush=True)
    all_ok = True
    for node_id in OUR_NODE_IDS:
        try:
            r = requests.get(BASE_URL + "/object_info/{}".format(node_id), timeout=10)
            ok = r.status_code == 200 and node_id in r.json()
        except Exception as e:
            ok = False
            print("  {} -> request failed: {}".format(node_id, e), flush=True)
            continue
        print("  {} -> {}".format(node_id, "REGISTERED" if ok else "MISSING (HTTP {})".format(r.status_code)),
              flush=True)
        all_ok = all_ok and ok
    return all_ok


def server_log_cache_lines():
    """Return ([CACHE ...] lines, [Prompt executed ... lines]) from the server log."""
    text = Path(SERVER_LOG_PATH).read_text(errors="replace")
    cache = [ln for ln in text.splitlines() if "[CACHE " in ln]
    executed = [ln for ln in text.splitlines() if "Prompt executed in" in ln]
    return cache, executed


def stop_server(server_pid, skip_sigint=False):
    if not skip_sigint:
        print("=== Stopping server cleanly (SIGINT), PID={} ===".format(server_pid), flush=True)
        try:
            os.kill(server_pid, signal.SIGINT)
        except ProcessLookupError:
            pass
    deadline = time.time() + 45
    while time.time() < deadline and psutil.pid_exists(server_pid):
        time.sleep(1)
    if psutil.pid_exists(server_pid):
        print("!!! still alive after 45s -- SIGTERM !!!", flush=True)
        try:
            os.kill(server_pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        time.sleep(5)
    if psutil.pid_exists(server_pid):
        print("!!! still alive after SIGTERM -- SIGKILL !!!", flush=True)
        try:
            os.kill(server_pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        time.sleep(3)
    print("=== Server PID {} exited: {} ===".format(server_pid, not psutil.pid_exists(server_pid)), flush=True)


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
    print("=== Server subprocess PID={} ===".format(server_pid), flush=True)

    watchdog = None
    stopped_by_watchdog = False
    results = {}

    try:
        print("=== Waiting for server readiness ===", flush=True)
        if not wait_for_server_ready():
            raise RuntimeError("server not ready in {}s -- see {}".format(
                SERVER_STARTUP_TIMEOUT_S, SERVER_LOG_PATH))
        print("=== Server ready ===", flush=True)

        try:
            conns = [c for c in psutil.net_connections(kind="inet")
                     if c.laddr and c.laddr.port == PORT and c.status == psutil.CONN_LISTEN]
            bound_pids = {c.pid for c in conns if c.pid}
            if bound_pids and server_pid not in bound_pids:
                print("!!! port {} bound by {} not launched PID {} -- using bound !!!".format(
                    PORT, bound_pids, server_pid), flush=True)
                server_pid = next(iter(bound_pids))
                _server_pid_for_cleanup = server_pid
        except Exception as e:
            print("--- port->PID cross-check failed ({}), trusting {} ---".format(e, server_pid), flush=True)

        watchdog = Watchdog(server_pid)
        watchdog.start()
        print("=== Watchdog started (RAM hard-stop > {:.0f}%) ===".format(RAM_HARD_STOP_PCT), flush=True)

        nodes_ok = check_nodes_registered()
        results["nodes_registered"] = nodes_ok
        if not nodes_ok:
            raise RuntimeError("one or both cached nodes did NOT register -- see startup log above")

        for label, iteration in (("MISS", 1), ("HIT", 2)):
            if watchdog.triggered:
                stopped_by_watchdog = True
                break
            print("\n=== Iteration {} (expect {}) ===".format(iteration, label), flush=True)
            cache_before, _ = server_log_cache_lines()
            t0 = time.time()
            prompt_id = submit_prompt()
            print("  prompt_id={}".format(prompt_id), flush=True)
            status_str, entry = wait_for_prompt_completion(prompt_id)
            dt = time.time() - t0
            print("  status={} round-trip {:.1f}s".format(status_str, dt), flush=True)
            if status_str != "success":
                print(json.dumps(entry, indent=2)[:3000] if entry else "(no history entry)", flush=True)
                raise RuntimeError("iteration {} did not succeed (status={})".format(iteration, status_str))

            cache_after, executed = server_log_cache_lines()
            new_cache = cache_after[len(cache_before):]
            server_rss_kb = _read_proc_field_kb("/proc/{}/status".format(server_pid), "VmRSS")
            mem_available_kb = _read_proc_field_kb("/proc/meminfo", "MemAvailable")
            results[label] = {
                "round_trip_s": dt,
                "new_cache_lines": new_cache,
                "last_executed_line": executed[-1] if executed else None,
                "server_rss_kb": server_rss_kb,
                "mem_available_kb": mem_available_kb,
            }
            print("  new [CACHE ...] lines: {}".format(new_cache), flush=True)
            print("  server log: {}".format(executed[-1] if executed else "(no 'Prompt executed' line)"), flush=True)
            print("  server_RSS={}kB MemAvailable={}kB".format(server_rss_kb, mem_available_kb), flush=True)

    finally:
        if watchdog is not None:
            watchdog.stop()
        stop_server(server_pid, skip_sigint=stopped_by_watchdog)
        server_log_f.close()

    # --- report ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("=== R7 RESULT ===")
    print("=" * 70)

    startup_frag = []
    for ln in Path(SERVER_LOG_PATH).read_text(errors="replace").splitlines():
        low = ln.lower()
        if "clip-cached" in low or "minimaxh3clipcached" in low or "import times" in low \
                or ("minimax" in low and ("cached" in low or "clipcache" in low)):
            startup_frag.append(ln)
    print("\n--- startup log (node registration) ---")
    print("\n".join(startup_frag) if startup_frag else "(see /object_info result below instead)")
    print("  /object_info: " + ", ".join(
        "{}={}".format(nid, "OK" if results.get("nodes_registered") else "?") for nid in OUR_NODE_IDS))

    for label in ("MISS", "HIT"):
        d = results.get(label)
        print("\n--- iteration: expected {} ---".format(label))
        if not d:
            print("  (not reached)")
            continue
        print("  round-trip:        {:.1f}s".format(d["round_trip_s"]))
        print("  [CACHE ...] lines: {}".format(d["new_cache_lines"] or "(NONE)"))
        print("  server exec line:  {}".format(d["last_executed_line"]))
        print("  server RSS:        {} kB".format(d["server_rss_kb"]))
        print("  MemAvailable:      {} kB".format(d["mem_available_kb"]))

    print("\n--- nvidia-smi after server stop ---")
    print(nvidia_smi())

    # verdict
    miss = results.get("MISS", {})
    hit = results.get("HIT", {})
    miss_ok = any("[CACHE MISS]" in ln for ln in miss.get("new_cache_lines", []))
    hit_ok = any("[CACHE HIT]" in ln for ln in hit.get("new_cache_lines", []))
    ok = results.get("nodes_registered") and miss_ok and hit_ok
    if stopped_by_watchdog:
        ok = False
        print("\n!!! run aborted by RAM watchdog: {} !!!".format(watchdog.reason if watchdog else "?"))
    print("\n=== VERDICT: {} ===".format(
        "PASS (both nodes registered, MISS then HIT observed)" if ok else "FAIL / INCOMPLETE"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
