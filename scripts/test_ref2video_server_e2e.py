"""R7: end-to-end test of the new MiniMaxH3CLIPCachedRef2VA node through a
LIVE ComfyUI server (python main.py, so aimdo / DynamicVRAM is actually
active -- see CLAUDE.md phase 24), monitored by an EXTERNAL watchdog thread
in this orchestrator process.

Scope: this script proves only (a) both cached nodes register in a real
server and (b) a real cache MISS runs end to end -- a genuine [CACHE MISS]
line, a real Qwen3-VL encode, and a written cache entry. It deliberately
does NOT try to prove the proxy HIT path. Resubmitting a byte-identical
graph in the SAME server session is short-circuited by ComfyUI's own
execution cache ("Prompt executed in 0.00 seconds") before our node ever
re-runs, so a "[CACHE HIT]" seen that way would say nothing about our
CachedClipProxy. The real HIT proof is
scripts/test_ref2video_server_hit.py: it starts a FRESH server (empty
execution cache) and submits the graph this MISS run wrote an entry for,
reading back the fingerprint recorded here via /tmp/r7_last_fingerprint.txt.

Flow:
  1. Launch `python main.py` (same interpreter as this orchestrator, so the
     already-activated comfyenv carries over) as a subprocess, log to
     /tmp/r7_server.log, poll GET /system_stats until it answers, then
     cross-check the PID bound to port 8188.
  2. Verify BOTH cached nodes registered: grep the startup log for our
     module + any traceback, and GET /object_info/<node_id> for each.
  3. External watchdog thread: every 2s log server RSS + RAM %; if RAM > 70%
     SIGTERM the server and stop.
  4. Submit a minimal Ref2VA graph
     (EmptyImage -> MiniMaxH3CLIPCachedRef2VA -> PreviewAny) with one
     ref_image_0, cache_mode="auto". This is a guaranteed cache MISS: expect
     a real Qwen3-VL encode (~20s+, maybe more with the second VAE loaded)
     and a [CACHE MISS] line in the server log. Record the observed
     fingerprint to /tmp/r7_last_fingerprint.txt for the HIT-path follow-up.
  5. Stop the server cleanly: SIGINT, escalating SIGINT -> SIGTERM -> SIGKILL
     on a grace timer so a slow graceful shutdown never hangs the run.
  6. Print: startup-log fragment (both nodes), the [CACHE MISS] fragment +
     timing, and final nvidia-smi.

Run with the comfyenv conda environment from anywhere, under a hard timeout:
    timeout 1200 conda run -n comfyenv --no-capture-output python -u \
        custom_nodes/ComfyUI-MiniMaxH3-CLIPCached/scripts/test_ref2video_server_e2e.py
"""

import json
import re
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

COMFYUI_ROOT = "/home/kamil/ComfyUI"

HOST = "127.0.0.1"
PORT = 8188
BASE_URL = "http://{}:{}".format(HOST, PORT)

SERVER_LOG_PATH = "/tmp/r7_server.log"
WATCHDOG_LOG_PATH = "/tmp/r7_server_watchdog.log"
# The HIT-path follow-up (test_ref2video_server_hit.py) reads the fingerprint
# this run actually observed from here, instead of carrying a hardcoded value
# that goes stale every time a fingerprint input changes (encoder ABI, cache
# schema version, stat fields, hash framing, ...).
FINGERPRINT_HANDOFF_PATH = Path("/tmp/r7_last_fingerprint.txt")
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
                    self.reason = "RAM {:.1f}% > {:.0f}% -- SIGTERM to server PID {}".format(
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
            all_ok = False  # the `continue` below skips the `all_ok and ok` fold
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
    print("=== Server subprocess PID={} ===".format(server_pid), flush=True)

    watchdog = None
    stopped_by_watchdog = False
    shutdown_signum = None
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
        print("=== Watchdog started (RAM hard-stop > {:.0f}%) ===".format(RAM_HARD_STOP_PCT), flush=True)

        nodes_ok = check_nodes_registered()
        results["nodes_registered"] = nodes_ok
        if not nodes_ok:
            raise RuntimeError("one or both cached nodes did NOT register -- see startup log above")

        # A SINGLE submission -- the guaranteed cache MISS. A second identical
        # submission in this same session would be intercepted by ComfyUI's own
        # execution cache before our node re-runs, so it cannot demonstrate a
        # proxy HIT; that is what scripts/test_ref2video_server_hit.py does,
        # from a fresh server, using the fingerprint recorded below.
        if watchdog.triggered:
            stopped_by_watchdog = True
        else:
            print("\n=== Submitting Ref2VA graph (expect a real cache MISS) ===", flush=True)
            cache_before, _ = server_log_cache_lines()
            t0 = time.time()
            prompt_id = submit_prompt()
            print("  prompt_id={}".format(prompt_id), flush=True)
            status_str, entry = wait_for_prompt_completion(prompt_id)
            dt = time.time() - t0
            print("  status={} round-trip {:.1f}s".format(status_str, dt), flush=True)
            if status_str != "success":
                print(json.dumps(entry, indent=2)[:3000] if entry else "(no history entry)", flush=True)
                raise RuntimeError("MISS submission did not succeed (status={})".format(status_str))

            cache_after, executed = server_log_cache_lines()
            new_cache = cache_after[len(cache_before):]
            server_rss_kb = _read_proc_field_kb("/proc/{}/status".format(server_pid), "VmRSS")
            mem_available_kb = _read_proc_field_kb("/proc/meminfo", "MemAvailable")
            results["MISS"] = {
                "round_trip_s": dt,
                "new_cache_lines": new_cache,
                "last_executed_line": executed[-1] if executed else None,
                "server_rss_kb": server_rss_kb,
                "mem_available_kb": mem_available_kb,
            }
            print("  new [CACHE ...] lines: {}".format(new_cache), flush=True)
            print("  server log: {}".format(executed[-1] if executed else "(no 'Prompt executed' line)"), flush=True)
            print("  server_RSS={}kB MemAvailable={}kB".format(server_rss_kb, mem_available_kb), flush=True)

    except OrchestratorShutdownSignal as sig:
        shutdown_signum = sig.signum
        print("!!! Orchestrator received signal {} -- stopping the server through the "
              "normal teardown (SIGINT->SIGTERM->SIGKILL), not a bare exit !!!".format(
                  sig.signum), flush=True)
    finally:
        if watchdog is not None:
            watchdog.stop()
        stop_live_server(server_proc, skip_sigint=stopped_by_watchdog)
        server_log_f.close()

    if shutdown_signum is not None:
        print("=== Run interrupted by signal {}; server stopped via normal teardown. ===".format(
            shutdown_signum), flush=True)
        sys.exit(128 + shutdown_signum)

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

    d = results.get("MISS")
    print("\n--- MISS submission ---")
    if not d:
        print("  (not reached)")
    else:
        print("  round-trip:        {:.1f}s".format(d["round_trip_s"]))
        print("  [CACHE ...] lines: {}".format(d["new_cache_lines"] or "(NONE)"))
        print("  server exec line:  {}".format(d["last_executed_line"]))
        print("  server RSS:        {} kB".format(d["server_rss_kb"]))
        print("  MemAvailable:      {} kB".format(d["mem_available_kb"]))

    print("\n--- nvidia-smi after server stop ---")
    print(nvidia_smi())

    # verdict
    miss = results.get("MISS", {})
    miss_ok = any("[CACHE MISS]" in ln for ln in miss.get("new_cache_lines", []))

    # Hand the fingerprint this run actually observed to the HIT-path follow-up
    # script (test_ref2video_server_hit.py), so it never needs a hardcoded value.
    miss_fp = None
    for ln in miss.get("new_cache_lines", []):
        m = re.search(r"\[CACHE MISS\]\s+([0-9a-f]+)", ln)
        if m:
            miss_fp = m.group(1)
            break
    if miss_fp:
        FINGERPRINT_HANDOFF_PATH.write_text(miss_fp)
        print("=== Wrote observed fingerprint {} to {} for the HIT-path "
              "follow-up script ===".format(miss_fp, FINGERPRINT_HANDOFF_PATH), flush=True)

    ok = results.get("nodes_registered") and miss_ok
    if stopped_by_watchdog:
        ok = False
        print("\n!!! run aborted by RAM watchdog: {} !!!".format(watchdog.reason if watchdog else "?"))
    print("\n=== VERDICT: {} ===".format(
        "PASS (both nodes registered, real cache MISS observed; run "
        "test_ref2video_server_hit.py next for the HIT path)"
        if ok else "FAIL / INCOMPLETE"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
