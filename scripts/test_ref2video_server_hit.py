"""R7 follow-up: exercise the CachedClipProxy HIT path for
MiniMaxH3CLIPCachedRef2VA through a LIVE ComfyUI server.

The first R7 run (test_ref2video_server_e2e.py) proved the MISS path: a real
[CACHE MISS] + ~30s Qwen3-VL encode, and it wrote a cache entry. It also
records the fingerprint it actually observed to a small handoff file
(/tmp/r7_last_fingerprint.txt); this script reads that value back rather than
carrying a hardcoded fingerprint that goes stale every time a fingerprint
input changes (encoder ABI, cache schema version, stat fields, hash framing,
...). The MISS run's second iteration could not prove the HIT path itself
because an identical graph in the SAME server session is short-circuited by
ComfyUI's own execution cache ("Prompt executed in 0.00 seconds") before our
node ever re-runs.

This script uses a FRESH server (empty execution cache) and submits that same
graph once. Now our node actually executes, recomputes the fingerprint, finds
the entry the earlier MISS wrote, and must:
  - log [CACHE HIT] <the fingerprint from the handoff file>
  - never emit "Requested to load MiniMaxH3TEModel_" (the ~26 GB encoder)
  - keep VRAM flat and low (video VAE only, no text encoder)
  - finish in ~0s of real compute

VRAM is sampled from nvidia-smi: before the prompt (server up, nothing
loaded) and right after it completes (still alive), so encoder residency
would show as a multi-GB jump.

Run under a hard timeout:
    timeout 900 conda run -n comfyenv --no-capture-output python -u \
        custom_nodes/ComfyUI-MiniMaxH3-CLIPCached/scripts/test_ref2video_server_hit.py
"""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import psutil
import requests

COMFYUI_ROOT = "/home/kamil/ComfyUI"
HOST, PORT = "127.0.0.1", 8188
BASE_URL = "http://{}:{}".format(HOST, PORT)

SERVER_LOG_PATH = "/tmp/r7_hit_server.log"
SERVER_STARTUP_TIMEOUT_S = 300
PROMPT_MAX_WAIT_S = 300
# Written by test_ref2video_server_e2e.py's MISS run and read back here, so
# this script never carries a hardcoded fingerprint that goes stale whenever
# a fingerprint input changes.
FINGERPRINT_HANDOFF_PATH = Path("/tmp/r7_last_fingerprint.txt")

CLIP_NAME = "qwen3vl_32b_minimax_h3_int8_convrot.safetensors"
VAE_NAME = "minimax_h3_video_vae_int8_convrot.safetensors"
AUDIO_VAE_NAME = "minimax_h3_audio_vae_fp32.safetensors"
PROMPT_TEXT = "a test prompt with <Picture 1>"
WIDTH, HEIGHT, LENGTH = 1344, 768, 124
REF_IMAGE_SIZE = "match"
REF_IMAGE_HW = 256

_server_pid_for_cleanup = None


def _sig(signum, frame):
    if _server_pid_for_cleanup:
        try:
            os.kill(_server_pid_for_cleanup, signal.SIGTERM)
        except ProcessLookupError:
            pass
    os._exit(1)


def nvidia_smi():
    try:
        return subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=30).stdout
    except Exception as e:
        return "(nvidia-smi failed: {})".format(e)


def vram_used_mib():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30).stdout.strip()
        return int(out.splitlines()[0])
    except Exception as e:
        return "(query failed: {})".format(e)


def wait_for_server_ready():
    deadline = time.time() + SERVER_STARTUP_TIMEOUT_S
    while time.time() < deadline:
        try:
            if requests.get(BASE_URL + "/system_stats", timeout=5).status_code == 200:
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(3)
    return False


def build_workflow():
    return {
        "1": {"class_type": "EmptyImage",
              "inputs": {"width": REF_IMAGE_HW, "height": REF_IMAGE_HW, "batch_size": 1, "color": 0x7F7F7F}},
        "2": {"class_type": "MiniMaxH3CLIPCachedRef2VA",
              "inputs": {"clip_name": CLIP_NAME, "vae": ["4", 0], "audio_vae": ["5", 0],
                         "prompt": PROMPT_TEXT, "width": WIDTH, "height": HEIGHT, "length": LENGTH,
                         "ref_image_size": REF_IMAGE_SIZE, "ref_image_0": ["1", 0], "cache_mode": "auto"}},
        "3": {"class_type": "PreviewAny", "inputs": {"source": ["2", 0]}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": VAE_NAME}},
        "5": {"class_type": "VAELoader", "inputs": {"vae_name": AUDIO_VAE_NAME}},
    }


def submit_prompt():
    r = requests.post(BASE_URL + "/prompt", json={"prompt": build_workflow()}, timeout=30)
    r.raise_for_status()
    data = r.json()
    if "error" in data:
        raise RuntimeError("rejected: {}".format(json.dumps(data, indent=2)))
    return data["prompt_id"]


def wait_for_completion(prompt_id):
    deadline = time.time() + PROMPT_MAX_WAIT_S
    while time.time() < deadline:
        data = requests.get(BASE_URL + "/history/{}".format(prompt_id), timeout=10).json()
        if prompt_id in data:
            return data[prompt_id].get("status", {}).get("status_str"), data[prompt_id]
        time.sleep(3)
    return "timeout", None


def stop_server(pid):
    try:
        os.kill(pid, signal.SIGINT)
    except ProcessLookupError:
        return
    for _grace, sig in ((45, signal.SIGTERM), (5, signal.SIGKILL)):
        deadline = time.time() + _grace
        while time.time() < deadline and psutil.pid_exists(pid):
            time.sleep(1)
        if not psutil.pid_exists(pid):
            break
        print("  escalating -> {}".format(sig), flush=True)
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            break


def main():
    global _server_pid_for_cleanup
    signal.signal(signal.SIGTERM, _sig)
    signal.signal(signal.SIGINT, _sig)

    if not FINGERPRINT_HANDOFF_PATH.is_file():
        print("!!! No fingerprint handoff file at {} -- run "
              "test_ref2video_server_e2e.py first !!!".format(FINGERPRINT_HANDOFF_PATH), flush=True)
        sys.exit(1)
    EXPECTED_FINGERPRINT = FINGERPRINT_HANDOFF_PATH.read_text().strip()
    print("=== Using fingerprint {} from {} ===".format(
        EXPECTED_FINGERPRINT, FINGERPRINT_HANDOFF_PATH), flush=True)

    cache_dir = Path(__file__).resolve().parent.parent / "cache"
    cache_files = sorted(p.name for p in cache_dir.glob(EXPECTED_FINGERPRINT + "*")) \
        if cache_dir.is_dir() else []
    print("=== Cache entries matching {} in {}: {} ===".format(
        EXPECTED_FINGERPRINT, cache_dir, cache_files), flush=True)
    if not cache_files:
        print("!!! No cache entry for {} -- run test_ref2video_server_e2e.py first !!!".format(
            EXPECTED_FINGERPRINT), flush=True)
        sys.exit(1)

    Path(SERVER_LOG_PATH).write_text("")
    print("=== Launching python main.py ===", flush=True)
    log_f = open(SERVER_LOG_PATH, "w")
    proc = subprocess.Popen([sys.executable, "main.py"], cwd=COMFYUI_ROOT,
                            stdout=log_f, stderr=subprocess.STDOUT)
    pid = proc.pid
    _server_pid_for_cleanup = pid
    print("=== Server PID={} ===".format(pid), flush=True)

    result = {}
    try:
        if not wait_for_server_ready():
            raise RuntimeError("server not ready -- see {}".format(SERVER_LOG_PATH))
        print("=== Server ready ===", flush=True)
        try:
            bound = {c.pid for c in psutil.net_connections(kind="inet")
                     if c.laddr and c.laddr.port == PORT and c.status == psutil.CONN_LISTEN and c.pid}
        except Exception as e:
            raise RuntimeError(
                "Could not verify ownership of port {} for launched PID {}; "
                "refusing to run against or stop an unverified server".format(PORT, pid)
            ) from e
        if proc.poll() is not None or pid not in bound:
            raise RuntimeError(
                "Port {} is owned by PID(s) {}, not the launched server PID {}. "
                "Another ComfyUI may already be running; refusing to adopt or stop it.".format(
                    PORT, sorted(bound), pid)
            )

        vram_before = vram_used_mib()
        print("=== VRAM before prompt: {} MiB ===".format(vram_before), flush=True)

        print("=== Submitting identical Ref2VA graph (expect proxy CACHE HIT) ===", flush=True)
        t0 = time.time()
        prompt_id = submit_prompt()
        status_str, hist = wait_for_completion(prompt_id)
        dt = time.time() - t0
        print("=== status={} round-trip {:.1f}s ===".format(status_str, dt), flush=True)
        if status_str != "success":
            print(json.dumps(hist, indent=2)[:3000] if hist else "(no history)", flush=True)

        time.sleep(2)
        vram_after = vram_used_mib()
        print("=== VRAM after prompt: {} MiB ===".format(vram_after), flush=True)

        log_text = Path(SERVER_LOG_PATH).read_text(errors="replace")
        cache_lines = [ln for ln in log_text.splitlines() if "[CACHE " in ln]
        te_load_lines = [ln for ln in log_text.splitlines() if "MiniMaxH3TEModel" in ln]
        exec_lines = [ln for ln in log_text.splitlines() if "Prompt executed in" in ln]
        result = {
            "status": status_str, "round_trip_s": dt,
            "cache_lines": cache_lines, "te_load_lines": te_load_lines,
            "exec_lines": exec_lines,
            "vram_before": vram_before, "vram_after": vram_after,
        }
    finally:
        print("=== Stopping server ===", flush=True)
        stop_server(pid)
        print("=== Server exited: {} ===".format(not psutil.pid_exists(pid)), flush=True)
        log_f.close()

    print("\n" + "=" * 70)
    print("=== R7 HIT-PATH RESULT ===")
    print("=" * 70)
    print("  status:            {}".format(result.get("status")))
    print("  round-trip:        {:.1f}s".format(result.get("round_trip_s", -1)))
    print("  [CACHE ...] lines: {}".format(result.get("cache_lines") or "(NONE)"))
    print("  TE-model load:     {}".format(result.get("te_load_lines") or "(NONE -- encoder never loaded)"))
    print("  exec line:         {}".format(result.get("exec_lines")))
    print("  VRAM before:       {} MiB".format(result.get("vram_before")))
    print("  VRAM after:        {} MiB".format(result.get("vram_after")))
    print("\n--- nvidia-smi after server stop ---")
    print(nvidia_smi())

    hit_ok = any("[CACHE HIT] " + EXPECTED_FINGERPRINT in ln for ln in result.get("cache_lines", []))
    no_encoder = not result.get("te_load_lines")
    ok = result.get("status") == "success" and hit_ok and no_encoder
    print("=== VERDICT: {} ===".format(
        "PASS (proxy CACHE HIT through live server, encoder never loaded)" if ok
        else "FAIL / INCOMPLETE"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
