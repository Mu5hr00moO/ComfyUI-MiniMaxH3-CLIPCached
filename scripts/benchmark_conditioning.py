"""Conditioning-stage benchmark for native and cached MiniMax H3 FL2VA.

This is a manual, GPU-required benchmark.  It starts ``python main.py`` for
every measurement and submits workflows only through ComfyUI's HTTP/WebSocket
API.  The benchmark process never imports ComfyUI, torch, or a CLIP loader.
``nvidia-ml-py`` (imported as ``pynvml``) is a benchmark-only prerequisite.

The three groups use the same five unique prompt cases:

* native ``CLIPLoader -> MiniMaxH3ImageToVideo``;
* ``MiniMaxH3CLIPCachedFL2VA`` cache MISS;
* ``MiniMaxH3CLIPCachedFL2VA`` disk cache HIT.

Every attempt gets a new server process.  Before the baseline, the files a
path legitimately reuses (the VAE, and for a HIT the cache entry) are
prewarmed and verified with Linux ``mincore()``, while the ~27 GB encoder is
actively evicted from the page cache with ``POSIX_FADV_DONTNEED`` for the
Native and Cached MISS paths so each pays a genuine cold load.  FL2VA
conditioning-stage wall time ends when ComfyUI announces that
the output sink is about to execute, which proves that the upstream FL2VA node
has returned its CONDITIONING/LATENT outputs while excluding the sink itself.
It includes the minimal VAE/keyframe/latent preparation upstream of those
outputs; it is not pure Qwen encoder time.  There is no sampler, decoder, or
video encoder in either workflow.

Tensor equality is deliberately outside this performance benchmark.  The
repository's stock-vs-cache correctness tests cover that concern separately.

Install the benchmark-only dependency first -- it is deliberately not in any
node-level requirements::

    conda run -n comfyenv python -m pip install -r scripts/benchmark-requirements.txt

Then run from this repository with its existing ComfyUI environment::

    conda run -n comfyenv --no-capture-output python -u \
        scripts/benchmark_conditioning.py

Results are written incrementally to
``benchmark_results/conditioning_benchmark.json``.  Server logs are retained
next to it so every cache verdict can be audited.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import ctypes
import json
import mmap
import os
import platform
import re
import socket
import statistics
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiohttp
import psutil

try:
    import pynvml
except ImportError:
    pynvml = None

from _live_server import (
    OrchestratorShutdownSignal,
    install_shutdown_signal_handler,
    stop_live_server,
)


# os.path.abspath, not Path.resolve / os.path.realpath: a common install
# keeps this repo outside custom_nodes/ and symlinks it in, so the
# invocation path is <ComfyUI>/custom_nodes/<symlink>/scripts/<this file>.
# Walking that non-resolved path keeps every step inside the symlinked
# layout, so REPO_ROOT.parent.parent lands on the real ComfyUI root;
# resolving the symlink first would climb from the repo's true location
# and overshoot. Mirrors tests/conftest.py. There is deliberately no
# COMFYUI_ROOT env override here (unlike the conftest-style scripts): this
# benchmark already takes the ComfyUI checkout through --comfyui-root, and
# a second override path would only blur the precedence.
_here = os.path.abspath(__file__)
REPO_ROOT = Path(_here).parent.parent
DEFAULT_COMFYUI_ROOT = REPO_ROOT.parent.parent
RESULT_PATH = REPO_ROOT / "benchmark_results" / "conditioning_benchmark.json"
LOG_DIR = RESULT_PATH.parent / "logs"
CACHE_DIR = REPO_ROOT / "cache"

HOST = "127.0.0.1"
CASE_COUNT = 5
MAX_ATTEMPTS = 3
PREWARM_MIN_RESIDENCY_FRACTION = 0.99
# The Native and Cached MISS paths must read the encoder from a cold page
# cache.  After POSIX_FADV_DONTNEED a healthy run drops to ~0% residency
# (empirically exactly 0 on the target ext4/WSL2 box); 1% is a generous
# ceiling that still fails loudly if another process is pinning the file.
ENCODER_EVICT_MAX_RESIDENCY_FRACTION = 0.01
ENCODER_EVICT_VERIFY_ATTEMPTS = 5
ENCODER_EVICT_RETRY_DELAY_SECONDS = 0.2
VMHWM_RESET_MIN_TOLERANCE_BYTES = 64 * 1024 * 1024
VMHWM_RESET_RELATIVE_TOLERANCE = 0.05
FL2VA_NODE_ID = "2"
OUTPUT_NODE_ID = "3"
REQUIRED_NODES = (
    "CLIPLoader",
    "VAELoader",
    "EmptyImage",
    "MiniMaxH3ImageToVideo",
    "MiniMaxH3CLIPCachedFL2VA",
    "PreviewAny",
)
CACHE_EVENT_RE = re.compile(r"\[CACHE (HIT|MISS|REFRESH)\]\s+([0-9a-f]{12,64})")
FULL_FINGERPRINT_RE = re.compile(r"[0-9a-f]{64}")
ENCODER_LOAD_MARKER = "MiniMaxH3TEModel"

GROUPS = (
    ("native", "Native", "N/A"),
    ("cached_miss", "Cached MISS", "MISS"),
    ("cached_hit", "Cached HIT", "HIT"),
)
RERUN_GROUPS = {
    "native": "native",
    "miss": "cached_miss",
    "hit": "cached_hit",
}

SUMMARY_METRICS = {
    "fl2va_conditioning_stage_wall_seconds": (
        "seconds",
        lambda run: run["timing"]["fl2va_conditioning_stage_wall_seconds"],
    ),
    "peak_gpu_vram_mib": ("MiB", lambda run: run["memory"]["gpu_vram_mib"]["peak"]),
    "gpu_vram_delta_mib": ("MiB", lambda run: run["memory"]["gpu_vram_mib"]["delta"]),
    "peak_process_ram_bytes": (
        "bytes",
        lambda run: run["memory"]["process_ram_bytes"]["peak"],
    ),
    "process_ram_delta_bytes": (
        "bytes",
        lambda run: run["memory"]["process_ram_bytes"]["delta"],
    ),
    "system_mem_available_decrease_bytes": (
        "bytes",
        lambda run: run["memory"]["system_mem_available_bytes"]["decrease"],
    ),
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _command_output(command: list[str], timeout: float = 10.0) -> str | None:
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip()


def _nvml_text(value: Any) -> str:
    return value.decode(errors="replace") if isinstance(value, bytes) else str(value)


class NvmlDevice:
    """Small benchmark-only NVML wrapper; no nvidia-smi polling fallback."""

    def __init__(self, gpu_index: int):
        if pynvml is None:
            raise RuntimeError(
                "conditioning benchmark requires nvidia-ml-py (import pynvml); "
                "install it in the benchmark environment"
            )
        try:
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            self.index = gpu_index
            self._active = True
            self.compute_pids()
        except Exception as exc:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
            raise RuntimeError(
                "conditioning benchmark could not initialize NVML GPU {}: {}".format(
                    gpu_index, exc
                )
            ) from exc

    def close(self) -> None:
        if getattr(self, "_active", False):
            pynvml.nvmlShutdown()
            self._active = False

    def snapshot(self) -> dict[str, Any]:
        try:
            memory = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            return {
                "index": self.index,
                "uuid": _nvml_text(pynvml.nvmlDeviceGetUUID(self.handle)),
                "model": _nvml_text(pynvml.nvmlDeviceGetName(self.handle)),
                "total_vram_mib": memory.total / (1024 ** 2),
                "used_vram_mib": memory.used / (1024 ** 2),
            }
        except Exception as exc:
            raise RuntimeError("NVML device memory query failed: {}".format(exc)) from exc

    def compute_pids(self) -> set[int]:
        try:
            processes = pynvml.nvmlDeviceGetComputeRunningProcesses(self.handle)
        except Exception as exc:
            raise RuntimeError(
                "NVML compute-process query failed; contamination cannot be checked: {}".format(
                    exc
                )
            ) from exc
        return {int(process.pid) for process in processes}


def _cpu_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(errors="replace").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _git_commit(path: Path) -> str | None:
    return _command_output(["git", "-C", str(path), "rev-parse", "HEAD"])


def _port_has_listener(port: int) -> bool:
    try:
        with socket.create_connection((HOST, port), timeout=0.5):
            return True
    except OSError:
        return False


def _assert_server_owns_port(proc: subprocess.Popen[Any], port: int) -> None:
    try:
        listeners = {
            connection.pid
            for connection in psutil.net_connections(kind="tcp")
            if connection.laddr
            and connection.laddr.port == port
            and connection.status == psutil.CONN_LISTEN
            and connection.pid is not None
        }
    except psutil.Error as exc:
        raise RuntimeError(
            "could not verify ownership of port {} for launched PID {}; refusing "
            "to run against or stop an unverified server".format(port, proc.pid)
        ) from exc
    if proc.poll() is not None or proc.pid not in listeners:
        raise RuntimeError(
            "port {} is owned by PID(s) {}, not launched server PID {}; another "
            "ComfyUI may already be running; refusing to adopt or stop it".format(
                port, sorted(listeners), proc.pid
            )
        )


def _resolve_model_file(
    comfyui_root: Path,
    directories: tuple[str, ...],
    filename: str,
) -> Path:
    if Path(filename).is_absolute():
        raise RuntimeError("checkpoint names must be relative: {!r}".format(filename))
    candidates = [
        comfyui_root / "models" / directory / filename for directory in directories
    ]
    existing = [path.resolve() for path in candidates if path.is_file()]
    if len(existing) != 1:
        raise RuntimeError(
            "checkpoint {!r} resolved to {} files: {}".format(
                filename, len(existing), [str(path) for path in existing]
            )
        )
    return existing[0]


def _benchmark_model_files(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "encoder_checkpoint": _resolve_model_file(
            args.comfyui_root, ("text_encoders", "clip"), args.clip_name
        ),
        "vae_checkpoint": _resolve_model_file(
            args.comfyui_root, ("vae",), args.vae_name
        ),
    }


def _resident_pages_via_mincore(
    mapping: mmap.mmap,
    size: int,
    total_pages: int,
    path: Path,
) -> int:
    """Return how many pages of ``mapping`` Linux ``mincore()`` reports resident.

    ``mincore()`` reports page-cache residency of the file backing the mapping
    without faulting anything in, so this is safe to call on an untouched
    mapping to observe eviction as well as on a touched one to confirm prewarm.
    """
    libc = ctypes.CDLL(None, use_errno=True)
    try:
        mincore = libc.mincore
    except AttributeError as exc:
        raise RuntimeError("libc does not expose mincore()") from exc
    mincore.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_ubyte),
    ]
    mincore.restype = ctypes.c_int

    buffer_reference = ctypes.c_char.from_buffer(mapping)
    vector = (ctypes.c_ubyte * total_pages)()
    result = mincore(
        ctypes.c_void_p(ctypes.addressof(buffer_reference)),
        ctypes.c_size_t(size),
        vector,
    )
    del buffer_reference
    if result != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number), str(path))
    return sum(1 for value in vector if value & 1)


def _measure_residency(path: Path) -> tuple[int, int, int]:
    """Observe current page-cache residency of ``path`` without faulting it in."""
    size = path.stat().st_size
    page_size = mmap.PAGESIZE
    total_pages = (size + page_size - 1) // page_size
    with path.open("rb") as file_handle:
        with mmap.mmap(file_handle.fileno(), 0, access=mmap.ACCESS_COPY) as mapping:
            resident_pages = _resident_pages_via_mincore(
                mapping, size, total_pages, path
            )
    return resident_pages, total_pages, size


def _prewarm_file(path: Path, role: str) -> dict[str, Any]:
    """Touch every file page and verify Linux page residency with mincore()."""
    if sys.platform != "linux":
        raise RuntimeError("filesystem prewarm verification requires Linux mincore()")
    size = path.stat().st_size
    if size <= 0:
        raise RuntimeError("cannot prewarm empty required file: {}".format(path))

    page_size = mmap.PAGESIZE
    total_pages = (size + page_size - 1) // page_size

    with path.open("rb") as file_handle:
        with mmap.mmap(file_handle.fileno(), 0, access=mmap.ACCESS_COPY) as mapping:
            touched = 0
            for offset in range(0, size, page_size):
                touched ^= mapping[offset]
            del touched
            resident_pages = _resident_pages_via_mincore(
                mapping, size, total_pages, path
            )

    residency_fraction = resident_pages / total_pages
    metadata = {
        "role": role,
        "path": str(path),
        "file_size_bytes": size,
        "page_size_bytes": page_size,
        "resident_pages": resident_pages,
        "total_pages": total_pages,
        "residency_fraction": residency_fraction,
        "residency_percent": residency_fraction * 100.0,
        "required_residency_fraction": PREWARM_MIN_RESIDENCY_FRACTION,
        "verification_method": "linux_mmap_mincore",
        "verified": residency_fraction >= PREWARM_MIN_RESIDENCY_FRACTION,
    }
    if not metadata["verified"]:
        raise RuntimeError(
            "prewarm residency for {} is {:.3%}, below required {:.3%}".format(
                path, residency_fraction, PREWARM_MIN_RESIDENCY_FRACTION
            )
        )
    return metadata


def _evict_encoder_file(path: Path, role: str) -> dict[str, Any]:
    """Force ``path`` out of the Linux page cache and verify it with mincore().

    Native and Cached MISS must read the ~27 GB encoder from a cold page cache.
    The earlier design prewarmed it to full residency immediately before
    ComfyUI loaded the very same file into its own process, demanding double
    residency of half of system RAM and making the Native path thrash.  This
    instead asks the kernel to drop the file's clean pages
    (``POSIX_FADV_DONTNEED``) and confirms residency fell to near zero; it
    never raises residency above the idle baseline.
    """
    if sys.platform != "linux":
        raise RuntimeError("filesystem eviction verification requires Linux mincore()")
    size = path.stat().st_size
    if size <= 0:
        raise RuntimeError("cannot evict empty required file: {}".format(path))

    verify_attempts: list[dict[str, Any]] = []
    resident_pages = 0
    total_pages = 0
    for attempt_number in range(1, ENCODER_EVICT_VERIFY_ATTEMPTS + 1):
        with path.open("rb") as file_handle:
            os.posix_fadvise(file_handle.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        resident_pages, total_pages, size = _measure_residency(path)
        fraction = resident_pages / total_pages
        verify_attempts.append(
            {
                "attempt": attempt_number,
                "resident_pages": resident_pages,
                "residency_fraction": fraction,
            }
        )
        if fraction <= ENCODER_EVICT_MAX_RESIDENCY_FRACTION:
            break
        if attempt_number < ENCODER_EVICT_VERIFY_ATTEMPTS:
            time.sleep(ENCODER_EVICT_RETRY_DELAY_SECONDS)

    residency_fraction = resident_pages / total_pages
    metadata = {
        "role": role,
        "path": str(path),
        "file_size_bytes": size,
        "page_size_bytes": mmap.PAGESIZE,
        "resident_pages": resident_pages,
        "total_pages": total_pages,
        "residency_fraction": residency_fraction,
        "residency_percent": residency_fraction * 100.0,
        "maximum_residency_fraction": ENCODER_EVICT_MAX_RESIDENCY_FRACTION,
        "filesystem_state": "forced_cold_read",
        "eviction_method": "linux_posix_fadvise_dontneed",
        "verification_method": "linux_mmap_mincore",
        "verify_attempts": verify_attempts,
        "verified": residency_fraction <= ENCODER_EVICT_MAX_RESIDENCY_FRACTION,
    }
    if not metadata["verified"]:
        raise RuntimeError(
            "eviction residency for {} is {:.3%} after {} attempt(s), above the "
            "{:.3%} cold-read ceiling; another process may be pinning the "
            "file".format(
                path,
                residency_fraction,
                ENCODER_EVICT_VERIFY_ATTEMPTS,
                ENCODER_EVICT_MAX_RESIDENCY_FRACTION,
            )
        )
    return metadata


def _prepare_filesystem_cache_for_run(
    group_key: str,
    expected_fingerprint: str | None,
    model_files: dict[str, Path],
) -> list[dict[str, Any]]:
    """Prewarm the files a path legitimately reuses and force the encoder cold
    for the paths that must pay a real load.

    The VAE is prewarmed for every path exactly as before.  For Native and
    Cached MISS the encoder is evicted instead of prewarmed; for Cached HIT the
    encoder is never read, so only the cache entry files are added.
    """
    evict_specifications: list[tuple[str, Path]] = []
    prewarm_specifications: list[tuple[str, Path]] = [
        ("vae_checkpoint", model_files["vae_checkpoint"])
    ]
    if group_key in ("native", "cached_miss"):
        evict_specifications.append(
            ("encoder_checkpoint", model_files["encoder_checkpoint"])
        )
    else:
        if (
            expected_fingerprint is None
            or FULL_FINGERPRINT_RE.fullmatch(expected_fingerprint) is None
        ):
            raise RuntimeError("cached HIT prewarm requires a full MISS fingerprint")
        prewarm_specifications.extend(
            [
                (
                    "cached_conditioning_metadata",
                    CACHE_DIR / "{}.json".format(expected_fingerprint),
                ),
                (
                    "cached_conditioning_tensors",
                    CACHE_DIR / "{}.safetensors".format(expected_fingerprint),
                ),
            ]
        )

    results = []
    for role, path in evict_specifications:
        if not path.is_file():
            raise RuntimeError("required encoder file does not exist: {}".format(path))
        results.append(_evict_encoder_file(path, role))
    for role, path in prewarm_specifications:
        if not path.is_file():
            raise RuntimeError("required prewarm file does not exist: {}".format(path))
        results.append(_prewarm_file(path, role))
    return results


def _read_proc_memory(pid: int) -> dict[str, int]:
    values: dict[str, int] = {}
    status_path = Path("/proc") / str(pid) / "status"
    for line in status_path.read_text(encoding="utf-8").splitlines():
        name, separator, remainder = line.partition(":")
        if separator and name in ("VmRSS", "VmHWM", "VmSize"):
            fields = remainder.split()
            if len(fields) != 2 or fields[1] != "kB":
                raise RuntimeError("unexpected {} line: {!r}".format(name, line))
            values[name] = int(fields[0]) * 1024
    if "VmRSS" not in values or "VmHWM" not in values:
        raise RuntimeError("{} does not expose VmRSS and VmHWM".format(status_path))
    return values


def _prepare_ram_peak_measurement(pid: int) -> dict[str, Any]:
    clear_refs_path = Path("/proc") / str(pid) / "clear_refs"
    try:
        clear_refs_path.write_text("5\n", encoding="ascii")
        status = _read_proc_memory(pid)
    except (OSError, RuntimeError, ValueError) as exc:
        return {
            "ram_peak_method": "rss_polling",
            "fallback_reason": "VmHWM reset unavailable: {}: {}".format(
                type(exc).__name__, exc
            ),
            "verified": False,
        }

    difference = abs(status["VmHWM"] - status["VmRSS"])
    tolerance = max(
        VMHWM_RESET_MIN_TOLERANCE_BYTES,
        int(status["VmRSS"] * VMHWM_RESET_RELATIVE_TOLERANCE),
    )
    if difference > tolerance:
        return {
            "ram_peak_method": "rss_polling",
            "fallback_reason": (
                "VmHWM reset sanity check failed: VmRSS={}, VmHWM={}, tolerance={}"
            ).format(status["VmRSS"], status["VmHWM"], tolerance),
            "vmrss_after_reset_bytes": status["VmRSS"],
            "vmhwm_after_reset_bytes": status["VmHWM"],
            "sanity_tolerance_bytes": tolerance,
            "verified": False,
        }
    return {
        "ram_peak_method": "vmhwm_reset",
        "vmrss_after_reset_bytes": status["VmRSS"],
        "vmhwm_after_reset_bytes": status["VmHWM"],
        "sanity_difference_bytes": difference,
        "sanity_tolerance_bytes": tolerance,
        "verified": True,
    }


def _memory_sample(
    process: psutil.Process,
    nvml_device: NvmlDevice,
    allowed_compute_pids: set[int],
) -> dict[str, Any]:
    process_memory = process.memory_info()
    virtual_memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    gpu = nvml_device.snapshot()
    compute_pids = nvml_device.compute_pids()
    return {
        "_monotonic": time.perf_counter(),
        "timestamp_utc": _utc_now(),
        "gpu_vram_used_mib": gpu["used_vram_mib"],
        "allowed_gpu_compute_pids": sorted(allowed_compute_pids),
        "gpu_compute_pids": sorted(compute_pids),
        "foreign_gpu_compute_pids": sorted(compute_pids - allowed_compute_pids),
        "process_rss_bytes": process_memory.rss,
        "process_vms_bytes": process_memory.vms,
        "system_mem_available_bytes": virtual_memory.available,
        "system_swap_used_bytes": swap.used,
    }


class MemorySampler:
    """Collect NVML device VRAM, PIDs, and host-memory diagnostic samples."""

    def __init__(
        self,
        process: psutil.Process,
        nvml_device: NvmlDevice,
        allowed_compute_pids: set[int],
        interval: float,
    ):
        self.process = process
        self.nvml_device = nvml_device
        self.allowed_compute_pids = allowed_compute_pids
        self.interval = interval
        self.samples: list[dict[str, Any]] = []
        self.error: BaseException | None = None
        self._origin = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self, origin: float) -> None:
        self._origin = origin
        self._thread.start()

    def _loop(self) -> None:
        next_sample = time.perf_counter()
        while not self._stop.is_set():
            try:
                sample = _memory_sample(
                    self.process, self.nvml_device, self.allowed_compute_pids
                )
                sample["elapsed_since_submission_seconds"] = (
                    sample["_monotonic"] - self._origin
                )
                self.samples.append(sample)
            except BaseException as exc:
                self.error = exc
                self._stop.set()
                return
            next_sample += self.interval
            self._stop.wait(max(0.0, next_sample - time.perf_counter()))

    def finish(self) -> None:
        self._stop.set()
        self._thread.join(timeout=15.0)
        if self._thread.is_alive():
            raise RuntimeError("memory sampler did not stop")
        if self.error is not None:
            raise RuntimeError("memory sampling failed: {}".format(self.error)) from self.error


def _public_sample(sample: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in sample.items() if key != "_monotonic"}


def _memory_result(
    baseline: dict[str, Any],
    samples: list[dict[str, Any]],
    boundary: float,
    ram_peak_setup: dict[str, Any],
    boundary_proc_memory: dict[str, int] | None,
) -> dict[str, Any]:
    measured = [sample for sample in samples if sample["_monotonic"] <= boundary]
    all_samples = [baseline, *measured]

    gpu_peak = max(sample["gpu_vram_used_mib"] for sample in all_samples)
    polled_rss_peak = max(sample["process_rss_bytes"] for sample in all_samples)
    vms_peak = max(sample["process_vms_bytes"] for sample in all_samples)
    mem_min = min(sample["system_mem_available_bytes"] for sample in all_samples)
    swap_peak = max(sample["system_swap_used_bytes"] for sample in all_samples)

    if (
        ram_peak_setup["ram_peak_method"] == "vmhwm_reset"
        and boundary_proc_memory is not None
    ):
        ram_baseline = ram_peak_setup["vmrss_after_reset_bytes"]
        ram_peak = max(ram_baseline, boundary_proc_memory["VmHWM"])
        ram_method = "vmhwm_reset"
    else:
        ram_baseline = baseline["process_rss_bytes"]
        ram_peak = polled_rss_peak
        ram_method = "rss_polling"

    observed_pids = sorted(
        {
            pid
            for sample in all_samples
            for pid in sample["gpu_compute_pids"]
        }
    )
    foreign_pids = sorted(
        {
            pid
            for sample in all_samples
            for pid in sample["foreign_gpu_compute_pids"]
        }
    )

    return {
        "vram_peak_method": "nvml_device_polling",
        "vram_scope": "device_wide",
        "ram_peak_method": ram_method,
        "gpu_vram_mib": {
            "baseline": baseline["gpu_vram_used_mib"],
            "peak": gpu_peak,
            "delta": gpu_peak - baseline["gpu_vram_used_mib"],
        },
        "process_ram_bytes": {
            "baseline": ram_baseline,
            "peak": ram_peak,
            "delta": ram_peak - ram_baseline,
            "baseline_source": (
                "VmRSS_after_clear_refs" if ram_method == "vmhwm_reset" else "VmRSS"
            ),
            "peak_source": "VmHWM" if ram_method == "vmhwm_reset" else "RSS_polling",
            "vmhwm_at_boundary_bytes": (
                boundary_proc_memory["VmHWM"]
                if ram_method == "vmhwm_reset" and boundary_proc_memory is not None
                else None
            ),
            "reset": ram_peak_setup,
        },
        "process_rss_timeline_bytes": {
            "baseline": baseline["process_rss_bytes"],
            "sampled_peak": polled_rss_peak,
            "sampled_delta": polled_rss_peak - baseline["process_rss_bytes"],
        },
        "process_vms_bytes": {
            "baseline": baseline["process_vms_bytes"],
            "peak": vms_peak,
            "delta": vms_peak - baseline["process_vms_bytes"],
        },
        "system_mem_available_bytes": {
            "baseline": baseline["system_mem_available_bytes"],
            "minimum": mem_min,
            "decrease": baseline["system_mem_available_bytes"] - mem_min,
            "scope": "system_memory_pressure",
        },
        "system_swap_used_bytes": {
            "baseline": baseline["system_swap_used_bytes"],
            "peak": swap_peak,
            "delta": swap_peak - baseline["system_swap_used_bytes"],
        },
        "sample_count": len(measured),
        "gpu_process_observation": {
            "allowed_compute_pids": sorted(
                baseline["allowed_gpu_compute_pids"]
            ),
            "observed_compute_pids": observed_pids,
            "foreign_compute_pids": foreign_pids,
            "contaminated": bool(foreign_pids),
            "method": "nvml_compute_process_polling",
        },
        "baseline_sample": _public_sample(baseline),
        "samples": [_public_sample(sample) for sample in measured],
    }


def _native_workflow(prompt: str, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "1": {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": args.clip_name, "type": "minimax"},
        },
        FL2VA_NODE_ID: {
            "class_type": "MiniMaxH3ImageToVideo",
            "inputs": {
                "clip": ["1", 0],
                "vae": ["4", 0],
                "prompt": prompt,
                "width": args.width,
                "height": args.height,
                "length": args.length,
                "first_frame": ["5", 0],
            },
        },
        OUTPUT_NODE_ID: {
            "class_type": "PreviewAny",
            "inputs": {"source": [FL2VA_NODE_ID, 0]},
        },
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": args.vae_name}},
        "5": {
            "class_type": "EmptyImage",
            "inputs": {
                "width": args.input_image_size,
                "height": args.input_image_size,
                "batch_size": 1,
                "color": 0x7F7F7F,
            },
        },
    }


def _cached_workflow(prompt: str, args: argparse.Namespace) -> dict[str, Any]:
    return {
        FL2VA_NODE_ID: {
            "class_type": "MiniMaxH3CLIPCachedFL2VA",
            "inputs": {
                "clip_name": args.clip_name,
                "vae": ["4", 0],
                "prompt": prompt,
                "width": args.width,
                "height": args.height,
                "length": args.length,
                "first_frame": ["5", 0],
                "cache_mode": "auto",
            },
        },
        OUTPUT_NODE_ID: {
            "class_type": "PreviewAny",
            "inputs": {"source": [FL2VA_NODE_ID, 0]},
        },
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": args.vae_name}},
        "5": {
            "class_type": "EmptyImage",
            "inputs": {
                "width": args.input_image_size,
                "height": args.input_image_size,
                "batch_size": 1,
                "color": 0x7F7F7F,
            },
        },
    }


async def _get_json(
    session: aiohttp.ClientSession,
    url: str,
    timeout: float = 10.0,
) -> dict[str, Any]:
    async with session.get(url, timeout=timeout) as response:
        response.raise_for_status()
        return await response.json()


async def _wait_for_ready(
    session: aiohttp.ClientSession,
    base_url: str,
    proc: subprocess.Popen[Any],
    timeout: float,
) -> dict[str, Any]:
    deadline = time.perf_counter() + timeout
    last_error: BaseException | None = None
    while time.perf_counter() < deadline:
        if proc.poll() is not None:
            raise RuntimeError("ComfyUI exited during startup with rc={}".format(proc.returncode))
        try:
            return await _get_json(session, base_url + "/system_stats", timeout=5.0)
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            last_error = exc
        await asyncio.sleep(0.5)
    raise RuntimeError("ComfyUI was not ready within {}s: {}".format(timeout, last_error))


async def _check_required_nodes(
    session: aiohttp.ClientSession,
    base_url: str,
) -> None:
    missing: list[str] = []
    for node_name in REQUIRED_NODES:
        payload = await _get_json(
            session,
            base_url + "/object_info/{}".format(node_name),
        )
        if node_name not in payload:
            missing.append(node_name)
    if missing:
        raise RuntimeError("required ComfyUI nodes are not registered: {}".format(missing))


async def _execute_until_fl2va_complete(
    session: aiohttp.ClientSession,
    base_url: str,
    workflow: dict[str, Any],
    process: psutil.Process,
    nvml_device: NvmlDevice,
    allowed_compute_pids: set[int],
    args: argparse.Namespace,
) -> dict[str, Any]:
    client_id = uuid.uuid4().hex
    websocket_url = "ws://{}:{}/ws?clientId={}".format(HOST, args.port, client_id)

    async with session.ws_connect(websocket_url, heartbeat=30.0) as websocket:
        ram_peak_setup = _prepare_ram_peak_measurement(process.pid)
        baseline = _memory_sample(process, nvml_device, allowed_compute_pids)
        sampler = MemorySampler(
            process, nvml_device, allowed_compute_pids, args.sampling_interval
        )
        print(
            "[BASELINE] VRAM={:.0f} MiB, VmRSS={:.2f} GiB, RAM peak method={}".format(
                baseline["gpu_vram_used_mib"],
                baseline["process_rss_bytes"] / (1024 ** 3),
                ram_peak_setup["ram_peak_method"],
            ),
            flush=True,
        )
        print("[MEASUREMENT] submitting /prompt", flush=True)
        started = time.perf_counter()
        started_utc = _utc_now()
        sampler.start(started)
        boundary: float | None = None
        boundary_utc: str | None = None
        boundary_proc_memory: dict[str, int] | None = None
        prompt_id: str | None = None
        execution_error: dict[str, Any] | None = None

        try:
            async with session.post(
                base_url + "/prompt",
                json={"prompt": workflow, "client_id": client_id},
                timeout=30.0,
            ) as response:
                response_text = await response.text()
                if response.status >= 400:
                    raise RuntimeError(
                        "prompt submission failed (HTTP {}): {}".format(
                            response.status, response_text[:2000]
                        )
                    )
                submitted = json.loads(response_text)
            if submitted.get("error"):
                raise RuntimeError("prompt rejected: {}".format(submitted))
            prompt_id = submitted["prompt_id"]

            deadline = started + args.prompt_timeout
            completed = False
            while time.perf_counter() < deadline:
                remaining = deadline - time.perf_counter()
                message = await asyncio.wait_for(websocket.receive(), timeout=remaining)
                if message.type == aiohttp.WSMsgType.BINARY:
                    continue
                if message.type != aiohttp.WSMsgType.TEXT:
                    raise RuntimeError("ComfyUI WebSocket closed before prompt completion")
                event = json.loads(message.data)
                data = event.get("data", {})
                if data.get("prompt_id") != prompt_id:
                    continue

                if event.get("type") == "executing" and data.get("node") == OUTPUT_NODE_ID:
                    if boundary is None:
                        boundary = time.perf_counter()
                        boundary_utc = _utc_now()
                        if ram_peak_setup["ram_peak_method"] == "vmhwm_reset":
                            try:
                                boundary_proc_memory = _read_proc_memory(process.pid)
                            except (OSError, RuntimeError, ValueError) as exc:
                                ram_peak_setup = {
                                    **ram_peak_setup,
                                    "ram_peak_method": "rss_polling",
                                    "verified": False,
                                    "fallback_reason": (
                                        "VmHWM boundary read unavailable: {}: {}"
                                    ).format(type(exc).__name__, exc),
                                }
                elif event.get("type") in ("execution_error", "execution_interrupted"):
                    execution_error = data
                    break
                elif event.get("type") == "execution_success":
                    completed = True
                    break

            if execution_error is not None:
                raise RuntimeError(
                    "ComfyUI execution failed: {}".format(
                        json.dumps(execution_error, ensure_ascii=False)[:3000]
                    )
                )
            if not completed:
                raise RuntimeError(
                    "prompt did not complete within {}s".format(args.prompt_timeout)
                )
            if boundary is None:
                raise RuntimeError(
                    "output-node start event was not observed; conditioning boundary is unproven"
                )
        finally:
            sampler.finish()

    assert boundary is not None
    return {
        "prompt_id": prompt_id,
        "timing": {
            "fl2va_conditioning_stage_wall_seconds": boundary - started,
            "measurement_started_utc": started_utc,
            "conditioning_completed_utc": boundary_utc,
            "metric": "FL2VA conditioning-stage wall time",
            "boundary": (
                "HTTP /prompt submission start to WebSocket 'executing' event for "
                "the downstream PreviewAny node"
            ),
            "includes": (
                "minimal upstream work required for FL2VA CONDITIONING/LATENT "
                "outputs, including applicable VAE/keyframe/latent preparation"
            ),
            "pure_qwen_encoder_time": False,
            "server_startup_included": False,
            "filesystem_cache_preparation_included": False,
            "downstream_output_node_included": False,
        },
        "memory": _memory_result(
            baseline,
            sampler.samples,
            boundary,
            ram_peak_setup,
            boundary_proc_memory,
        ),
    }


def _cache_events(log_path: Path) -> list[dict[str, str]]:
    text = log_path.read_text(errors="replace")
    return [
        {"kind": match.group(1), "fingerprint_prefix": match.group(2)}
        for match in CACHE_EVENT_RE.finditer(text)
    ]


async def _wait_for_cache_event(log_path: Path, timeout: float = 10.0) -> None:
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        if _cache_events(log_path):
            return
        await asyncio.sleep(0.1)


def _resolve_full_fingerprint(prefix: str) -> str:
    candidates = []
    if CACHE_DIR.is_dir():
        for path in CACHE_DIR.glob(prefix + "*.json"):
            if not path.name.endswith(".verbose.json") and len(path.stem) == 64:
                candidates.append(path.stem)
    candidates = sorted(set(candidates))
    if len(candidates) != 1:
        raise RuntimeError(
            "cache prefix {} resolved to {} core JSON entries: {}".format(
                prefix, len(candidates), candidates
            )
        )
    fingerprint = candidates[0]
    tensor_path = CACHE_DIR / "{}.safetensors".format(fingerprint)
    if not tensor_path.is_file():
        raise RuntimeError("cache entry is missing {}".format(tensor_path))
    return fingerprint


def _cache_artifacts(fingerprint: str) -> list[Path]:
    paths = [
        CACHE_DIR / "{}.json".format(fingerprint),
        CACHE_DIR / "{}.safetensors".format(fingerprint),
        CACHE_DIR / "{}.verbose.json".format(fingerprint),
    ]
    paths.extend(
        sorted((CACHE_DIR / "thumbnails").glob("{}_*.jpg".format(fingerprint)))
    )
    return [path for path in paths if path.is_file()]


async def _delete_cache_entry(
    session: aiohttp.ClientSession,
    base_url: str,
    fingerprint: str,
) -> dict[str, Any]:
    if FULL_FINGERPRINT_RE.fullmatch(fingerprint) is None:
        raise RuntimeError("invalid rerun cache fingerprint: {!r}".format(fingerprint))

    before = _cache_artifacts(fingerprint)
    request_timeout = aiohttp.ClientTimeout(total=30.0)
    async with session.post(
        base_url + "/h3_cache_manager/delete",
        json={"fingerprint": fingerprint},
        timeout=request_timeout,
    ) as response:
        try:
            payload = await response.json()
        except (aiohttp.ContentTypeError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                "cache delete returned HTTP {} with a non-JSON response".format(
                    response.status
                )
            ) from exc
        if (
            response.status != 200
            or not isinstance(payload, dict)
            or payload.get("deleted") != fingerprint
        ):
            raise RuntimeError(
                "cache delete failed for {}: HTTP {} {!r}".format(
                    fingerprint, response.status, payload
                )
            )

    remaining = _cache_artifacts(fingerprint)
    if remaining:
        raise RuntimeError(
            "cache delete left artifacts for {}: {}".format(
                fingerprint, [str(path) for path in remaining]
            )
        )
    return {
        "fingerprint": fingerprint,
        "endpoint": "/h3_cache_manager/delete",
        "artifacts_before": [str(path.relative_to(REPO_ROOT)) for path in before],
        "artifacts_remaining": [],
        "verified": True,
    }


def _verify_cache(
    group_key: str,
    log_path: Path,
    expected_fingerprint: str | None,
) -> dict[str, Any]:
    events = _cache_events(log_path)
    log_text = log_path.read_text(errors="replace")
    encoder_load_lines = [
        line for line in log_text.splitlines() if ENCODER_LOAD_MARKER in line
    ]

    if group_key == "native":
        if events:
            raise RuntimeError(
                "native workflow unexpectedly emitted cache events: {}".format(
                    events
                )
            )
        return {
            "expected": "N/A",
            "observed": "N/A",
            "verified": True,
            "events": events,
            "encoder_load_log_lines": encoder_load_lines,
        }

    expected_kind = "MISS" if group_key == "cached_miss" else "HIT"
    if len(events) != 1 or events[0]["kind"] != expected_kind:
        raise RuntimeError(
            "expected exactly one CACHE {}, observed {}".format(expected_kind, events)
        )

    prefix = events[0]["fingerprint_prefix"]
    if group_key == "cached_miss":
        full_fingerprint = _resolve_full_fingerprint(prefix)
        if (
            expected_fingerprint is not None
            and full_fingerprint != expected_fingerprint
        ):
            raise RuntimeError(
                "rerun MISS fingerprint {} does not match original fingerprint {}".format(
                    full_fingerprint, expected_fingerprint
                )
            )
    else:
        full_fingerprint = _resolve_full_fingerprint(prefix)
        if expected_fingerprint is None or full_fingerprint != expected_fingerprint:
            raise RuntimeError(
                "HIT fingerprint {} does not match MISS fingerprint {}".format(
                    full_fingerprint, expected_fingerprint
                )
            )
        if encoder_load_lines:
            raise RuntimeError(
                "cache HIT server loaded the MiniMax encoder: {}".format(encoder_load_lines)
            )

    return {
        "expected": expected_kind,
        "observed": events[0]["kind"],
        "verified": True,
        "fingerprint": full_fingerprint,
        "events": events,
        "encoder_load_log_lines": encoder_load_lines,
    }


def _tracked_process_pids(proc: subprocess.Popen[Any]) -> set[int]:
    pids = {proc.pid}
    try:
        pids.update(
            child.pid
            for child in psutil.Process(proc.pid).children(recursive=True)
        )
    except psutil.Error:
        pass
    return pids


def _wait_for_gpu_release(
    tracked_pids: set[int],
    nvml_device: NvmlDevice,
    pre_server_vram_mib: float,
    timeout: float,
    tolerance_mib: float,
) -> dict[str, Any]:
    deadline = time.perf_counter() + timeout
    last_gpu = nvml_device.snapshot()
    last_active: list[int] = []
    pid_allocations_gone = False
    device_memory_restored = False
    while time.perf_counter() < deadline:
        compute_pids = nvml_device.compute_pids()
        last_active = sorted(tracked_pids.intersection(compute_pids))
        last_gpu = nvml_device.snapshot()
        pid_allocations_gone = not last_active
        device_memory_restored = (
            last_gpu["used_vram_mib"] <= pre_server_vram_mib + tolerance_mib
        )
        if pid_allocations_gone and device_memory_restored:
            return {
                "verified": True,
                "verification_basis": (
                    "tracked_compute_pids_absent_and_device_memory_restored"
                ),
                "tracked_pids": sorted(tracked_pids),
                "process_gpu_query_supported": True,
                "remaining_tracked_compute_pids": last_active,
                "pre_server_vram_mib": pre_server_vram_mib,
                "post_server_vram_mib": last_gpu["used_vram_mib"],
                "device_memory_returned_to_baseline": device_memory_restored,
                "tolerance_mib": tolerance_mib,
            }
        time.sleep(0.5)
    raise RuntimeError(
        "GPU release was incomplete after {}s "
        "(remaining tracked PIDs={}, pre-server VRAM={} MiB, "
        "current VRAM={} MiB, tolerance={} MiB, PID release succeeded={}, "
        "memory restoration succeeded={})".format(
            timeout,
            last_active,
            pre_server_vram_mib,
            last_gpu["used_vram_mib"],
            tolerance_mib,
            pid_allocations_gone,
            device_memory_restored,
        )
    )


def _server_environment(system_stats: dict[str, Any]) -> dict[str, Any]:
    system = system_stats.get("system", {})
    primary_device = (system_stats.get("devices") or [{}])[0]
    return {
        "python_version": system.get("python_version"),
        "pytorch_version": system.get("pytorch_version"),
        "comfyui_version": system.get("comfyui_version"),
        "server_os": system.get("os"),
        "primary_torch_device": {
            key: primary_device.get(key)
            for key in ("name", "type", "index", "vram_total", "torch_vram_total")
        },
    }


async def _run_one(
    group_key: str,
    group_label: str,
    case: dict[str, Any],
    workflow: dict[str, Any],
    expected_fingerprint: str | None,
    args: argparse.Namespace,
    nvml_device: NvmlDevice,
    model_files: dict[str, Path],
    attempt_number: int,
    delete_fingerprint: str | None = None,
) -> dict[str, Any]:
    case_index = case["case_index"]
    group_index = next(
        index for index, group in enumerate(GROUPS) if group[0] == group_key
    )
    measurement_index = (case_index - 1) * len(GROUPS) + group_index + 1
    attempt_id = uuid.uuid4().hex
    log_path = LOG_DIR / "{}_{}_attempt_{}_{}.log".format(
        group_key, case["case_id"], attempt_number, attempt_id[:8]
    )
    terminal_label = "CASE {} / {}".format(case_index, group_label.upper())
    result: dict[str, Any] = {
        "attempt_id": attempt_id,
        "attempt_number": attempt_number,
        "group": group_key,
        "path": group_label,
        "case_index": case_index,
        "case_id": case["case_id"],
        "prompt": case["prompt"],
        "process_state": "fresh_per_measurement",
        "filesystem_cache": (
            "vae_prewarmed_encoder_forced_cold"
            if group_key in ("native", "cached_miss")
            else "explicitly_prewarmed"
        ),
        "filesystem_cache_verification": "linux_mmap_mincore",
        "sampling_interval_seconds": args.sampling_interval,
        "accepted": False,
        "status": "running",
        "server_log": str(log_path.relative_to(REPO_ROOT)),
    }
    proc: subprocess.Popen[Any] | None = None
    log_file: Any = None
    failure: BaseException | None = None
    tracked_pids: set[int] = set()
    pre_server_vram_mib: float | None = None

    print("\n==================== START: {} ====================".format(terminal_label))
    print(
        "[{}/15] [ATTEMPT {}/{}] starting fresh ComfyUI server".format(
            measurement_index, attempt_number, MAX_ATTEMPTS
        ),
        flush=True,
    )

    try:
        if _port_has_listener(args.port):
            raise RuntimeError(
                "{}:{} already has a listener; stop it before benchmarking".format(
                    HOST, args.port
                )
            )

        pre_server_vram_mib = nvml_device.snapshot()["used_vram_mib"]
        environment = os.environ.copy()
        environment["PYTHONUNBUFFERED"] = "1"
        command = [
            sys.executable,
            "main.py",
            "--listen",
            HOST,
            "--port",
            str(args.port),
            "--cuda-device",
            str(args.gpu_index),
        ]
        result["server_command"] = command
        log_file = log_path.open("w", encoding="utf-8")
        startup_started = time.perf_counter()
        proc = subprocess.Popen(
            command,
            cwd=args.comfyui_root,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=environment,
        )
        result["server_pid"] = proc.pid

        session_timeout = aiohttp.ClientTimeout(total=None, sock_connect=10.0)
        async with aiohttp.ClientSession(timeout=session_timeout) as session:
            base_url = "http://{}:{}".format(HOST, args.port)
            system_stats = await _wait_for_ready(
                session, base_url, proc, args.startup_timeout
            )
            _assert_server_owns_port(proc, args.port)
            await _check_required_nodes(session, base_url)
            result["timing"] = {
                "server_startup_seconds": time.perf_counter() - startup_started
            }
            result["server_environment"] = _server_environment(system_stats)

            if delete_fingerprint is not None:
                print(
                    "[CACHE VERIFICATION] deleting only fingerprint {} before "
                    "the MISS attempt".format(delete_fingerprint),
                    flush=True,
                )
                result["cache_delete_before_attempt"] = await _delete_cache_entry(
                    session, base_url, delete_fingerprint
                )

            print(
                "[FS CACHE] prewarming reused files and forcing the encoder cold",
                flush=True,
            )
            fs_cache_started = time.perf_counter()
            result["filesystem_cache_preparation"] = await asyncio.to_thread(
                _prepare_filesystem_cache_for_run,
                group_key,
                expected_fingerprint,
                model_files,
            )
            result["timing"]["filesystem_cache_preparation_seconds"] = (
                time.perf_counter() - fs_cache_started
            )
            prewarmed = [
                entry
                for entry in result["filesystem_cache_preparation"]
                if "eviction_method" not in entry
            ]
            evicted = [
                entry
                for entry in result["filesystem_cache_preparation"]
                if "eviction_method" in entry
            ]
            print(
                "[FS CACHE] prewarmed {} file(s) at >= {:.1%} residency; forced "
                "{} encoder file(s) cold at <= {:.1%} residency".format(
                    len(prewarmed),
                    PREWARM_MIN_RESIDENCY_FRACTION,
                    len(evicted),
                    ENCODER_EVICT_MAX_RESIDENCY_FRACTION,
                ),
                flush=True,
            )

            tracked_pids = _tracked_process_pids(proc)
            execution = await _execute_until_fl2va_complete(
                session,
                base_url,
                copy.deepcopy(workflow),
                psutil.Process(proc.pid),
                nvml_device,
                tracked_pids,
                args,
            )
            result["timing"].update(execution["timing"])
            result["prompt_id"] = execution["prompt_id"]
            result["memory"] = execution["memory"]
            result["contamination"] = execution["memory"][
                "gpu_process_observation"
            ]

        if group_key != "native":
            await _wait_for_cache_event(log_path)
        print("[CACHE VERIFICATION] checking server log", flush=True)
        result["cache"] = _verify_cache(group_key, log_path, expected_fingerprint)
        print(
            "[CACHE VERIFICATION] observed {}".format(result["cache"]["observed"]),
            flush=True,
        )
        result["status"] = "success"
    except BaseException as exc:
        failure = exc
        result["status"] = "failed"
        result["error"] = "{}: {}".format(type(exc).__name__, exc)
    finally:
        cleanup_errors: list[str] = []
        if proc is not None:
            print("[SERVER STOP] stopping fresh ComfyUI process", flush=True)
            tracked_pids = _tracked_process_pids(proc)
            exit_code = stop_live_server(proc, sigint_grace_s=60.0)
            result["server_exit_code"] = exit_code
            if exit_code is None:
                cleanup_errors.append("ComfyUI process did not exit")
        if log_file is not None:
            log_file.close()
        if proc is not None and pre_server_vram_mib is not None:
            try:
                result["gpu_release_after_shutdown"] = await asyncio.to_thread(
                    _wait_for_gpu_release,
                    tracked_pids,
                    nvml_device,
                    pre_server_vram_mib,
                    args.gpu_release_timeout,
                    args.gpu_release_tolerance_mib,
                )
            except BaseException as exc:
                cleanup_errors.append("{}: {}".format(type(exc).__name__, exc))
        if cleanup_errors:
            result["cleanup_errors"] = cleanup_errors
            result["status"] = "failed"
            if failure is None:
                failure = RuntimeError("; ".join(cleanup_errors))
                result["error"] = "RuntimeError: {}".format("; ".join(cleanup_errors))

    if isinstance(failure, OrchestratorShutdownSignal):
        raise failure
    if failure is not None:
        print("  FAILED: {}".format(result["error"]), flush=True)
    else:
        print(
            "[MEASUREMENT] {:.2f}s, cache={}, peak VRAM {:.0f} MiB, "
            "peak process RAM {:.2f} GiB".format(
                result["timing"]["fl2va_conditioning_stage_wall_seconds"],
                result["cache"]["observed"],
                result["memory"]["gpu_vram_mib"]["peak"],
                result["memory"]["process_ram_bytes"]["peak"] / (1024 ** 3),
            ),
            flush=True,
        )
    print("===================== END: {} =====================".format(terminal_label))
    return result


async def _run_with_retries(
    group_key: str,
    group_label: str,
    case: dict[str, Any],
    workflow: dict[str, Any],
    expected_fingerprint: str | None,
    args: argparse.Namespace,
    nvml_device: NvmlDevice,
    model_files: dict[str, Path],
    initial_delete_fingerprint: str | None = None,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    attempts: list[dict[str, Any]] = []
    retry_fingerprint = expected_fingerprint
    delete_fingerprint = initial_delete_fingerprint

    for attempt_number in range(1, MAX_ATTEMPTS + 1):
        attempt = await _run_one(
            group_key,
            group_label,
            case,
            workflow,
            retry_fingerprint,
            args,
            nvml_device,
            model_files,
            attempt_number,
            delete_fingerprint=delete_fingerprint,
        )
        attempts.append(attempt)
        if attempt["status"] != "success":
            return None, attempts

        foreign_pids = attempt["contamination"]["foreign_compute_pids"]
        if not foreign_pids:
            attempt["accepted"] = True
            return attempt, attempts

        attempt["rejection_reason"] = "foreign_gpu_compute_process"
        print(
            "[CONTAMINATION / RETRY] foreign GPU compute PID(s) {} observed; "
            "attempt {}/{} rejected".format(
                foreign_pids, attempt_number, MAX_ATTEMPTS
            ),
            flush=True,
        )
        if group_key == "cached_miss":
            retry_fingerprint = attempt["cache"]["fingerprint"]
            delete_fingerprint = retry_fingerprint
        else:
            delete_fingerprint = None

    print(
        "[CONTAMINATION / RETRY] all {} attempts were contaminated".format(
            MAX_ATTEMPTS
        ),
        file=sys.stderr,
        flush=True,
    )
    return None, attempts


def _statistics_for(runs: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {"n": len(runs)}
    for metric_name, (unit, getter) in SUMMARY_METRICS.items():
        values = [float(getter(run)) for run in runs]
        median = statistics.median(values)
        output[metric_name] = {
            "unit": unit,
            "n": len(values),
            "mean": statistics.mean(values),
            "median": median,
            "median_absolute_deviation": statistics.median(
                [abs(value - median) for value in values]
            ),
            "minimum": min(values),
            "maximum": max(values),
            "standard_deviation": (
                statistics.stdev(values) if len(values) >= 2 else None
            ),
            "standard_deviation_kind": "sample",
        }
    return output


def _format_seconds(value: float) -> str:
    return "{:.2f}s".format(value)


def _format_gib_from_mib(value: float) -> str:
    return "{:.2f} GiB".format(value / 1024.0)


def _format_gib_from_bytes(value: float) -> str:
    return "{:.2f} GiB".format(value / (1024 ** 3))


def _markdown_tables(report: dict[str, Any]) -> str:
    summary_lines = [
        "| Path | n | Median stage time | MAD | Min–max | Median peak VRAM | "
        "Median VRAM delta | Median peak process RAM | Median process RAM delta |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for group_key, group_label, _ in GROUPS:
        stats = report["statistics"][group_key]
        timing_stats = stats["fl2va_conditioning_stage_wall_seconds"]
        summary_lines.append(
            "| {} | {} | {} | {} | {}–{} | {} | {} | {} | {} |".format(
                group_label,
                stats["n"],
                _format_seconds(timing_stats["median"]),
                _format_seconds(timing_stats["median_absolute_deviation"]),
                _format_seconds(timing_stats["minimum"]),
                _format_seconds(timing_stats["maximum"]),
                _format_gib_from_mib(stats["peak_gpu_vram_mib"]["median"]),
                _format_gib_from_mib(stats["gpu_vram_delta_mib"]["median"]),
                _format_gib_from_bytes(stats["peak_process_ram_bytes"]["median"]),
                _format_gib_from_bytes(stats["process_ram_delta_bytes"]["median"]),
            )
        )

    individual_lines = [
        "| Path | Case | Attempt | Cache | Startup | Stage wall time | Peak VRAM | "
        "VRAM delta | Peak process RAM | Process RAM delta | RAM method | "
        "MemAvailable decrease |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |",
    ]
    for run in report["runs"]:
        memory = run["memory"]
        individual_lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                run["path"],
                run["case_id"],
                run["attempt_number"],
                run["cache"]["observed"],
                _format_seconds(run["timing"]["server_startup_seconds"]),
                _format_seconds(
                    run["timing"]["fl2va_conditioning_stage_wall_seconds"]
                ),
                _format_gib_from_mib(memory["gpu_vram_mib"]["peak"]),
                _format_gib_from_mib(memory["gpu_vram_mib"]["delta"]),
                _format_gib_from_bytes(memory["process_ram_bytes"]["peak"]),
                _format_gib_from_bytes(memory["process_ram_bytes"]["delta"]),
                memory["ram_peak_method"],
                _format_gib_from_bytes(
                    memory["system_mem_available_bytes"]["decrease"]
                ),
            )
        )
    methodology_lines = [
        "Methodology / notes:",
        "",
        "- Values above use accepted, uncontaminated attempts only; rejected "
        "attempts remain in the JSON audit history.",
        "- Time is FL2VA conditioning-stage wall time, not pure Qwen encoder "
        "time; startup and filesystem cache preparation are excluded.",
        (
            "- The VAE (and, for a HIT, the cache entry) is prewarmed and "
            "verified with Linux mmap + mincore at a {:.1%} residency "
            "threshold. For Native and Cached MISS the ~27 GB encoder is "
            "instead evicted with POSIX_FADV_DONTNEED and verified below "
            "{:.1%} residency, so each pays a genuine cold load."
        ).format(
            PREWARM_MIN_RESIDENCY_FRACTION,
            ENCODER_EVICT_MAX_RESIDENCY_FRACTION,
        ),
        "- Process RAM uses reset-scoped VmHWM when supported; MemAvailable "
        "decrease is separate system memory pressure.",
        "- VRAM is device-wide NVML memory and its peak is sampled, so a very "
        "short peak may be missed.",
        "- The GPU should otherwise be idle; foreign compute PIDs make an "
        "attempt contaminated.",
    ]
    return "\n".join(
        [*summary_lines, "", *individual_lines, "", *methodology_lines]
    )


def _write_report(report: dict[str, Any]) -> None:
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = RESULT_PATH.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, RESULT_PATH)


def _initial_report(args: argparse.Namespace, gpu: dict[str, Any]) -> dict[str, Any]:
    benchmark_id = uuid.uuid4().hex
    cases = [
        {
            "case_index": index,
            "case_id": "case_{}".format(index),
            "prompt": "{} [conditioning-benchmark:{}:case_{}]".format(
                args.base_prompt, benchmark_id, index
            ),
        }
        for index in range(1, CASE_COUNT + 1)
    ]
    return {
        # 3: the encoder is forced cold (POSIX_FADV_DONTNEED) for Native and
        # Cached MISS instead of being prewarmed, so schema 2 timings measured
        # a materially different thing and must not be mixed in via --rerun.
        "schema_version": 3,
        "status": "running",
        "benchmark_id": benchmark_id,
        "started_at_utc": _utc_now(),
        "process_state": "fresh_per_measurement",
        "run_order": "interleaved_by_case_native_miss_hit",
        "filesystem_cache": (
            "vae_prewarmed_for_all_paths; encoder_forced_cold_for_native_and_"
            "cached_miss; cache_entry_prewarmed_for_cached_hit"
        ),
        "filesystem_cache_verification": "linux_mmap_mincore",
        "prewarm_required_residency_fraction": PREWARM_MIN_RESIDENCY_FRACTION,
        "encoder_eviction_method": "linux_posix_fadvise_dontneed",
        "encoder_eviction_max_residency_fraction": (
            ENCODER_EVICT_MAX_RESIDENCY_FRACTION
        ),
        "vram_peak_method": "nvml_device_polling",
        "vram_scope": "device_wide",
        "ram_peak_method": "vmhwm_reset_with_explicit_rss_polling_fallback",
        "timing_metric": "FL2VA conditioning-stage wall time",
        "execution_model": (
            "The orchestrator never imports ComfyUI, torch, or CLIP. Native CLIPLoader "
            "exists only as a node in the API workflow executed by the fresh server."
        ),
        "correctness_scope": (
            "Tensor equality/allclose is intentionally excluded; repository correctness "
            "tests cover stock-vs-cache tensor equivalence separately."
        ),
        "cache_isolation": (
            "A new UUID is embedded in all five case prompts. Cached MISS runs must "
            "prove [CACHE MISS] from the project log. Contaminated MISS retries delete "
            "only that case's recorded full fingerprint."
        ),
        "limitations": [
            "The VAE and any cache entry are deliberately prewarmed, so those "
            "reads are not cold-disk; the encoder is forced cold only for "
            "Native and Cached MISS.",
            "Encoder eviction relies on POSIX_FADV_DONTNEED plus a mincore "
            "check; it forces the encoder cold but does not control the rest "
            "of the page cache or kernel readahead during the load.",
            "VRAM is device-wide NVML memory on the target WSL2 environment, "
            "not per-process memory.",
            "NVML has no resettable peak equivalent to VmHWM here; sampled VRAM "
            "can miss a very short peak.",
            "The GPU should otherwise be idle; observed foreign compute PIDs reject the attempt.",
            "NVML compute-process observation is sampled and could miss an "
            "extremely short-lived foreign process.",
            "FL2VA conditioning-stage wall time includes minimal applicable "
            "VAE/keyframe/latent preparation and is not pure Qwen encoder time.",
            "RSS polling is an explicitly labelled portability fallback if "
            "reset-scoped VmHWM is unavailable and can miss a short RAM peak.",
        ],
        "configuration": {
            "comfyui_root": str(args.comfyui_root),
            "encoder_checkpoint": args.clip_name,
            "vae_checkpoint": args.vae_name,
            "width": args.width,
            "height": args.height,
            "length": args.length,
            "images": [
                {
                    "input": "first_frame",
                    "source": "EmptyImage",
                    "width": args.input_image_size,
                    "height": args.input_image_size,
                    "batch_size": 1,
                    "color": "0x7F7F7F",
                }
            ],
            "cache_mode": "auto",
            "runs_per_group": CASE_COUNT,
            "maximum_attempts_per_measurement": MAX_ATTEMPTS,
            "sampling_interval_seconds": args.sampling_interval,
            "prewarm_required_residency_fraction": PREWARM_MIN_RESIDENCY_FRACTION,
            "encoder_eviction_max_residency_fraction": (
                ENCODER_EVICT_MAX_RESIDENCY_FRACTION
            ),
            "gpu_index": args.gpu_index,
            "port": args.port,
        },
        "environment": {
            "gpu_model": gpu["model"],
            "gpu_uuid": gpu["uuid"],
            "total_vram_mib": gpu["total_vram_mib"],
            "total_system_ram_bytes": psutil.virtual_memory().total,
            "cpu": _cpu_name(),
            "python_version": platform.python_version(),
            "pytorch_version": None,
            "comfyui_version": None,
            "comfyui_commit": _git_commit(args.comfyui_root),
            "operating_system": platform.platform(),
            "kernel": platform.release(),
            "benchmark_date_utc": _utc_now(),
            "benchmark_script": "scripts/benchmark_conditioning.py",
            "benchmark_repo_commit": _git_commit(REPO_ROOT),
            "sampling_interval_seconds": args.sampling_interval,
            "encoder_checkpoint": args.clip_name,
        },
        "cases": cases,
        "workflows": {
            case["case_id"]: {
                "native": _native_workflow(case["prompt"], args),
                "cached_miss_and_hit": _cached_workflow(case["prompt"], args),
            }
            for case in cases
        },
        "runs": [],
        "attempts": [],
        "reruns": [],
        "statistics": {},
    }


async def _run_benchmark(args: argparse.Namespace) -> int:
    if not (args.comfyui_root / "main.py").is_file():
        raise RuntimeError("ComfyUI main.py not found under {}".format(args.comfyui_root))
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    model_files = _benchmark_model_files(args)
    nvml_device = NvmlDevice(args.gpu_index)
    try:
        report = _initial_report(args, nvml_device.snapshot())
        _write_report(report)
        fingerprints: dict[str, str] = {}
        observed_server_environment: dict[str, Any] | None = None

        for case in report["cases"]:
            for group_key, group_label, _ in GROUPS:
                workflow_key = (
                    "native" if group_key == "native" else "cached_miss_and_hit"
                )
                expected_fingerprint = fingerprints.get(case["case_id"])
                run, attempts = await _run_with_retries(
                    group_key,
                    group_label,
                    case,
                    report["workflows"][case["case_id"]][workflow_key],
                    expected_fingerprint,
                    args,
                    nvml_device,
                    model_files,
                )
                report["attempts"].extend(attempts)

                if run is None:
                    report["status"] = "failed"
                    report["error"] = (
                        attempts[-1].get("error")
                        or "all attempts were contaminated for {} {}".format(
                            group_label, case["case_id"]
                        )
                    )
                    report["finished_at_utc"] = _utc_now()
                    _write_report(report)
                    return 1

                current_server_environment = run["server_environment"]
                if observed_server_environment is None:
                    observed_server_environment = current_server_environment
                    report["environment"]["python_version"] = (
                        current_server_environment["python_version"]
                    )
                    report["environment"]["pytorch_version"] = (
                        current_server_environment["pytorch_version"]
                    )
                    report["environment"]["comfyui_version"] = (
                        current_server_environment["comfyui_version"]
                    )
                    report["environment"]["comfyui_server_os"] = (
                        current_server_environment["server_os"]
                    )
                    report["environment"]["primary_torch_device"] = (
                        current_server_environment["primary_torch_device"]
                    )
                elif current_server_environment != observed_server_environment:
                    run["accepted"] = False
                    run["status"] = "failed"
                    run["error"] = "server environment changed between benchmark runs"
                    report["status"] = "failed"
                    report["finished_at_utc"] = _utc_now()
                    _write_report(report)
                    return 1

                if group_key == "cached_miss":
                    fingerprint = run["cache"]["fingerprint"]
                    if fingerprint in fingerprints.values():
                        run["accepted"] = False
                        run["status"] = "failed"
                        run["error"] = (
                            "two unique benchmark cases produced one fingerprint"
                        )
                        report["status"] = "failed"
                        report["finished_at_utc"] = _utc_now()
                        _write_report(report)
                        return 1
                    fingerprints[case["case_id"]] = fingerprint

                report["runs"].append(run)
                _write_report(report)

        _refresh_report_summaries(report)
        _write_report(report)

        print("\n" + report["markdown_summary"])
        print("\nResults: {}".format(RESULT_PATH))
        print(
            "The VAE and any cache entry were prewarmed and verified; the "
            "encoder was forced cold with POSIX_FADV_DONTNEED for Native and "
            "Cached MISS."
        )
        return 0
    finally:
        nvml_device.close()


def _load_rerun_report(args: argparse.Namespace) -> dict[str, Any]:
    try:
        report = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(
            "--rerun requires an existing {}".format(RESULT_PATH)
        ) from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "could not read existing benchmark results: {}".format(exc)
        ) from exc

    if report.get("schema_version") != 3:
        raise RuntimeError(
            "--rerun requires a schema_version 3 report produced by the "
            "current scripts/benchmark_conditioning.py; found schema_version "
            "{!r}. This is a methodology change, not just a version number: "
            "schema 2 reports prewarmed the ~27 GB encoder into the page "
            "cache, whereas schema 3 forces it cold (POSIX_FADV_DONTNEED) for "
            "the Native and Cached MISS paths. Their timings are not "
            "comparable, so a schema 2 report cannot be extended in place -- "
            "run a fresh full benchmark instead.".format(
                report.get("schema_version")
            )
        )
    if report.get("status") not in ("complete", "rerun_failed"):
        raise RuntimeError(
            "--rerun requires a complete benchmark report (a prior failed "
            "rerun may be retried)"
        )

    expected_case_ids = {
        "case_{}".format(index) for index in range(1, CASE_COUNT + 1)
    }
    cases = report.get("cases")
    if (
        not isinstance(cases, list)
        or {case.get("case_id") for case in cases} != expected_case_ids
        or {case.get("case_index") for case in cases} != set(range(1, CASE_COUNT + 1))
    ):
        raise RuntimeError("existing report does not contain the expected five cases")

    runs = report.get("runs")
    if not isinstance(runs, list) or len(runs) != len(GROUPS) * CASE_COUNT:
        raise RuntimeError("existing report must contain exactly 15 runs")
    run_keys = [(run.get("group"), run.get("case_id")) for run in runs]
    expected_run_keys = {
        (group_key, case_id)
        for group_key, _, _ in GROUPS
        for case_id in expected_case_ids
    }
    if len(set(run_keys)) != len(run_keys) or set(run_keys) != expected_run_keys:
        raise RuntimeError("existing report does not contain one run per group and case")
    if any(
        run.get("status") != "success"
        or not run.get("accepted")
        or run.get("contamination", {}).get("contaminated")
        for run in runs
    ):
        raise RuntimeError(
            "--rerun requires 15 accepted, successful, uncontaminated runs"
        )

    attempts = report.get("attempts")
    if not isinstance(attempts, list):
        raise RuntimeError("existing report has no attempt audit history")

    workflows = report.get("workflows")
    if not isinstance(workflows, dict) or set(workflows) != expected_case_ids:
        raise RuntimeError("existing report does not contain all saved workflows")

    try:
        configuration = report["configuration"]
        args.comfyui_root = Path(configuration["comfyui_root"]).resolve()
        args.gpu_index = int(configuration["gpu_index"])
        args.port = int(configuration["port"])
        args.sampling_interval = float(configuration["sampling_interval_seconds"])
        args.clip_name = str(configuration["encoder_checkpoint"])
        args.vae_name = str(configuration["vae_checkpoint"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("existing report has an invalid configuration") from exc
    return report


def _report_server_environment(report: dict[str, Any]) -> dict[str, Any]:
    environment = report["environment"]
    return {
        "python_version": environment.get("python_version"),
        "pytorch_version": environment.get("pytorch_version"),
        "comfyui_version": environment.get("comfyui_version"),
        "server_os": environment.get("comfyui_server_os"),
        "primary_torch_device": environment.get("primary_torch_device"),
    }


def _refresh_report_summaries(report: dict[str, Any]) -> None:
    report["statistics"] = {}
    for group_key, _, _ in GROUPS:
        group_runs = [
            run
            for run in report["runs"]
            if run["group"] == group_key
            and run.get("accepted")
            and not run.get("contamination", {}).get("contaminated")
        ]
        if len(group_runs) != CASE_COUNT:
            raise RuntimeError(
                "cannot recalculate statistics for {}: expected {} runs, found {}".format(
                    group_key, CASE_COUNT, len(group_runs)
                )
            )
        report["statistics"][group_key] = _statistics_for(group_runs)
    report["status"] = "complete"
    report["finished_at_utc"] = _utc_now()
    report["markdown_summary"] = _markdown_tables(report)


async def _run_reruns(args: argparse.Namespace) -> int:
    report = _load_rerun_report(args)
    if not (args.comfyui_root / "main.py").is_file():
        raise RuntimeError(
            "ComfyUI main.py not found under {}".format(args.comfyui_root)
        )
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    model_files = _benchmark_model_files(args)

    group_key = args.rerun_group
    group_label = next(label for key, label, _ in GROUPS if key == group_key)
    cases_by_id = {case["case_id"]: case for case in report["cases"]}
    expected_server_environment = _report_server_environment(report)
    nvml_device = NvmlDevice(args.gpu_index)

    try:
        for case_id in args.rerun_case_ids:
            run_index = next(
                index
                for index, run in enumerate(report["runs"])
                if run["group"] == group_key and run["case_id"] == case_id
            )
            old_run = report["runs"][run_index]
            workflow_key = (
                "native" if group_key == "native" else "cached_miss_and_hit"
            )
            expected_fingerprint = None
            if group_key != "native":
                miss_run = next(
                    run
                    for run in report["runs"]
                    if run["group"] == "cached_miss"
                    and run["case_id"] == case_id
                )
                expected_fingerprint = miss_run.get("cache", {}).get("fingerprint")
                if (
                    not isinstance(expected_fingerprint, str)
                    or FULL_FINGERPRINT_RE.fullmatch(expected_fingerprint) is None
                ):
                    raise RuntimeError(
                        "existing Cached MISS {} has no valid full fingerprint".format(
                            case_id
                        )
                    )

            print(
                "\nRerunning {} {} only; the other 14 accepted records "
                "will be preserved.".format(group_label, case_id),
                flush=True,
            )
            run, attempts = await _run_with_retries(
                group_key,
                group_label,
                cases_by_id[case_id],
                report["workflows"][case_id][workflow_key],
                expected_fingerprint,
                args,
                nvml_device,
                model_files,
                initial_delete_fingerprint=(
                    expected_fingerprint if group_key == "cached_miss" else None
                ),
            )
            report["attempts"].extend(attempts)
            rerun_record = {
                "timestamp_utc": _utc_now(),
                "group": group_key,
                "case_id": case_id,
                "previous_accepted_attempt_id": old_run["attempt_id"],
                "new_attempt_ids": [attempt["attempt_id"] for attempt in attempts],
                "status": "failed" if run is None else "success",
            }

            if run is None or run["server_environment"] != expected_server_environment:
                if run is not None:
                    run["accepted"] = False
                    run["status"] = "failed"
                    run["error"] = (
                        "rerun server environment differs from the existing report"
                    )
                    rerun_record["status"] = "failed"
                report["reruns"].append(rerun_record)
                report["status"] = "rerun_failed"
                report["last_rerun_error"] = (
                    run.get("error")
                    if run is not None
                    else attempts[-1].get("error")
                    or "all rerun attempts were contaminated"
                )
                _write_report(report)
                print(
                    "Existing {} {} accepted result was preserved in the JSON.".format(
                        old_run["path"], case_id
                    ),
                    file=sys.stderr,
                )
                return 1

            for attempt in report["attempts"]:
                if (
                    attempt.get("group") == group_key
                    and attempt.get("case_id") == case_id
                    and attempt.get("accepted")
                    and attempt["attempt_id"] != run["attempt_id"]
                ):
                    attempt["accepted"] = False
                    attempt["superseded_by_attempt_id"] = run["attempt_id"]
                    attempt["superseded_at_utc"] = _utc_now()

            report["runs"][run_index] = run
            rerun_record["accepted_attempt_id"] = run["attempt_id"]
            report["reruns"].append(rerun_record)
            report.pop("last_rerun_error", None)
            _refresh_report_summaries(report)
            _write_report(report)
            print("  Replaced the existing {} {} record.".format(group_label, case_id))

        print("\n" + report["markdown_summary"])
        print("\nUpdated results: {}".format(RESULT_PATH))
        print(
            "The VAE and any cache entry were prewarmed and verified; the "
            "encoder was forced cold with POSIX_FADV_DONTNEED for Native and "
            "Cached MISS."
        )
        return 0
    finally:
        nvml_device.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--comfyui-root",
        type=Path,
        default=DEFAULT_COMFYUI_ROOT,
        help="ComfyUI checkout containing main.py (default: %(default)s)",
    )
    parser.add_argument(
        "--clip-name",
        default="qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
        help="text encoder checkpoint filename",
    )
    parser.add_argument(
        "--vae-name",
        default="minimax_h3_video_vae_int8_convrot.safetensors",
        help="MiniMax H3 video VAE checkpoint filename",
    )
    parser.add_argument("--base-prompt", default="A cinematic landscape")
    parser.add_argument("--width", type=int, default=1344)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--length", type=int, default=124)
    parser.add_argument("--input-image-size", type=int, default=256)
    parser.add_argument("--port", type=int, default=8188)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--sampling-interval", type=float, default=0.1)
    parser.add_argument("--startup-timeout", type=float, default=300.0)
    parser.add_argument("--prompt-timeout", type=float, default=900.0)
    parser.add_argument("--gpu-release-timeout", type=float, default=120.0)
    # Idle VRAM on this WSL2 box is not stable: with no CUDA process it drifts
    # across a ~1100-1650 MiB band (host driver/compositor overhead through the
    # GPU passthrough, documented in CLAUDE.md). A 64 MiB gate is inside that
    # noise and would raise a false "GPU release incomplete" after the timeout
    # on an otherwise healthy run, so allow the full observed drift plus margin.
    parser.add_argument("--gpu-release-tolerance-mib", type=float, default=600.0)
    parser.add_argument(
        "--rerun",
        nargs="+",
        metavar="VALUE",
        help="rerun selected existing results: GROUP CASE [CASE ...]",
    )
    args = parser.parse_args()
    if args.rerun is not None:
        if len(args.rerun) < 2:
            parser.error("--rerun requires GROUP and at least one CASE")
        requested_group = args.rerun[0].lower()
        if requested_group not in RERUN_GROUPS:
            parser.error("--rerun GROUP must be one of: native, miss, hit")
        case_ids: list[str] = []
        for value in args.rerun[1:]:
            try:
                case_number = int(value)
            except ValueError:
                parser.error("--rerun CASE must be an integer from 1 to 5")
            if not 1 <= case_number <= CASE_COUNT:
                parser.error("--rerun CASE must be an integer from 1 to 5")
            case_id = "case_{}".format(case_number)
            if case_id in case_ids:
                parser.error("--rerun CASE values must be unique")
            case_ids.append(case_id)
        args.rerun_group = RERUN_GROUPS[requested_group]
        args.rerun_case_ids = case_ids
    if args.sampling_interval <= 0:
        parser.error("--sampling-interval must be positive")
    if args.gpu_release_tolerance_mib < 0:
        parser.error("--gpu-release-tolerance-mib cannot be negative")
    if args.width <= 0 or args.height <= 0 or args.input_image_size <= 0:
        parser.error("image dimensions must be positive")
    for name in ("startup_timeout", "prompt_timeout", "gpu_release_timeout"):
        if getattr(args, name) <= 0:
            parser.error("--{} must be positive".format(name.replace("_", "-")))
    args.comfyui_root = args.comfyui_root.resolve()
    return args


def main() -> int:
    install_shutdown_signal_handler()
    try:
        args = _parse_args()
        if args.rerun is not None:
            return asyncio.run(_run_reruns(args))
        return asyncio.run(_run_benchmark(args))
    except OrchestratorShutdownSignal as signal_error:
        print(
            "Benchmark interrupted by signal {}; active ComfyUI server was stopped.".format(
                signal_error.signum
            ),
            file=sys.stderr,
        )
        return 128 + signal_error.signum
    except Exception as exc:
        print("Benchmark failed: {}: {}".format(type(exc).__name__, exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
