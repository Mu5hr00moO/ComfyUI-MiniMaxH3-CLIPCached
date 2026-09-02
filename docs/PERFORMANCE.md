# Performance & Benchmarks

CLIPCached targets one specific part of the MiniMax H3 workflow: the expensive
Qwen3-VL text/vision conditioning stage. The benchmark therefore stops after
H3 conditioning/latent preparation, before diffusion sampling begins.

The final benchmark report uses **report format/schema v3** and compares the same five FL2VA
cases in three modes: **Native**, **CLIPCached MISS**, and **CLIPCached HIT**.
That produces **15 accepted measured runs** in total. No measured run was
rerun or discarded.

The run was produced by [`scripts/benchmark_conditioning.py`](../scripts/benchmark_conditioning.py)
with the benchmark-only dependencies listed in
[`benchmark-requirements.txt`](../scripts/benchmark-requirements.txt). The numbers in
this document come from the final `conditioning_benchmark.json` report. The
benchmark report format/schema v3 is separate from CLIPCached's **on-disk cache
schema v2**.

## Compared modes

| Mode | What is measured | Encoder read / encode | Cache write |
|---|---|---:|---:|
| **Native** | Stock ComfyUI MiniMax H3 conditioning path | Yes | No |
| **CLIPCached MISS** | No reusable cache entry exists; CLIPCached performs the real encode and saves the result | Yes | Yes |
| **CLIPCached HIT** | Matching conditioning is restored from disk | **No** | No |

Native and MISS intentionally include the cost of loading and running the real
Qwen3-VL encoder. HIT measures the repeat-use case CLIPCached is designed for:
the stored conditioning is reused without loading `MiniMaxH3TEModel` / Qwen3-VL.

## Filesystem and prewarm policy

The benchmark explicitly controls operating-system file residency so the large
encoder checkpoint is cold for every Native and MISS measurement. Before each
of those runs, the encoder file is evicted with `POSIX_FADV_DONTNEED`, and
`mincore` is used to verify that its resident-page ratio is at most **1%**.

The VAE is prewarmed. This isolates the comparison from unrelated VAE cold-load
noise while preserving the real cold-read cost of the encoder.

For CLIPCached HIT, the VAE and the matching cache entry are prewarmed. The
encoder itself is not read. This makes HIT a measurement of restoring reusable
conditioning rather than a hidden encoder warm-cache path.

These rules are deliberate. A slow Native or MISS run caused by a real cold
checkpoint read is part of the measurement and is not removed merely because
it is an outlier.

## Run acceptance

Each of the five cases was measured once in each mode. All **15/15** runs were
accepted. The completed run set had:

- no reruns,
- no cross-mode contamination,
- successful server shutdown for every run,
- process high-water memory (`VmHWM`) captured for every run,
- GPU release confirmed for **15/15** runs,
- no `MiniMaxH3TEModel` load on any HIT.

## Final results

The primary public summary uses **median + MAD (median absolute deviation)**
across the five cases in each mode.

| Mode | Conditioning stage | Peak VRAM | VRAM delta | Peak process RAM | Process RAM delta |
|---|---:|---:|---:|---:|---:|
| Native | **29.85 s** (MAD 0.82 s) | 15.24 GiB | 13.41 GiB | 29.25 GiB | 27.22 GiB |
| CLIPCached MISS | **32.23 s** (MAD 0.36 s) | 15.24 GiB | 13.41 GiB | 28.25 GiB | 26.22 GiB |
| CLIPCached HIT | **1.12 s** (MAD 0.02 s) | 2.67 GiB | 0.84 GiB | 3.38 GiB | 1.33 GiB |

Observed conditioning-stage ranges across the five cases were:

| Mode | Minimum | Maximum |
|---|---:|---:|
| Native | 27.61 s | 164.57 s |
| CLIPCached MISS | 31.39 s | 102.70 s |
| CLIPCached HIT | 1.08 s | 1.22 s |

The much wider Native and MISS ranges are expected under the deliberate cold
encoder-read policy. They are reported rather than filtered away. Median + MAD
is therefore the primary comparison for normal README-level interpretation.

### Individual accepted runs

All measured runs are listed below. Values come directly from the final
`conditioning_benchmark.json`; no run was discarded or replaced.

| Case | Mode | Conditioning stage | Peak VRAM | VRAM delta | Peak process RAM | Process RAM delta |
|---|---|---:|---:|---:|---:|---:|
| 1 | Native | 27.61 s | 15.24 GiB | 13.41 GiB | 29.27 GiB | 27.22 GiB |
| 1 | CLIPCached MISS | 31.39 s | 15.24 GiB | 13.41 GiB | 28.25 GiB | 26.22 GiB |
| 1 | CLIPCached HIT | 1.08 s | 2.67 GiB | 0.84 GiB | 3.36 GiB | 1.33 GiB |
| 2 | Native | 29.97 s | 15.71 GiB | 13.88 GiB | 29.25 GiB | 27.22 GiB |
| 2 | CLIPCached MISS | 32.23 s | 15.24 GiB | 13.41 GiB | 28.24 GiB | 26.21 GiB |
| 2 | CLIPCached HIT | 1.12 s | 2.67 GiB | 0.84 GiB | 3.38 GiB | 1.33 GiB |
| 3 | Native | 29.03 s | 15.24 GiB | 13.41 GiB | 29.25 GiB | 27.22 GiB |
| 3 | CLIPCached MISS | 31.88 s | 15.24 GiB | 13.41 GiB | 28.28 GiB | 26.23 GiB |
| 3 | CLIPCached HIT | 1.12 s | 2.67 GiB | 0.84 GiB | 3.38 GiB | 1.33 GiB |
| 4 | Native | 29.85 s | 15.24 GiB | 13.41 GiB | 29.25 GiB | 27.22 GiB |
| 4 | CLIPCached MISS | 102.70 s | 15.24 GiB | 13.41 GiB | 28.25 GiB | 26.22 GiB |
| 4 | CLIPCached HIT | 1.10 s | 2.67 GiB | 0.84 GiB | 3.38 GiB | 1.33 GiB |
| 5 | Native | 164.57 s | 15.24 GiB | 13.41 GiB | 29.31 GiB | 27.28 GiB |
| 5 | CLIPCached MISS | 32.59 s | 15.24 GiB | 13.41 GiB | 28.26 GiB | 26.21 GiB |
| 5 | CLIPCached HIT | 1.22 s | 2.67 GiB | 0.84 GiB | 3.38 GiB | 1.34 GiB |

## Interpretation

A **MISS** is not the fast path. It still has to load and run Qwen3-VL and then
write the reusable conditioning to disk, so its median time is close to Native
and can be slightly higher.

A **HIT** is the fast path. In this benchmark, median conditioning time fell
from **29.85 s** Native to **1.12 s** on HIT, while peak process RAM fell from
**29.25 GiB** to **3.38 GiB** and peak VRAM from **15.24 GiB** to **2.67 GiB**.
The diffusion/sampling stage is unchanged; these gains apply to the
conditioning stage only.

The trade-off is persistent disk usage. Every unique encoder-visible
conditioning request can create a cache entry that remains until it is deleted.

## Benchmark environment

| Item | Benchmark value |
|---|---|
| GPU | NVIDIA GeForce RTX 5080 (16,303 MiB reported VRAM) |
| CPU | AMD Ryzen 9 9900X3D 12-Core Processor |
| System RAM | 58,970,329,088 bytes (~54.9 GiB) |
| OS / kernel | WSL2 Linux; kernel `6.18.40.1-microsoft-standard-WSL2-zswap` |
| Python | 3.14.6 (Anaconda build) |
| PyTorch | `2.13.0+cu130` |
| ComfyUI | `0.34.2` |
| ComfyUI commit | `169fcf35a2fc163fec31338b816503ddac0d3fcf` |
| CLIPCached benchmark repo commit | `e9c2b68e78e70ef04c55531e8e14a71a9c16874e` |
| Encoder | `qwen3vl_32b_minimax_h3_int8_convrot.safetensors` |
| VAE | `minimax_h3_video_vae_int8_convrot.safetensors` |
| Benchmark resolution | 1344 × 768 |
| Benchmark length | 124 frames |
| Runs | 5 cases × Native / MISS / HIT = 15 accepted measurements |
| Sampling interval | 0.1 s |

Filesystem-cache policy was part of the controlled methodology rather than an
uncontrolled environmental detail. The VAE was prewarmed for every path. For
Native and Cached MISS, the encoder file was forced cold with
`POSIX_FADV_DONTNEED`, then checked with Linux `mmap`/`mincore`; encoder
residency had to be at or below **1%** before the measurement was accepted. For
Cached HIT, the matching cache entry was prewarmed and the encoder was not read.
The benchmark required prewarmed files to reach at least **99%** residency.

VRAM was sampled device-wide through NVML, so the reported peak is not a
per-process allocation. Process RAM primarily uses reset-scoped `VmHWM`, with
explicit RSS polling as a labelled portability fallback.
