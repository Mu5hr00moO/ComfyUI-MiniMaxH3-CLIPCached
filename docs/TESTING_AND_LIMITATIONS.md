# Testing & Limitations

## Limitations

CLIPCached is deliberately narrow in scope. It solves repeated MiniMax H3
text/vision encoding, but it is not intended to become a general model loader,
prompt database, cache daemon, or replacement for ComfyUI's native H3 nodes.

The current version has the following limitations.

### No automatic cache eviction

Cached conditioning entries remain on disk until they are deleted.

There is currently no automatic policy based on:

- cache age,
- total cache size,
- number of entries,
- least-recently-used status.

Cache Manager makes manual cleanup easier, but deciding when and what to remove
is still the user's responsibility.

### No `cache_only` mode

The available modes are currently:

- `auto`,
- `refresh`.

With `auto`, a MISS falls through to the real Qwen3-VL encode. There is no
current mode that means:

> use an existing cache entry or fail, but never load the encoder.

A future `cache_mode="cache_only"` is planned, but it is not implemented.

### Ref2VA uses fixed reference slots

The CLIPCached Ref2VA family currently exposes:

- 9 reference-image slots,
- 3 reference-video slots,
- 3 matching video-audio slots,
- 3 standalone reference-audio slots.

ComfyUI's stock Ref2VA node uses dynamic `io.Autogrow` reference inputs. The
CLIPCached nodes do not currently reproduce that dynamic UI behavior.

If a workflow needs more references than the fixed limits, use the stock Ref2VA
node for that workflow.

### Encoder checkpoint path is filename-based

The cached H3 nodes take `clip_name` and load the selected checkpoint through
ComfyUI's normal MiniMax H3 loader path.

They do **not** accept an already-constructed `CLIP` object from an arbitrary
upstream node. This means a separately loaded, patched, or otherwise modified
CLIP object cannot simply be connected to CLIPCached in place of `clip_name`.

CLIPCached is a replacement for the MiniMax H3 **conditioning node**, not a
generic cache wrapper for every possible upstream CLIP pipeline.

### `.safetensors` is the tested encoder path

Development and real-world use have focused on MiniMax H3 encoder checkpoints
loaded through the normal `.safetensors` path.

GGUF MiniMax H3 text/vision encoders are currently **untested** with this
project and should not be assumed to work merely because another ComfyUI loader
can use them.

### One ComfyUI process per cache directory

The cache synchronization described in
[Technical Details](TECHNICAL_DETAILS.md#technical-details) uses in-process
Python locks.

The supported assumption is therefore:

> one running ComfyUI process owns a given CLIPCached `cache/` directory.

Two independent ComfyUI servers pointing at the same cache directory are not
coordinated by the current locking mechanism. This limitation is discussed in
more detail under
[Cache Compatibility, Upgrading & Warnings](TECHNICAL_DETAILS.md#cache-compatibility-upgrading--warnings).

### Cache Manager is not a prompt library

Cache Manager indexes conditioning entries that exist because the encoder has
already been run.

It can store user-facing names, notes, tags, favorites, thumbnails, and the
prompt associated with a cache entry, but it is not an independent prompt
database.

In particular:

- a prompt does not appear there until a corresponding cache entry exists,
- deleting the cache entry also removes that cached item's role in the manager,
- Cache Manager is not intended to replace a dedicated prompt-management tool.

### Reference source filenames: tracked for Ref2VA, not yet for FL2VA

For Ref2VA, Cache Manager can show the on-disk file a reference was
originally loaded from -- for example, that a reference came from
`portrait.png` or `reference.mp4` -- alongside the thumbnail and positional
metadata it already stored. This depends on the workflow: it works when a
reference traces back to a loader node, and may find nothing for a reference
built through less direct graph paths.

FL2VA's `first_frame` / `last_frame` keyframes do not have this yet. Cache
Manager can still show their thumbnails, but not the file they came from.

Either way, this is descriptive information only: it does not restore the
original files into workflow inputs, and it plays no part in the cache
fingerprint or HIT/MISS decision for either node.

### Known Cache Manager pairing UI edge case

Dual Resolution pairing is primarily a presentation feature; the two core cache
entries remain independent.

One known UI edge case remains: if filtering/search hides the base side of a
valid Dual Resolution pair while the hidden upscale side still matches the
filter, that pair may temporarily disappear from the filtered list instead of
showing the upscale entry by itself.

This does **not** delete, corrupt, or invalidate either cache entry. Clearing or
changing the filter reveals the pair again.

### Scope is limited to MiniMax H3 conditioning

CLIPCached currently wraps the MiniMax H3 conditioning paths implemented by:

- the FL2VA/Image-to-Video node,
- the Ref2VA/Reference-to-Video node,

plus their Dual Resolution convenience variants.

It does not cache diffusion sampling, DiT execution, VAE work, audio VAE work,
or arbitrary conditioning nodes from other model families.

## Testing & Validation

CLIPCached is validated at two different levels:

1. a fast automated test suite that exercises the cache logic, node contracts,
   storage, invalidation, concurrency, and UI/backend behavior without loading
   the real ~27 GB Qwen3-VL encoder;
2. separate manual diagnostics that run against real ComfyUI, real MiniMax H3
   model files, and a GPU.

These two layers are intentionally kept separate. Passing the automated suite
does not by itself prove real-model memory behavior, while a successful manual
GPU run does not replace deterministic regression coverage for the cache
implementation.

### Automated test suite

The repository runs its automated test suite in GitHub Actions on pushes and
pull requests to `master`. Release validation also includes the relevant Python
syntax/checking steps rather than treating one historical pass count as a
permanent project contract.

The automated tests use small stand-in tensors and controlled test doubles where
appropriate. They do **not** load the real Qwen3-VL MiniMax H3 encoder.

Coverage includes the areas that are most important for cache correctness, such
as:

- deterministic fingerprint construction and invalidation,
- prompt and encoder-checkpoint identity,
- encoder ABI invalidation,
- textual-inversion `embedding:` content identity,
- tensor shape / dtype / byte-level fingerprinting,
- FL2VA and Ref2VA reference ordering and invalidation behavior,
- lazy proxy behavior,
- HIT / MISS / REFRESH decisions,
- output hidden-dimension validation,
- cache serialization and reconstruction,
- schema-v2 generation IDs,
- interrupted/torn writes and corruption handling,
- orphan cleanup,
- per-fingerprint locking and concurrent access,
- ComfyUI `IS_CHANGED` integration,
- FL2VA and Ref2VA node contracts,
- both Dual Resolution node variants,
- `generate_upscale_cond=False`,
- Dual Resolution pairing metadata,
- the CLIP Name helper,
- Cache Manager scanning, metadata, routes, delete/update behavior, and
  `Last Used` state.

The purpose of these tests is to make cache behavior deterministic and
reviewable without requiring a large model load for every development change.

### Real FL2VA equivalence test

The primary real-model correctness diagnostic is:

[`scripts/test_stock_vs_cache.py`](../scripts/test_stock_vs_cache.py)

It uses the real MiniMax H3 text/vision encoder and compares three paths:

```text
(a) stock ComfyUI MiniMax H3
(b) CLIPCached cache MISS
(c) CLIPCached cache HIT
```

The outputs are compared recursively with exact `torch.equal`, not
`torch.allclose`.

The intended correctness condition is:

```text
stock == cached MISS == cached HIT
```

This is deliberately stronger than a numerical-tolerance check. A cache HIT is
supposed to replay the exact stored conditioning bytes, not merely produce a
numerically similar tensor.

The diagnostic also verifies that:

- the MISS path really loads the encoder,
- the HIT path does not load the encoder,
- the generated conditioning and latent remain compatible with the normal
  MiniMax H3 workflow.

It additionally contains a downstream `MiniMaxH3AddGuide` comparison. That
guide step is treated as an extra compatibility check rather than part of the
core cache verdict because it performs additional VAE work and can have a
different memory ceiling from the conditioning test itself.

This FL2VA path has received the most development-time and real-world testing
and should be treated as the project's primary empirically validated workflow.

### Live ComfyUI validation

Standalone Python tests are useful, but they do not reproduce every part of a
running ComfyUI server — particularly ComfyUI's own execution cache and normal
model-management behavior.

Dedicated live-server diagnostics therefore launch a real `python main.py`,
submit workflows through the ComfyUI API, inspect the resulting logs, and shut
the server down afterward.

This matters especially for distinguishing:

```text
ComfyUI execution-cache reuse
```

from:

```text
CLIPCached disk-cache HIT
```

An unchanged graph submitted twice to the same server may be skipped entirely
by ComfyUI before the CLIPCached node executes. To prove a real persistent HIT,
the live-server tests use a fresh ComfyUI process with an already-existing disk
cache entry.

### Ref2VA validation status

Ref2VA has meaningful automated and real-model validation, but it has not
received the same depth of everyday workflow testing as FL2VA.

Its validation includes:

- automated node and fingerprint tests,
- reference-slot compaction and ordering tests,
- image / video / audio cache-invalidation rules,
- Ref2VA Dual Resolution coverage,
- a real-model stock-vs-proxy equivalence diagnostic,
- live-server MISS validation,
- a separate fresh-server HIT validation.

The real equivalence diagnostic:

[`scripts/test_ref2video_equivalence.py`](../scripts/test_ref2video_equivalence.py)

runs stock `MiniMaxH3ReferenceToVideo` and the proxy path with value-identical
inputs containing:

- a reference image,
- a multi-frame reference video,
- the video's soundtrack,
- standalone reference audio.

Its nested outputs are compared field-by-field with exact `torch.equal`.

That test validates that substituting the proxy does not change the stock
Ref2VA result, but it is not itself a disk-cache HIT/MISS test.

Actual cached Ref2VA behavior is also exercised through live ComfyUI server
diagnostics. The MISS test performs a real Qwen3-VL encode and writes an entry.
The HIT follow-up starts a **fresh server**, re-runs the matching workflow,
checks for `[CACHE HIT]`, and verifies from the server log that the MiniMax H3
text encoder was not loaded.

Despite this coverage, the author's note from the
[Node Guide](NODE_GUIDE.md#minimax-h3-clip-cached-ref2va) still applies:
Ref2VA is **less battle-tested in real production-style workflows than FL2VA**.

### Memory and model-lifecycle diagnostics

The repository also contains manual diagnostics for areas that are difficult to
validate meaningfully with tiny unit-test tensors, including:

- Qwen3-VL load/unload behavior,
- VRAM before and after targeted unload,
- system-RAM behavior during large encoder loads,
- VAE and CLIP residency/isolation,
- live-server model loading,
- behavior under constrained-memory conditions.

Examples include scripts such as:

- [scripts/test_clip_unload_isolation.py](../scripts/test_clip_unload_isolation.py)
- [scripts/test_vae_memory_isolation.py](../scripts/test_vae_memory_isolation.py)

These should not all be interpreted as ordinary pass/fail regression tests.
Some are deliberately measurement-oriented diagnostics, include watchdogs or
hard timeouts, or isolate one memory-management hypothesis at a time.

That distinction is important: memory residency depends on the actual ComfyUI,
PyTorch, CUDA, driver, hardware, and launch configuration. The controlled
performance methodology in
[Performance & Benchmarks](PERFORMANCE.md) is therefore the source for
publishable timing/RAM/VRAM numbers, rather than incidental values printed by
development diagnostics.

### What the tests do not claim

The validation above does not mean that every possible ComfyUI installation or
MiniMax H3 configuration is guaranteed to behave identically.

In particular, it does not establish:

- GGUF encoder compatibility,
- safe multi-process sharing of one cache directory,
- unlimited Ref2VA reference counts,
- behavior of arbitrary third-party patched `CLIP` objects,
- performance numbers for hardware that was not actually benchmarked.

Those remain within the limitations described in the previous section.

When reporting a cache bug, use the repository's
[Bug report template](https://github.com/Mu5hr00moO/ComfyUI-MiniMaxH3-CLIPCached/issues/new?template=bug_report.yml).
The most useful distinction is whether the problem occurs on:

```text
MISS
HIT
REFRESH
```

and whether the same effective inputs produce the expected stock MiniMax H3
result. For memory/performance issues, include the environment information
listed in [Performance & Benchmarks](PERFORMANCE.md).
