# Technical Details & Cache Compatibility

## Technical Details

This section describes the implementation details behind CLIPCached. They are
not required for normal use, but they explain how cache identity, persistence,
invalidations, concurrency, and interaction with ComfyUI are handled.

### What Is Cached

CLIPCached stores only the result of MiniMax H3's expensive text/vision encode:
the object returned by:

`clip.encode_from_tokens_scheduled()`

That includes the conditioning tensor and the accompanying encoder metadata
needed by the stock MiniMax H3 node.

It does **not** cache the complete output of the FL2VA or Ref2VA node. VAE
encoding, keyframe/reference processing, AV latent construction,
`minimax_keyframes`, `minimax_refs`, and the final node `LATENT` are still
created through ComfyUI's stock MiniMax H3 implementation on every execution.

The integration is deliberately narrow. The cached nodes pass a lazy
`CachedClipProxy` to ComfyUI's stock `MiniMaxH3ImageToVideo` or
`MiniMaxH3ReferenceToVideo` implementation. The stock node performs its normal
preprocessing and calls the proxy using the same `tokenize()` /
`encode_from_tokens_scheduled()` interface it expects from a real CLIP object.

Conceptually:

```text
stock MiniMax H3 node
        |
        v
CachedClipProxy.tokenize(...)
  records the exact encoder request
        |
        v
CachedClipProxy.encode_from_tokens_scheduled(...)
        |
        +---- cache HIT ----> restore conditioning from disk
        |
        +---- cache MISS ---> load real Qwen3-VL
                              tokenize + encode
                              unload Qwen3-VL
                              save conditioning
        |
        v
stock MiniMax H3 node continues normally
(VAE / references / AV latent / outputs)
```

The main implementation lives in
[`minimaxh3_clipcache/proxy.py`](../minimaxh3_clipcache/proxy.py).

### Fingerprint Construction

Every cacheable encode request is identified by a deterministic SHA-256
fingerprint generated in
[`minimaxh3_clipcache/fingerprint.py`](../minimaxh3_clipcache/fingerprint.py).

The fingerprint includes:

- the cache schema version,
- the exact prompt text,
- the exact set of keyword arguments passed to the encoder tokenizer,
- encoder-visible tensor contents, including tensor shape and dtype,
- list/tuple order for ordered inputs such as references,
- the selected encoder checkpoint filename,
- checkpoint file size,
- checkpoint `mtime_ns`,
- checkpoint `ctime_ns`,
- the MiniMax H3 encoder ABI identity,
- resolved textual-inversion embedding tensor contents when the prompt uses
  them.

Variable-sized values are length-framed and type-tagged before hashing. Tensor
identity is based on **shape + dtype + raw bytes**, not Python object identity.
This prevents differently shaped or typed tensors that happen to share the same
byte representation from being treated as equivalent.

Ordered inputs stay ordered. For example, FL2VA `first_frame` and `last_frame`
are not interchangeable, and Ref2VA reference ordering is preserved exactly as
it reaches the tokenizer.

`width`, `height`, and `length` are not simply appended to the fingerprint as
blind numeric fields. They matter when the stock MiniMax H3 preprocessing they
control changes the actual encoder-visible inputs. This is why two different
numeric resolutions can occasionally share a cache entry while another
resolution change correctly creates a MISS.

The checkpoint itself is **not** hashed in full; hashing a ~27 GB model on every
lookup would defeat the purpose of a fast cache check. Instead, checkpoint
identity uses its filename plus filesystem size, modification time, and metadata
change time. Replacing the file under the same name therefore invalidates the
usual stale-cache cases without rereading the complete checkpoint.

### Encoder ABI

The prompt and references are not the only things that determine conditioning.
An upstream ComfyUI change to MiniMax H3 tokenization or encoder preprocessing
can change the result even when every workflow input remains identical.

CLIPCached therefore includes an **encoder ABI identity** in the fingerprint.
It is built from:

1. the installed ComfyUI version, and
2. a SHA-256 hash of the currently imported
   `comfy/text_encoders/minimax.py` source file.

The implementation is in
[`minimaxh3_clipcache/encoder_abi.py`](../minimaxh3_clipcache/encoder_abi.py).

When either component changes, the fingerprint changes and conditioning created
under the previous ABI is not reused as a HIT.

A concrete upstream example is ComfyUI commit
[`924743af083c151296cc16f925aeab113b6484e8`](https://github.com/Comfy-Org/ComfyUI/commit/924743af083c151296cc16f925aeab113b6484e8)
(PR #15808, **“Minimax-H3: Add missing special tokens”**). It changed
`comfy/text_encoders/minimax.py` so MiniMax H3 tokenization recognizes the
model-specific tokens `<d>`, `</d>`, `<|cutoff|>`, `<|lyrics_start|>`,
`<|lyrics_end|>`, `<|caption_start|>`, and `<|caption_end|>`. Because
CLIPCached hashes that exact MiniMax encoder source file as part of the ABI, an
update containing this change produces a different fingerprint instead of
reusing conditioning created under the older tokenizer behavior.

This is intentionally narrower than hashing the entire ComfyUI dependency tree.
Shared files used indirectly by `minimax.py` are not recursively tracked. The
trade-off avoids invalidating the cache for every unrelated ComfyUI source
change while still covering the MiniMax-specific implementation most likely to
change encoder behavior.

If the ABI identity cannot be determined safely, CLIPCached disables cache
**reuse** for that ComfyUI session and performs a real encode instead of risking
a HIT produced by an unverified encoder implementation. A successful encode may
still be written under the explicit unavailable-ABI identity, but it will not
be reused while ABI verification remains unavailable. If ABI detection later
recovers, the real ABI identity produces a different fingerprint, so the
previous unavailable-ABI entry does not suddenly become a HIT.

Cached conditioning is also structurally validated when it is loaded on a
**HIT**. In particular, the expected MiniMax H3 hidden dimension is checked on
the restored result; validation is not limited to newly written MISS entries.

### On-Disk Format

The core cache is pickle-free. Each fingerprint uses two files:

```text
<fingerprint>.safetensors
<fingerprint>.json
```

The `.safetensors` file stores all tensors. The JSON file stores a plain-data
"skeleton" describing the original nested conditioning structure and references
the corresponding tensors by path.

Serialization is implemented in
[`minimaxh3_clipcache/serialize.py`](../minimaxh3_clipcache/serialize.py). Lists,
tuples, dictionaries, scalar JSON values, `None`, and tensors are reconstructed
without using Python pickle.

The serializer rejects reserved internal dictionary keys and dotted dictionary
keys that could make a flattened tensor path ambiguous. These guards are
primarily defensive; stock MiniMax H3 conditioning uses compatible key names.

Cache Manager information is stored separately in optional verbose metadata and
thumbnail files. This layer can contain the prompt, node variant, displayed
resolution, creation time, reference previews, pairing data, custom name,
notes, tags, and favorite state.

Verbose metadata is **not the source of truth for cache reuse**. Losing or
editing user-facing Cache Manager metadata does not alter the core fingerprint
or turn otherwise valid conditioning into a different encode request.

### Schema v2 and Generation IDs

The current core cache schema is **version 2**. The schema version is itself
part of the fingerprint, allowing the on-disk format to evolve without silently
interpreting an older entry as if it used the current rules.

A schema-v2 JSON file contains an envelope with:

```text
generation_id
skeleton
```

Each save generates a fresh UUID4 value encoded as a 32-character lowercase
hexadecimal `generation_id`.

The same ID is written in two places:

- JSON: `generation_id`,
- safetensors metadata: `cache_generation_id`.

A pair is accepted only when both IDs are valid and equal. Missing, malformed,
or mismatched IDs cause a MISS.

The generation ID solves a problem that file-level atomic writes alone cannot:
a cache entry consists of **two separate files**. If a refresh is interrupted
after only one of those files has been replaced, the old and new halves could
otherwise look individually valid while belonging to different saves.

### Atomic Publishing and Corruption Handling

Core cache writes use a two-phase strategy implemented in
[`minimaxh3_clipcache/store.py`](../minimaxh3_clipcache/store.py).

First, both replacement files are written completely to temporary files inside
the cache directory. Only after both temporary writes succeed are they
published with `os.replace()`.

Publishing order is deliberate:

1. `.safetensors`,
2. `.json`.

`os.replace()` is atomic for each individual file on the same filesystem, but
the two-file pair cannot be replaced atomically as one operation. The shared
generation ID described above detects a partially published pair and converts
it into a clean MISS instead of reconstructing mixed conditioning.

If an interrupted fresh write leaves a `.safetensors` file without its JSON
partner, the loader treats it as an orphan and can remove it automatically.
Directory-wide orphan cleanup is also used by Cache Manager maintenance.

Known cache-entry failures — malformed JSON, invalid schema envelopes, missing
files, corrupt or structurally invalid safetensors data, malformed generation
IDs, generation mismatch, or invalid tensor reconstruction — are treated as
cache MISS conditions rather than making a stale or structurally invalid entry
look usable.

The read path is deliberately not a blanket "catch everything and MISS" path.
Only failures that a re-encode can reasonably repair are converted to a MISS.
Filesystem, permission, and resource failures such as `EACCES`, `EIO`, `EMFILE`,
or `ENOSPC` are allowed to propagate instead of silently triggering another
~27 GB encoder load and hiding the real problem.

Safetensors validates its structure but does not provide a checksum for every
tensor payload byte. The generation-ID mechanism protects against torn cache
publishes; it is not a general disk-corruption checksum for tensor contents.

A failure to **write** a newly computed cache entry is handled differently from
a failure to compute conditioning. Once the expensive real encode has completed
successfully, persistence is only an optimization: a cache-write failure is
logged and the valid freshly computed conditioning can still be returned to the
workflow.

### Concurrency and Encoder Lifecycle

CLIPCached uses two different lock scopes inside one ComfyUI process.

#### Per-fingerprint lock

A lock keyed by fingerprint covers the complete:

```text
lookup -> optional encode -> save
```

sequence for that cache entry. Cache Manager update/delete operations use the
same lock when they modify that fingerprint's files.

This prevents two racing MISS requests for the **same** conditioning from both
loading Qwen3-VL and prevents cache-file maintenance from interleaving with an
in-flight save. A second request re-checks the cache after acquiring the lock
and can use the result written by the first.

Different fingerprints use different locks, so unrelated cache reads and file
operations do not need to serialize behind one global cache lock.

#### Global encoder-load lock

A separate process-wide lock protects the expensive real encoder lifecycle
across **all fingerprints**.

Without it, simultaneous MISS requests for two different prompts could each
load a ~27 GB Qwen3-VL instance. CLIPCached instead serializes the real:

```text
load encoder -> tokenize -> encode -> unload encoder
```

portion of MISS/REFRESH work.

The global encoder lock is **not** held for cache HITs and is released before
the resulting conditioning is written to disk. This allows another MISS to
begin loading the encoder once the previous encoder has been successfully
unloaded instead of waiting for unrelated cache I/O.

On a HIT, the real encoder is never loaded.

On a MISS or REFRESH, the encoder is loaded through ComfyUI's normal MiniMax
`CLIPLoader` path, used for the real tokenize/encode call, and explicitly
released with ComfyUI model management before control returns from the cached
encode path. The conditioning hidden dimension is also validated before a new
entry is persisted; an obviously wrong checkpoint is rejected instead of being
stored as a valid MiniMax H3 cache entry.

If the real encode succeeds but the targeted encoder unload fails, the operation
is treated as a hard error rather than silently reporting success while the
large encoder may still be resident.

These locks are Python process-local locks. They coordinate one running ComfyUI
server process; they are not a cross-process filesystem-locking protocol. The
shared-cache implications are called out again under **Cache Compatibility,
Upgrading & Warnings** and **Limitations**.

### Interaction with ComfyUI's Execution Cache

CLIPCached's disk cache and ComfyUI's own execution cache are separate layers.

ComfyUI can decide that a node does not need to execute again at all when its
inputs and change signature still look valid. If that happens, the CLIPCached
node is never entered, so there is no new disk-cache lookup to log as HIT or
MISS.

The cached nodes therefore implement `IS_CHANGED` behavior designed to keep the
two systems coherent.

With `cache_mode="refresh"`, the node returns a fresh non-equal change marker so
ComfyUI is forced to execute the node again even when all visible inputs are
unchanged. The disk layer can then perform the requested REFRESH.

With `cache_mode="auto"`, the change signature also reflects important state
that can change without the visible `clip_name` string changing, including the
checkpoint file identity and encoder ABI. Resolved textual-inversion embedding
identity is folded into this path as well. Replacing a checkpoint or embedding
file under the same filename therefore forces the cached node to execute again,
after which the normal fingerprint logic decides HIT or MISS.

The two cache layers serve different purposes:

- **ComfyUI execution cache** — temporary graph-execution reuse inside the
  running ComfyUI process,
- **CLIPCached disk cache** — persistent MiniMax H3 conditioning reuse that can
  survive ComfyUI restarts and be reused by other workflows.

This distinction is also why controlled benchmarks restart ComfyUI between
measurements: otherwise ComfyUI may skip node execution before CLIPCached's
persistent cache is even consulted.

## Cache Compatibility, Upgrading & Warnings

CLIPCached is deliberately conservative about cache compatibility. A cached
conditioning result is reused only when the project can identify it as belonging
to the same effective encoder request and a compatible MiniMax H3 encoder
implementation.

After an update, a sudden MISS is therefore not automatically a bug. In some
cases it is the intended safety behavior.

### Updating CLIPCached

The cache format has an explicit schema version. The current format uses
**schema v2**.

The schema version is part of the cache fingerprint, so a future schema change
can intentionally make older entries unreachable without trying to reinterpret
them as the new format.

Older entries are not automatically rewritten in place. They may remain in the
cache directory and continue to consume disk space even when the current
version no longer looks them up. In particular, legacy schema-v1 pairs are not
valid schema-v2 HITs; Cache Manager can surface such stale/incompatible pairs as
`inconsistent` rather than treating them as reusable conditioning.

This is safe: delete obsolete entries through Cache Manager or clear the cache
directory if you no longer need them. The next request will simply be rebuilt as
a MISS under the current schema.

Unless the current project documentation explicitly says otherwise, upgrading
CLIPCached does not require manually converting cache files.

### Updating ComfyUI

CLIPCached also fingerprints an **encoder ABI identity** derived from:

- the running ComfyUI version,
- the contents of ComfyUI's `comfy/text_encoders/minimax.py` implementation.

This protects against reusing conditioning produced under a different MiniMax
H3 tokenizer/encoder implementation.

As a result, updating ComfyUI can intentionally invalidate existing cache
entries. A ComfyUI version change is enough to produce a new ABI identity, and
a change to `minimax.py` also changes it even if the selected encoder checkpoint
file itself is unchanged.

The practical effect is usually a one-time MISS for each conditioning setup
under the new environment. Once rebuilt, the new entries can be reused
normally.

If the encoder ABI cannot be determined safely in the running ComfyUI process,
CLIPCached disables HIT reuse for that session and performs real encodes instead
of risking conditioning created under an unverified implementation.

### Replacing or changing the encoder checkpoint

The checkpoint identity includes its filename plus filesystem metadata such as
file size, modification time, and filesystem `ctime_ns`.

Replacing the encoder file under the same filename therefore normally causes a
new fingerprint and a MISS. This is intentional: a cache produced by one
checkpoint should not silently be served for another checkpoint just because
the dropdown text stayed the same.

If the checkpoint file selected by `clip_name` is missing, the node is expected
to execute and report the missing checkpoint from the node itself rather than
silently serving an unrelated old result.

### `cache_mode="refresh"` is not "clear cache"

`refresh` means:

1. ignore a matching cached result for the current request,
2. run the real Qwen3-VL encode,
3. publish the newly computed conditioning for that fingerprint.

It does **not** delete unrelated cache entries.

For a normal FL2VA or Ref2VA node, one refresh means one real encode for the
current conditioning request.

For a Dual Resolution node, `refresh` applies independently to both active
resolution passes. When `generate_upscale_cond=True`, both the base and upscale
passes are deliberately re-encoded, even when they would otherwise resolve to
the same cache fingerprint. This can therefore load and run Qwen3-VL twice.

Use `refresh` when you deliberately want to recompute the current conditioning,
not as a general maintenance or cleanup command.

### `generate_upscale_cond=False`

On a Dual Resolution node, disabling `generate_upscale_cond` skips the entire
second-resolution pass.

The upscale-conditioning output is then:

- `positive_upscale = None`.

This is expected behavior, not a cache failure.

Any downstream branch that requires `positive_upscale` must either be disabled
or be able to accept the missing value. Bypassing only the downstream consumer
does not by itself disable Dual Resolution's second pass; the
`generate_upscale_cond` switch is what controls that work.

### Cache files are disposable, but pairs should not be hand-edited

The disk cache is an optimization, not the source of truth for the workflow.
Deleting a cache entry does not modify the prompt stored in a workflow, the
workflow itself, the model, or generated video. It does remove that cache
entry's own stored prompt copy and Cache Manager metadata. The next matching
request must run Qwen3-VL again.

Each core schema-v2 entry consists of a matching `.json` and `.safetensors`
pair with the same generation ID. Do not manually combine one file from one
entry/write with the other file from another write. A mismatched pair is
rejected as a MISS by design.

Cache Manager is the preferred way to delete individual entries because it can
also clean up the related user-facing metadata and thumbnails.

A partial, interrupted, or recognized-corrupt cache entry is normally treated
as a MISS and can be rebuilt. Some orphan files are also cleaned automatically.
Unexpected runtime failures that are not recognized cache-file read errors are
allowed to surface instead of being silently converted into a costly re-encode.

### Cache write warnings

A successful Qwen3-VL encode is still usable even if writing its cache entry to
disk fails. CLIPCached logs the cache-write failure and returns the completed
conditioning rather than throwing away an expensive encode solely because the
optimization could not be persisted.

The consequence is that the same request may MISS again later.

If cache-write warnings appear repeatedly, check:

- free disk space,
- write permissions for the cache directory,
- filesystem or storage errors,
- whether the cache directory is located on a reliable writable filesystem.

### One ComfyUI process per cache directory

The locking used by CLIPCached is process-local. It protects concurrent work
inside one ComfyUI server process, including races between cache writes and
Cache Manager operations.

It is **not** a cross-process filesystem lock.

Do not run multiple independent ComfyUI processes that write to the same
CLIPCached cache directory at the same time. If multiple servers are needed,
give them separate copies/cache directories.

### CLIP Name and future ComfyUI validation changes

`MiniMax H3 CLIP Name` intentionally outputs a value compatible with the cached
nodes' `clip_name` combo input so one encoder selection can drive several nodes.

This currently works with the ComfyUI validation behavior against which the
project is tested. Because combo/socket validation is part of ComfyUI rather
than a stable CLIPCached API, a significant future ComfyUI frontend or input
validation change may require this helper to be rechecked.

If a future ComfyUI update suddenly reports an incompatible `clip_name`
connection while the same encoder remains selectable directly on the cached
node, check the current CLIPCached documentation or release notes, if available, before modifying the workflow.

### Upgrade checklist

After updating ComfyUI, CLIPCached, or the MiniMax H3 encoder checkpoint:

1. Restart ComfyUI.
2. Run the workflow with `cache_mode="auto"` first.
3. Treat a safety-driven MISS as normal if the schema, encoder ABI, or checkpoint
   identity changed.
4. Confirm that the new entry can HIT on a later run.
5. Use `refresh` only when you intentionally want to recompute that request.
6. Remove old unreachable cache entries later if disk usage matters.

Do not depend on a cache HIT across incompatible versions. The safe fallback is
always to recompute the conditioning through the stock MiniMax H3 encoder path.
