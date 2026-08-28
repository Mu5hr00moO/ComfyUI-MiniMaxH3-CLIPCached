# ComfyUI-MiniMaxH3-CLIPCached

Drop-in replacements for ComfyUI's stock **MiniMax H3 Image to Video** and
**MiniMax H3 Reference to Video** nodes that disk-cache the text/vision
encode step, so identical prompts skip loading the ~27 GB Qwen3-VL encoder
entirely.

> **Not to be confused with** `ComfyUI-MiniMaxH3-Cache`, `-TeaCache`, or
> `-FirstBlockCache`. Those projects cache/skip *diffusion sampling steps*
> on the DiT model — a different bottleneck entirely. This node caches the
> *text/vision encode* step and never touches the sampler. They are
> complementary, not competing — see [See also](#see-also).

## The problem this solves

MiniMax H3's text/vision encoder (Qwen3-VL, ~27 GB on disk) is expensive to
load and run. If you're iterating on the same prompt — testing seeds,
samplers, or downstream settings — the stock node re-runs the full encode
every single time, even though the conditioning it produces hasn't changed
at all.

This node caches the raw output of the encode step on disk, keyed by exactly
the inputs that determine it (prompt, reference images, encoder checkpoint
identity). On a repeat request, the encoder is never loaded — you go
straight from an empty VRAM to conditioning in a fraction of a second.

## What this is *not*

This is not a reimplementation of MiniMax H3. It calls ComfyUI's own stock
`MiniMaxH3ImageToVideo.execute()` (or `MiniMaxH3ReferenceToVideo.execute()`)
directly and lets it do all the real work (frame resizing, VAE keyframe /
reference encoding, AV latent construction, `minimax_keyframes` /
`minimax_refs`). The only thing these nodes change is what the stock node
sees as its `clip` input: instead of a real, already-loaded CLIP object, it
gets a transparent proxy that checks a disk cache before deciding whether to
load the real encoder at all.

Because of this design, each node's `CONDITIONING`/`LATENT` output is
**bit-for-bit identical** to the matching stock node's output for the same
inputs.
The bit-exact proof lives in [`scripts/test_stock_vs_cache.py`](scripts/test_stock_vs_cache.py):
it loads the real ~27 GB Qwen3-VL encoder and checks
`stock == cached-MISS == cached-HIT` with exact `torch.equal` (not
`torch.allclose`). That script is a **manually run, GPU-required diagnostic**,
not part of the automatic `pytest` suite — the CI tests all use small
stand-in tensors and never load the real encoder. An earlier proxy-equivalence
check ([`scripts/test_proxy_equivalence.py`](scripts/test_proxy_equivalence.py))
used a tolerance (`torch.allclose`, `atol=1e-6`, `rtol=1e-5`) while the
architecture was still being built; the final proof on real hardware uses
exact equality.

Downstream nodes like **MiniMax H3 Add Guide** work completely unchanged,
since they only ever see a standard ComfyUI `CONDITIONING` object and have no
idea it came from a cache.

## Requirements

- ComfyUI with native MiniMax H3 support (`comfy_extras/nodes_minimax_h3.py`
  in ComfyUI core — this has been part of ComfyUI since v0.30.0; developed
  and tested against v0.34.2).
- A MiniMax H3 text/vision encoder checkpoint in `models/text_encoders`
  (e.g. `qwen3vl_32b_minimax_h3_int8_convrot.safetensors`, `.safetensors`
  format — GGUF checkpoints are untested with this node).
- The matching MiniMax H3 VAE in `models/vae`.
- `safetensors` (already a standard ComfyUI dependency — nothing extra to
  install).

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Mu5hr00moO/ComfyUI-MiniMaxH3-CLIPCached
```

Restart ComfyUI. Two new nodes, **"MiniMax H3 CLIP-Cached FL2VA"** and
**"MiniMax H3 CLIP-Cached Ref2VA"**, will appear under
`model/conditioning/minimax/cached` — a deliberately separate category from
the stock nodes' `model/conditioning/minimax`, since these nodes have
different timing behavior (a first run can be much slower than a cached
repeat) and a different input contract.

## The FL2VA node

| Input | Type | Notes |
|---|---|---|
| `clip_name` | dropdown (`models/text_encoders`) | Replaces the stock `clip` input. Loaded lazily — **only on a cache miss**. |
| `vae` | VAE | Same as stock — always required, since keyframe encoding is never cached. |
| `prompt` | string (multiline) | Same as stock. |
| `width` / `height` | int | Same defaults and range as stock (1344×768). |
| `length` | int | Same as stock (default 124 = ~5s at 24 fps, snapped to the model's 17k+5 frame grid). |
| `first_frame` / `last_frame` | image (optional) | Same as stock. |
| `cache_mode` | `auto` / `refresh` | `auto`: reuse a cached encode if one exists for this exact prompt+images+checkpoint, otherwise encode and save it. `refresh`: ignore any existing cache entry, always re-encode, and overwrite it. |

Outputs (`positive` / `latent`) are identical in type and shape to the
stock node's `CONDITIONING` / `LATENT`.

## How the cache works

**What gets cached:** only the raw return value of
`clip.encode_from_tokens_scheduled()` — the conditioning tensor plus its
extras (`pooled_output`, `minimax_token_tags`). Nothing else. The AV
latent, VAE-encoded keyframes, and `minimax_keyframes` metadata are
computed by the stock node on *every* call — hit or miss — because they
depend on `vae`, not `clip`, and were never part of the expensive step
this node targets.

**Cache key:** a SHA-256 fingerprint over:
- a cache schema version (so the on-disk format can evolve without
  silently misinterpreting old entries),
- the encoder checkpoint's identity — filename + file size + modification
  time (not a hash of the full 27 GB file),
- the exact prompt text,
- the exact list of images the stock node's frame-resize step produced
  (i.e. already resized to your requested `width`/`height` — hashing this
  is equivalent to hashing exactly what the encoder would see, without
  reimplementing the resize logic), including each tensor's shape and
  dtype alongside its bytes, and preserving order (`first_frame` and
  `last_frame` are never interchangeable).

**What does *not* affect the cache key:** seed, sampler, steps, scheduler,
or anything else that only affects the sampling stage downstream — those
have no bearing on what the encoder produces.

**On-disk format:** no `pickle`. Each entry is a `<fingerprint>.safetensors`
file (tensors) plus a `<fingerprint>.json` file (everything else — the
structure needed to reconstruct the original object, with tensors replaced
by references into the `.safetensors` file). Both are written atomically
(temp file + `os.replace()`), tensors written before the skeleton, so a
process interrupted mid-write always leaves either nothing or a detectably
incomplete entry — never a plausible-looking but corrupt one. A cache
entry that fails to load for any reason (missing file, corrupted JSON,
mismatched tensor references) is treated as a plain cache miss and logged
as a warning; it never raises and never blocks generation.

## Behavior: hit vs. miss

- **Cache hit:** the encoder is never loaded. `clip_name`'s file is never
  touched beyond the initial `stat()` used to build the fingerprint.
- **Cache miss:** the encoder is loaded through ComfyUI's own `CLIPLoader`
  path (`clip_type=MINIMAX`), so it gets the exact same quantization and
  VRAM-management behavior as the stock node — used once, the result is
  saved to disk, and the encoder is explicitly unloaded
  (`comfy.model_management.unload_model_and_clones`) before the node
  returns. The encoder never lingers in VRAM after a single request
  completes, regardless of whether that request was a hit or a miss.

## The Ref2VA node

**"MiniMax H3 CLIP-Cached Ref2VA"** is the same design applied to ComfyUI's
stock **MiniMax H3 Reference to Video** node: a transparent cached-CLIP
proxy in front of `MiniMaxH3ReferenceToVideo.execute()`, which still does
all the real work (reference resizing, VAE image/video/audio encoding, AV
latent construction, `minimax_refs`). Only the raw output of
`clip.encode_from_tokens_scheduled()` is cached.

| Input | Type | Notes |
|---|---|---|
| `clip_name` | dropdown (`models/text_encoders`) | Replaces the stock `clip` input. Loaded lazily — **only on a cache miss**. |
| `vae` | VAE | Video VAE. Same as stock — always required; reference/keyframe encoding is never cached. |
| `audio_vae` | VAE | Audio VAE, for reference audio and video soundtracks. Not part of the cached state (see below), but still required by the stock node. |
| `prompt` | string (multiline) | Same as stock. Refers to references by ordinal tag — `<Picture 1>`, `<Video 1>`, `<Audio 1>` — counted 1-based per type in presentation order (images, then videos, then standalone audio). |
| `width` / `height` | int | Same defaults and range as stock (1344×768, min 32, step 32). |
| `length` | int | Same as stock (default 124, min 5, max 3600, step 17). |
| `ref_image_size` | `match` / `max` | How reference images are sized before encoding. `match` scales each image down to the generation's pixel area; `max` scales to a 2048px short edge for best identity fidelity (and is several times slower, since reference tokens ride through every sampling step). Matches the stock option exactly. |
| `ref_image_0`–`ref_image_8` | image (optional) | Up to 9 reference images. |
| `ref_video_0`–`ref_video_2` | image (optional) | Up to 3 reference videos, each an `IMAGE` batch of frames (not a `VIDEO`). |
| `ref_video_audio_0`–`ref_video_audio_2` | audio (optional) | Soundtrack for the same-numbered reference video (`ref_video_audio_1` pairs with `ref_video_1`). |
| `ref_audio_0`–`ref_audio_2` | audio (optional) | Up to 3 standalone reference audios. |
| `cache_mode` | `auto` / `refresh` | Same as the FL2VA node. |

These 18 reference slots are **v1-style fixed optional inputs** — one
socket per slot, with the counts (9 / 3 / 3 / 3) mirroring the stock node's
own limits. The stock node uses a dynamic `io.Autogrow` input that adds
sockets on demand; this node does not — the slot count is hard-capped in
v1.

Outputs (`positive` / `latent`) are identical in type and shape to the
stock node's `CONDITIONING` / `LATENT`.

### How caching works for Ref2VA

The cache key is built the same way as for the FL2VA node — a SHA-256
fingerprint over the schema version, the encoder checkpoint identity, the
prompt, and every argument passed into `clip.tokenize()`, with list order
preserved and never sorted. For Ref2VA that last part is the list of
reference items (`minimax_ref_items`) the stock node assembles before
tokenizing. A few consequences are specific to this node:

- **Reference order is semantically load-bearing.** The prompt addresses
  references by position (`<Picture 2>` is the second image in the list,
  not a slot literally named `ref_image_2`), so the fingerprint preserves
  that order exactly — the same mechanism by which `first_frame` and
  `last_frame` are never interchangeable in the FL2VA node.

- **The absolute UI slot index does not matter — only the relative order
  of the connected references does.** Moving your only reference image from
  `ref_image_0` to `ref_image_5` still produces a cache **hit**: the stock
  node compacts the connected slots into a dense list before tokenizing, so
  both cases present exactly one `<Picture 1>`. This is intended, not a
  bug, and is covered by a test (R8).

- **Audio content never enters the cached state.** The raw waveform of a
  reference audio or a video soundtrack is encoded by `audio_vae` into
  `minimax_refs` (recomputed on every call, hit or miss) and is *never*
  passed to the text/vision encoder — the encoder only sees an `<Audio N>`
  marker. So swapping just the audio file, with the same prompt / images /
  videos, does **not** invalidate the cache: the encoder cannot see the
  difference. The *number and position* of audio references does affect the
  cache, though, even with identical content, because that changes the
  `<Audio N>` tags in the token stream.

- **`ref_image_size` affects the cache only indirectly, through pixels.**
  It changes the resized reference image that goes into the encoder, so for
  large images (short edge well above the generation size) `match` and
  `max` produce different fingerprints. For small images that neither mode
  upscales, the resized tensor is bit-identical either way and the
  fingerprints deliberately collide — a safe collision, since the encoder
  input really is identical.

- **`length` can affect the cache key too, through reference videos.** The
  stock node trims each reference video to `length`'s frame count before
  subsampling it to 2 fps for the encoder. For a reference video shorter
  than that cut point, the same frames reach the encoder regardless of
  `length`, so changing it is still a cache **hit**. For a reference video
  longer than the cut point, a different `length` produces a different
  subsampled frame set, so it **misses** — correctly, since the encoder
  really did see different frames, but this can look surprising if you
  expect `length` to only affect the output duration.

### Limitations specific to Ref2VA

- **There is no "safe" reorder operation in the UI.** Because reference
  order changes what the prompt's `<Picture N>` / `<Video N>` / `<Audio N>`
  tags mean, any reordering of connected references is a real semantic
  change and will (correctly) miss the cache.
- **The slot limits are fixed in v1** (9 images, 3 videos, 3 video
  soundtracks, 3 standalone audios), not the stock node's dynamic
  `io.Autogrow`. If you need more references than that, use the stock node.

## Measured performance

Numbers below are from real runs on one machine (RTX-class GPU, 16 GB
VRAM, WSL2) and will vary with your disk speed, OS page cache state, and
whether ComfyUI's DynamicVRAM streaming (`aimdo`) is active on your
hardware — treat these as orders of magnitude, not guarantees:

- Cache **hit**: ~0.0s, encoder never touched.
- Real encode through a live ComfyUI server with DynamicVRAM active:
  ~19–20s per distinct prompt.
- Real encode in a cold, isolated process (no DynamicVRAM, encoder not
  yet resident): well over a minute.

Side-by-side RAM/VRAM monitor captures of a cache miss (the ~27 GB encoder
loading) vs. a cache hit (the load line never appears) are still to be
added here.

<!-- TODO: memory comparison screenshots, see CLAUDE.md R10 prep -->

## A note on memory behavior

This node was extensively stress-tested for VRAM/RAM leaks across repeated
load/unload cycles. The short version: **when ComfyUI is started normally**
(`python main.py` — the way essentially everyone runs it), repeated cache
misses across a live session show a flat memory trend; the targeted
unload after each miss does its job.

Isolated test scripts that load the encoder by bypassing `main.py`
entirely (calling `CLIPLoader` directly, skipping ComfyUI's own
DynamicVRAM initialization) *can* show large, non-representative memory
growth per cycle. That growth is an artifact of skipping ComfyUI's normal
startup path in a synthetic test harness — not a bug in this node's
caching logic, and not something you'll see running ComfyUI the normal
way. The full investigation, including the two incidents that triggered
it, is documented in `CLAUDE.md` for anyone curious or extending this
project.

## Limitations

- **No cache eviction.** Entries accumulate in `cache/` forever; nothing
  in this node deletes old ones automatically. Each entry is small
  (roughly tens of MB, not gigabytes — it's a conditioning tensor, not a
  model), but a long-running install will still need occasional manual
  cleanup of the `cache/` directory. A management UI (browse, name,
  delete, see total size) is planned as a separate follow-up, not part of
  this node.
- **No "cache-only" mode.** On a miss, this node always falls through to
  loading the full encoder (or raises, if that fails) — there's currently
  no way to say "only ever use the cache, never load the 27 GB model." A
  `cache_mode="cache_only"` option is planned but not implemented yet.
- Wraps only the two stock MiniMax H3 conditioning nodes that actually run
  the encoder: `MiniMaxH3ImageToVideo` (the FL2VA node) and
  `MiniMaxH3ReferenceToVideo` (the Ref2VA node). The other stock MiniMax H3
  nodes (`EmptyMiniMaxH3LatentAV`, `MiniMax H3 Add Guide`) never touch the
  encoder, so there is nothing there to cache.
- No built-in prompt library or named-prompt management — this node
  caches by content fingerprint only, with no way to browse, name, or
  manage saved entries from the UI. (Planned as a separate follow-up.)
- Requires ComfyUI's native MiniMax H3 support already present in your
  ComfyUI install — this is a wrapper around it, not a substitute.
- This node loads the encoder checkpoint itself by filename
  (`clip_name`) rather than accepting an already-constructed `CLIP`
  socket. If you rely on a separately loaded/patched CLIP object
  (e.g. from another custom node), this node cannot use it -- it is a
  drop-in replacement for the *node*, not for an arbitrary upstream
  CLIP socket.
- `.safetensors` checkpoints only; GGUF encoders are untested.

## Testing

97 automated tests (pytest), none of which require a GPU or the real
encoder — they all use small stand-in tensors:

- cache-key determinism and invalidation (prompt/image/checkpoint
  changes produce a different key; sampler/seed/scheduler changes do not),
- cache serialization round-trips, including atomic-write, dotted-key
  rejection, and corrupted/partial-entry handling,
- proxy laziness (the real encoder loader is never called on a hit),
- node wiring (`cache_mode` correctly reaches the proxy, node
  registration is correct).

Proxy/stock output equivalence against the **real 27 GB encoder** is not in
this suite — it is proven separately by
[`scripts/test_stock_vs_cache.py`](scripts/test_stock_vs_cache.py) (phase 23),
a manually run script that needs the checkpoint and a GPU and compares
`stock == cached-MISS == cached-HIT` with exact `torch.equal`.

The Ref2VA node went through the same verification process as the FL2VA
node: equivalence against the real encoder (`torch.equal`), an end-to-end
test through a live server, and a short memory-trend check — see `CLAUDE.md`
for the full history.

```bash
pytest
```

See `CLAUDE.md` in this repo for the full engineering history and the
reasoning behind specific design decisions, if you're extending this.

## See also

This node speeds up the text/vision encode step only. If you're also
looking to speed up the diffusion sampling itself (the DiT forward
passes), these are complementary, independent projects worth combining
with this one:

- [`ComfyUI-MiniMaxH3-Cache`](https://github.com/lihaoyun6/ComfyUI-MiniMaxH3-Cache)
- [`ComfyUI-MiniMaxH3-TeaCache`](https://github.com/Icyoung/ComfyUI-MiniMaxH3-TeaCache)
- [`ComfyUI-MiniMaxH3-FirstBlockCache`](https://github.com/duckyshell/ComfyUI-MiniMaxH3-FirstBlockCache)

## Credits

The pattern of disk-caching text-encoder output is inspired by Kijai's
`WanVideoTextEncodeCached` node in `ComfyUI-WanVideoWrapper`. The
non-pickle cache format and targeted-unload approach were informed by
`ComfyUI-H3-Multishot`.
