# TODO — deferred / backlog items

Not urgent, not forgotten. Each item below was deliberately deferred during
active development rather than solved on the spot -- either genuinely low
priority, or blocked on real-world usage surfacing whether it's worth the
complexity. Pick any of these up when it becomes relevant; delete the entry
once done (this file is not a changelog -- HANDOFF.md and git history are).

## scripts/ -- decouple test_proxy_gate.py's two roles

`scripts/test_proxy_gate.py` is both (a) a standalone historical phase 4-5
verification gate and (b) a shared fixture module: `SpyClipProxy`, the loader
constants (`CLIP_NAME`, `CLIP_TYPE`, `VAE_NAME`, `PROMPT`, `WIDTH`, `HEIGHT`,
`LENGTH`) and `log_memory` are imported from it by four other scripts
(`test_stock_vs_cache.py`, `test_ref2video_equivalence.py`,
`test_clip_unload_isolation.py`, `test_vae_memory_isolation.py` -- only
`test_ref2video_equivalence.py` currently pulls `SpyClipProxy` itself, the
rest take only constants and `log_memory`). Splitting the fixture pieces into
a dedicated `scripts/_fixtures.py` would let `test_proxy_gate.py` itself
become a deletion candidate (its own historical-gate role would be its only
remaining reason to exist), cleanly separating "shared test infrastructure"
from "one-off historical verification." Not done because nothing is broken
today -- the coupling is just a little unusual to read.

## Cache Manager UI -- one known edge case in dual-resolution pairing display

From the pairing UI (commit 576b0c4, "Fold dual-resolution pairs into one
row in the Cache Manager"); rare, does not lose data or crash:

- If a text/tag search filters out the BASE side of a valid pair but the
  UPSCALE side happens to still match (their `user` metadata -- tags,
  name -- is independent per fingerprint, even though the prompt is
  shared), the upscale side still gets `continue`d in `renderList()` (per
  design: valid pairs never render their upscale side standalone), so
  nothing renders for that pair even though `filtered.length > 0`.
  Confusing, not harmful.

## Cache Manager -- optional future feature: explicit paired-delete

Current behavior (deliberate): deleting either side of a pair never cascades
to the other. An orphaned pairing renders as a normal, fully visible row with
a "pairing partner missing" badge -- visible, but requires a manual second
Delete click if you want to free that disk space too. If this proves
annoying in practice, a future option would be an explicit, opt-in
"also delete the paired entry" action in the UI (with the freed size shown
before confirming) -- never a silent/default cascade.

## Ref2VA family -- dynamic reference slots instead of the fixed v1 count

The Ref2VA nodes (`MiniMaxH3CLIPCachedRef2VA` and its Dual Resolution
sibling) expose a fixed count of typed reference sockets (9 images / 3
videos / 3 video soundtracks / 3 standalone audios), whereas the stock
`MiniMaxH3ReferenceToVideo` uses `io.Autogrow.Input` groups that add sockets
on demand and assembles them into the heterogeneous, order-significant
`minimax_ref_items=` list the encoder actually sees. Adapting our nodes to a
dynamic slot mechanism (so a workflow needing more than the v1 cap does not
have to fall back to the stock node) was deliberately deferred early on as a
larger adaptation than the rest of this project's scope; it predates the
dual-resolution / pairing work.

## `cache_mode="cache_only"` -- never load the encoder on a miss

Only `auto` and `refresh` exist today. On a miss both fall through to
loading the full ~27 GB encoder (or raising if that fails). A `cache_only`
mode -- "serve from cache or fail loudly, never load the model" -- is
mentioned as planned in `README.md` ("Limitations") but not implemented.
Useful for batch/headless runs where an unexpected 27 GB load is worse than
a clear error.

## Reference source filenames -- FL2VA's `first_frame` / `last_frame`

The Ref2VA nodes already track where each reference came from on disk:
`minimaxh3_clipcache/provenance.py` walks the API-format prompt graph
backward from every `ref_*` slot to the leaf loader that supplied it and
reads that loader's literal filename, `_sync_ref_sources()` in `nodes.py`
stores the result as `system.ref_sources` in the verbose sidecar, and the
Cache Manager shows it under each reference in the detail panel. The FL2VA
nodes have no equivalent: their `first_frame` / `last_frame` remain
thumbnail-only, so the manager cannot say a keyframe came from
`portrait.png`.

Extending it would not mean a new mechanism. The walk itself is generic --
only its slot-name filter (`_REF_INPUT_PREFIXES`) is Ref2VA-specific. What
is actually missing is the walk's input: `MiniMaxH3CLIPCachedFL2VA` and its
Dual Resolution sibling declare no `hidden` block, so unlike the Ref2VA pair
(`_ref2va_hidden_input_spec()`) they never receive ComfyUI's `PROMPT` /
`UNIQUE_ID` in the first place.

The original objection to graph introspection -- brittle, and reaching
outside the node's own contract -- is no longer what stands in the way: the
Ref2VA implementation went ahead with it under a deliberately narrow scope
(nothing feeds `compute_fingerprint()`, a failed walk cannot disturb a
cached encode, the result is a navigation aid for the Cache Manager UI, not
part of the cache contract). Those limits are stated authoritatively in
`provenance.py`'s module docstring and would have to hold for FL2VA too.
The slots were simply left out when the walk was built for Ref2VA.

Adding that `hidden` block is not free, though. Declaring `UNIQUE_ID` opts
a node's class into ComfyUI's own in-memory execution-cache signature --
folding the node's id into it, so a rebuilt or renumbered graph stops
reusing ComfyUI's RAM-cached output for that node (see the docstring of
`_ref2va_hidden_input_spec()` in `nodes.py:936` for the mechanism). For
Ref2VA this was accepted as an intended, harmless cost: the on-disk cache
is keyed by the encode fingerprint, not by node id, so a rebuilt graph
still HITs the saved encode regardless. The same reasoning would presumably
carry over to FL2VA, but that is an assumption to verify against FL2VA's
own execution path at implementation time, not something to inherit from
the Ref2VA precedent without checking.
