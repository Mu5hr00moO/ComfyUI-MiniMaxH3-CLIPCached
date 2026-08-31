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

## Cache Manager UI -- two known edge cases in dual-resolution pairing display

Both from the pairing UI (commit 576b0c4, "Fold dual-resolution pairs into
one row in the Cache Manager"), both rare, neither loses data or crashes:

1. If a text/tag search filters out the BASE side of a valid pair but the
   UPSCALE side happens to still match (their `user` metadata -- tags,
   name -- is independent per fingerprint, even though the prompt is
   shared), the upscale side still gets `continue`d in `renderList()` (per
   design: valid pairs never render their upscale side standalone), so
   nothing renders for that pair even though `filtered.length > 0`.
   Confusing, not harmful.
2. An entry classified `"inconsistent"` (corrupt/mismatched core files) that
   happens to be the base side of an otherwise-valid pairing renders via
   `buildInconsistentRow()`, which does not know about pairing at all --
   no rescale badge, and the upscale side is still hidden (nothing points
   the user at it).

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

## Reference source filenames -- dedicated loader wrappers

Tracking the source filename of a `first_frame` / `last_frame` / reference
input (so the Cache Manager could show "portrait.png" instead of only a
thumbnail) was considered and deferred -- see the "Rozważone i ODŁOŻONE"
section in `CLAUDE.md` for the full reasoning. The short version: doing it
by graph introspection inside the H3-cached node would be brittle and reach
outside the node's own contract; the correct design is a separate family of
dedicated `LoadImage` / `LoadVideo` / `LoadAudio` wrapper nodes that pass
the filename through explicitly as part of their own output contract. That
is a real but much larger project than a one-line change, not rejected --
just not worth it yet.
