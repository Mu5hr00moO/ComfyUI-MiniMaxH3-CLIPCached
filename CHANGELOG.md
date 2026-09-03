# Changelog

All notable changes to this project are documented in this file.

The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned

- `cache_mode = "cache_only"`: serve conditioning from the cache or fail
  loudly on a miss, without ever loading the large encoder. Intended for
  batch or headless runs where an unexpected encoder load is worse than a
  clear error.
- Dynamic reference slots for the Ref2VA nodes, matching the stock
  `MiniMaxH3ReferenceToVideo` autogrow inputs, so a workflow that needs
  more references than the fixed slot count does not have to fall back to
  the stock node.

## [1.0.0] - 2026-09-03

First public release.

### Added

- Five nodes: **MiniMax H3 CLIP-Cached FL2VA**, **Ref2VA**,
  **FL2VA (Dual Resolution)**, **Ref2VA (Dual Resolution)**, and the
  **MiniMax H3 CLIP Name** helper.
- On-disk cache of the H3 text/vision conditioning. A cache hit restores
  the conditioning from disk without loading Qwen3-VL; a miss runs the
  real encoder, unloads it to release VRAM, and stores the result.
- Cache Manager web panel: browse, search, tag, and favorite cache
  entries, and delete entries to reclaim disk space.
- Documentation in `docs/` covering the node guide, cache behavior,
  performance, technical details, and testing and limitations.

### Notes before installing

- Requires ComfyUI **v0.30.0 or newer** (native MiniMax H3 nodes);
  developed and validated against **v0.34.2**.
- Uses on-disk cache schema **v2**. There is no migration from earlier
  pre-release caches; unmatched entries are simply re-encoded on first
  use.

[Unreleased]: https://github.com/Mu5hr00moO/ComfyUI-MiniMaxH3-CLIPCached/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/Mu5hr00moO/ComfyUI-MiniMaxH3-CLIPCached/releases/tag/v1.0.0
