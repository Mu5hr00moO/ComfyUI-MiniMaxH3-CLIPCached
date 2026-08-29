"""Identity of the MiniMax H3 encoder implementation currently importable
from ComfyUI, used as an additional fingerprint component (plan audit
point 1) so an upstream tokenizer/preprocessing change (e.g. PR #15808,
"Minimax-H3: Add missing special tokens") invalidates old cache entries
instead of serving a stale HIT computed under different tokenization.

Deliberately narrow scope: comfyui_version (bumped at tagged releases)
plus a SHA-256 of comfy/text_encoders/minimax.py specifically - not the
full qwen3vl.py -> qwen_vl.py -> qwen35.py -> llama.py -> sd1_clip.py
dependency chain that file itself uses. This is a conscious, accepted
residual risk (a change ONLY in one of those shared dependency files,
not touching minimax.py itself, would not be caught) in exchange for not
building a full ComfyUI dependency tracker or invalidating cache on every
unrelated upstream change.

Computed and cached ONCE per process (the file cannot change while
ComfyUI is running) - both success and failure are cached, and a failure
is logged at WARNING only once per session, not once per call.
"""

import hashlib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_cached_abi_id = None
_cached_available = None
_warned = False


def get_encoder_abi_id():
    """Returns (abi_id: str | None, available: bool).

    On success, abi_id is "{comfyui_version}:{sha256_hexdigest}" - full
    64-char hexdigest, not truncated (the cost of the extra characters is
    negligible). On any failure (comfyui_version or
    comfy.text_encoders.minimax missing/moved/unreadable), returns
    (None, False) - callers MUST treat False as "cache is unsafe to use
    for a HIT this session", never fall back to comfyui_version alone.
    """
    global _cached_abi_id, _cached_available, _warned
    if _cached_available is not None:
        return _cached_abi_id, _cached_available

    try:
        from comfyui_version import __version__ as comfyui_version
        import comfy.text_encoders.minimax as minimax_module
        source_bytes = Path(minimax_module.__file__).read_bytes()
        file_hash = hashlib.sha256(source_bytes).hexdigest()
        _cached_abi_id = "{}:{}".format(comfyui_version, file_hash)
        _cached_available = True
    except Exception as e:
        if not _warned:
            logger.warning(
                "[ENCODER ABI UNAVAILABLE] could not determine the MiniMax H3 "
                "encoder ABI identity (%s) - disk caching is disabled for this "
                "session (every run will be a real encode, cache_mode is "
                "ignored) until this is resolved, to avoid ever serving a HIT "
                "computed under a different, unverified tokenizer "
                "implementation", e,
            )
            _warned = True
        _cached_abi_id, _cached_available = None, False
    return _cached_abi_id, _cached_available


def _reset_for_tests():
    """Test-only: clear the process-wide cache so each test starts fresh."""
    global _cached_abi_id, _cached_available, _warned
    _cached_abi_id, _cached_available, _warned = None, None, False
