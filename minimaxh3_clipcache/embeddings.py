"""Resolve the textual-inversion embeddings a prompt references, using the
*real* stock MiniMax H3 tokenizer -- no reimplementation of the parsing.

Why this exists: CachedClipProxy.tokenize() only remembers (prompt, kwargs)
and compute_fingerprint() hashes that string as-is. The `embedding:<name>`
syntax is resolved to a file on disk only later, inside the real CLIP, on a
cache MISS -- so a textual-inversion file swapped under an unchanged name,
with the prompt and checkpoint untouched, could otherwise be served as a
stale HIT (Codex audit MEDIUM #1). Folding the *resolved embedding tensors*
into the fingerprint (and into nodes._is_changed_common) closes that hole.

The resolution runs through comfy.text_encoders.minimax.MiniMaxH3Tokenizer
-> its `qwen3vl_32b` comfy.sd1_clip.SDTokenizer, exactly the object the real
CLIP builds (for CLIPType.MINIMAX comfy.sd passes tokenizer_data={}), so the
`embedding:` split (comfy/sd1_clip.py), the name cleanup
(SDTokenizer._try_get_embedding) and the file lookup (comfy.sd1_clip.load_embed
-- directory walk, extension probing, path-traversal guard) are all the
stock ones. Building it loads only the bundled HF tokenizer (~4 MB), never
the ~27 GB encoder.

This is an auxiliary identity layer: like nodes._sync_verbose_metadata it
must never break a normal cache HIT/MISS. Every failure path returns [] and
logs at most once per session.
"""

import logging

import torch

logger = logging.getLogger(__name__)

_cached_tokenizer = None
_cached_available = None
_warned_build = False
_warned_resolve = False


def _build_minimax_tokenizer():
    """Construct the stock MiniMax H3 sub-tokenizer. May raise -- the caller
    (_get_minimax_tokenizer) turns any failure into a cached ``None``.

    embedding_directory is folder_paths.get_folder_paths("embeddings"), the
    same value minimaxh3_clipcache.loader.build_clip_loader_fn already hands
    to comfy.sd.load_clip, so name resolution matches the real CLIP.
    """
    import folder_paths
    import comfy.text_encoders.minimax as minimax

    tokenizer = minimax.MiniMaxH3Tokenizer(
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
        tokenizer_data={},
    )
    return tokenizer.qwen3vl_32b


def _get_minimax_tokenizer():
    """Return the process-wide stock sub-tokenizer, or ``None`` if it could
    not be built. Success and failure are both cached once per process (the
    bundled vocab/merges cannot change at runtime), and a build failure is
    logged at WARNING exactly once -- the same pattern as
    minimaxh3_clipcache.encoder_abi.get_encoder_abi_id().
    """
    global _cached_tokenizer, _cached_available, _warned_build
    if _cached_available is not None:
        return _cached_tokenizer

    try:
        _cached_tokenizer = _build_minimax_tokenizer()
        _cached_available = True
    except Exception as e:
        if not _warned_build:
            logger.warning(
                "[EMBEDDING TOKENIZER UNAVAILABLE] could not build the stock "
                "MiniMax H3 tokenizer to resolve embedding: references (%s) - "
                "prompts that reference a textual inversion will be fingerprinted "
                "from the prompt string alone this session; a swapped embedding "
                "file under an unchanged name may not invalidate the cache", e,
            )
            _warned_build = True
        _cached_tokenizer, _cached_available = None, False
    return _cached_tokenizer


def resolve_prompt_embedding_tensors(prompt):
    """Return the list of textual-inversion embedding tensors the stock
    MiniMax H3 tokenizer resolves from ``prompt``, in order of appearance.

    Returns ``[]`` when the prompt references no embedding, when a referenced
    file does not exist (stock logs ``warning, embedding:X does not exist,
    ignoring`` and resolves nothing -- the real encode then behaves as if the
    reference were absent, so the fingerprint must too), when ``prompt`` is
    not a string, or when the tokenizer could not be built. Never raises.

    A multi-vector textual inversion (shape ``(N, 5120)``) yields ``N``
    separate ``(5120,)`` tensors, matching how comfy.sd1_clip expands it into
    the token stream.
    """
    if not isinstance(prompt, str):
        return []
    subtokenizer = _get_minimax_tokenizer()
    if subtokenizer is None:
        return []
    try:
        batches = subtokenizer.tokenize_with_weights(
            prompt, return_word_ids=False, disable_weights=True,
        )
    except Exception as e:
        global _warned_resolve
        if not _warned_resolve:
            logger.warning(
                "[EMBEDDING RESOLVE FAILED] the stock tokenizer raised while "
                "resolving embedding: references from a prompt (%s) - "
                "fingerprinting from the prompt string alone", e,
            )
            _warned_resolve = True
        return []

    tensors = []
    for batch in batches:
        for entry in batch:
            token = entry[0]
            if isinstance(token, torch.Tensor):
                tensors.append(token)
    return tensors


def embedding_identity_digest(prompt):
    """SHA-256 hex digest of the embedding tensors ``prompt`` resolves to, or
    ``None`` when it resolves none. Convenience wrapper over
    resolve_prompt_embedding_tensors() + fingerprint.hash_embedding_tensors()
    for nodes._is_changed_common(), so IS_CHANGED folds in the exact same
    tensor list that compute_fingerprint() folds into the disk cache key.
    Returning ``None`` (not a digest of nothing) keeps IS_CHANGED byte-for-byte
    unchanged for every prompt without a resolvable embedding.
    """
    from minimaxh3_clipcache.fingerprint import hash_embedding_tensors

    return hash_embedding_tensors(resolve_prompt_embedding_tensors(prompt))


def _reset_for_tests():
    """Test-only: clear the process-wide tokenizer cache so each test starts
    fresh (mirrors minimaxh3_clipcache.encoder_abi._reset_for_tests)."""
    global _cached_tokenizer, _cached_available, _warned_build, _warned_resolve
    _cached_tokenizer = None
    _cached_available = None
    _warned_build = False
    _warned_resolve = False
