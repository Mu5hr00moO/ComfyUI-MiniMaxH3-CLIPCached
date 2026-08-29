"""Canonical fingerprinting for MiniMax H3 cached-CLIP requests.

Turns (prompt, the kwargs SpyClipProxy.tokenize() captured, clip identity,
cache schema version) into one deterministic sha256 hex digest, used as the
cache key. Pure function of the data already captured by the proxy -- no
ComfyUI imports, no GPU, no disk I/O (disk I/O is phase 12+).

Encoder identity is (clip_name, file_size, mtime_ns, ctime_ns) from os.stat(),
never a hash of the ~27 GB model file itself (see CLAUDE.md). ctime_ns closes
the common stale-cache hole where a checkpoint is replaced under the same
name while preserving its size and mtime: replacing or rewriting a file
changes its filesystem metadata-change time even when mtime is restored.
"""

import hashlib
import json

import torch

CACHE_SCHEMA_VERSION = 2


def _hash_blob(h, payload):
    """Feed one length-delimited byte string into ``h``.

    Every variable-sized component in the fingerprint stream is framed this
    way. Without the length prefix, adjacent scalar values such as ``1.0`` +
    ``23.0`` and ``1.02`` + ``3.0`` produce the same concatenated bytes.
    """
    h.update(len(payload).to_bytes(8, "big"))
    h.update(payload)


def _hash_tensor(h, tensor):
    t = tensor.detach().cpu().contiguous()
    # Shape/dtype go in before the raw bytes: two tensors with identical byte
    # content but different shape or dtype (e.g. a padded vs unpadded buffer,
    # or float16 vs float32 reinterpreting the same bytes) must not collide.
    metadata = json.dumps(
        {"shape": list(t.shape), "dtype": str(t.dtype)}, sort_keys=True,
    ).encode("utf-8")
    _hash_blob(h, metadata)
    # Hash a flat uint8 byte view instead of going through t.numpy(): numpy
    # has no bfloat16 (or float8) dtype, so t.numpy() raises "unsupported
    # ScalarType" for the dtypes a quantised text encoder can emit. .flatten()
    # first because .view(torch.uint8) refuses a size-changing reinterpret on
    # a 0-dim tensor. The shape/dtype recorded above still disambiguate two
    # tensors that share raw bytes; for float32 this produces byte-for-byte
    # the same raw tensor bytes as the old t.numpy().tobytes() path; the
    # surrounding length framing deliberately changes the overall digest.
    # Pass a buffer view straight to hashlib. ndarray.tobytes() used to make
    # a second full-sized Python bytes allocation (hundreds of MB for a long
    # reference video) on top of any CPU/contiguous tensor copy above.
    byte_array = t.reshape(-1).view(torch.uint8).numpy()
    h.update(byte_array.nbytes.to_bytes(8, "big"))
    h.update(memoryview(byte_array))


def _hash_value(h, value):
    """Recursively feed one kwargs value into the hash, preserving list/tuple order.

    List/tuple order is semantically meaningful (e.g. first_frame vs
    last_frame in the "images" list) and must never be sorted. Dict keys
    inside a value (not the top-level kwargs) are sorted since dict key
    order carries no meaning.
    """
    if isinstance(value, torch.Tensor):
        h.update(b"T")
        _hash_tensor(h, value)
    elif isinstance(value, (list, tuple)):
        h.update(b"L" if isinstance(value, list) else b"U")
        h.update(len(value).to_bytes(8, "big"))
        for item in value:
            _hash_value(h, item)
    elif isinstance(value, dict):
        keys = sorted(value.keys())
        h.update(b"D")
        h.update(len(keys).to_bytes(8, "big"))
        for key in keys:
            _hash_blob(h, key.encode("utf-8"))
            _hash_value(h, value[key])
    else:
        h.update(b"S")
        _hash_blob(h, json.dumps(value, sort_keys=True).encode("utf-8"))


def compute_fingerprint(prompt, tokenize_kwargs, clip_name, clip_file_size, clip_mtime_ns,
                         cache_schema_version=CACHE_SCHEMA_VERSION, *, encoder_abi_id,
                         clip_ctime_ns=None):
    """Deterministic sha256 hex digest identifying one cacheable encode request.

    Two calls with equal arguments always produce the same digest (tensors
    compared by shape/dtype/bytes, not object identity); any semantically
    relevant difference -- prompt, tokenize() kwargs (including list order),
    which kwargs keys are present, clip identity, schema version, or the
    encoder ABI identity, or checkpoint ctime -- changes the digest.

    encoder_abi_id is required and keyword-only (no default) so a caller can
    never silently forget it: it is the identity of the MiniMax H3 encoder
    *implementation* currently importable from ComfyUI
    (minimaxh3_clipcache.encoder_abi.get_encoder_abi_id -- comfyui_version
    plus a hash of comfy/text_encoders/minimax.py), folded in as plan audit
    point 1 so an upstream tokenizer/preprocessing change (e.g. PR #15808)
    invalidates old entries instead of being served as a stale HIT computed
    under different tokenization.

    Design decision: an empty list (e.g. tokenize_kwargs={"images": []}) and
    a missing key (tokenize_kwargs={}) hash DIFFERENTLY. The stock
    MiniMaxH3ImageToVideo node always calls tokenize(prompt, images=images)
    with images=[] when there are no keyframes, so in practice the "images"
    key is always present for that call path; a request with no "images" key
    at all can only come from a different tokenize() call shape (e.g. the
    ref2va path (MiniMaxH3CLIPCachedRef2VA) using minimax_ref_items instead).
    Treating those as different avoids ever conflating two different call
    signatures under one cache key.
    """
    metadata = {
        "cache_schema_version": cache_schema_version,
        "clip_name": clip_name,
        "clip_file_size": clip_file_size,
        "clip_mtime_ns": clip_mtime_ns,
        "clip_ctime_ns": clip_ctime_ns,
        "encoder_abi_id": encoder_abi_id,
        "prompt": prompt,
        "kwargs_keys": sorted(tokenize_kwargs.keys()),
    }
    h = hashlib.sha256()
    h.update(b"M")
    _hash_blob(h, json.dumps(metadata, sort_keys=True).encode("utf-8"))

    for key in sorted(tokenize_kwargs.keys()):
        h.update(b"K")
        _hash_blob(h, key.encode("utf-8"))
        _hash_value(h, tokenize_kwargs[key])

    return h.hexdigest()
