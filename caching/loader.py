"""Thin wrappers around ComfyUI's own CLIP-loading path (folder_paths,
comfy.sd.load_clip), used only for what CachedClipProxy needs: identifying
the encoder file on disk for the cache fingerprint, and lazily constructing
the real clip object on a cache MISS. Neither function reimplements
ComfyUI's loading logic -- both call the exact same folder_paths /
comfy.sd.load_clip path that nodes.CLIPLoader.load_clip() uses for
type="minimax" (see CLAUDE.md).
"""

import os

import comfy.sd
import folder_paths


def resolve_clip_stat(clip_name):
    """Return (file_size, mtime_ns) for a text_encoders file, used as the
    encoder-identity part of the cache fingerprint. Pure os.stat() -- never
    loads the file. Raises FileNotFoundError if clip_name isn't a real
    registered text_encoders file, since a cache key must never silently
    identify an encoder that doesn't exist.
    """
    clip_path = folder_paths.get_full_path("text_encoders", clip_name)
    if clip_path is None:
        raise FileNotFoundError(
            "text_encoders file '{}' not found by folder_paths.get_full_path() "
            "-- cannot compute a cache fingerprint for a clip that doesn't exist.".format(clip_name)
        )
    st = os.stat(clip_path)
    return st.st_size, st.st_mtime_ns


def build_clip_loader_fn(clip_name):
    """Return a zero-argument callable that, when called, loads the real
    MiniMax H3 clip exactly the way nodes.CLIPLoader.load_clip() does for
    type="minimax". Building this function does no I/O and touches no GPU --
    the comfy.sd.load_clip() call only happens when the returned function is
    actually invoked, so CachedClipProxy can hold it through a cache HIT
    without ever loading the encoder.
    """
    def _load():
        clip_path = folder_paths.get_full_path_or_raise("text_encoders", clip_name)
        return comfy.sd.load_clip(
            ckpt_paths=[clip_path],
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            clip_type=comfy.sd.CLIPType.MINIMAX,
            model_options={},
        )
    return _load
