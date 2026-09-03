"""Best-effort provenance for the cached Ref2VA nodes' reference inputs:
the on-disk file each ``ref_*`` slot ultimately came from.

Why this module exists
----------------------
The Cache Manager shows a Ref2VA entry's reference thumbnails, but a
thumbnail alone is not enough to find the original file again on disk. When
a reference was produced by a ``LoadImage`` / ``LoadAudio`` / ``LoadVideo``
node -- directly, or through a chain of pass-through nodes -- the
API-format prompt still records that loader's literal filename input.
Walking the prompt graph backward from each ``ref_*`` slot recovers it as a
*trail* the user can follow. Nothing here feeds ``compute_fingerprint()``
and a failure never disturbs the cached encode: this is a navigation aid
for the Cache Manager UI, not part of the cache contract.

Why the result is keyed by SLOT NAME, not by position
-----------------------------------------------------
The verbose sidecar's ``system.references`` list is ordered by the stock
node's own reference assembly (all images, then for each video its
soundtrack immediately before the video, then the standalone audios).
Zipping a positional provenance list against that would misalign the moment
a slot in the middle is left empty or the assembly order shifts. The prompt
input key -- ``ref_image_0``, ``ref_video_2``, ``ref_audio_1`` and so on --
is the one stable identifier both sides already agree on, so the provenance
map is keyed by it and the later UI phase joins on the key. No positional
zipping with the reference list happens anywhere.

Scope
-----
Only the fixed, flat reference slots of ``MiniMaxH3CLIPCachedRef2VA`` and
``MiniMaxH3CLIPCachedRef2VADualRes`` (``ref_image_0..8``, ``ref_video_0..2``,
``ref_video_audio_0..2``, ``ref_audio_0..2``). ``collect_ref_sources()``
never raises.
"""

import logging
import os
from collections import deque

logger = logging.getLogger(__name__)

# The reference input keys on both cached Ref2VA nodes. Kept deliberately as
# a name-prefix check rather than an exact list so a future slot-count bump
# in nodes._ref_slots_input_spec() does not silently stop being traced.
_REF_INPUT_PREFIXES = ("ref_image_", "ref_video_", "ref_video_audio_", "ref_audio_")

# A literal input value only counts as "the file a reference came from" when
# its extension names an image / video / audio container. Restricting to
# media extensions keeps an unrelated literal that happens to sit on the
# traced path -- an upscale model's ".pth"/".safetensors" name, a ".json"
# config -- from being mistaken for the reference source.
_MEDIA_EXTENSIONS = frozenset({
    # images
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tif", ".tiff",
    ".ppm", ".pgm", ".pnm", ".jfif", ".ico", ".heic", ".heif", ".avif",
    ".exr", ".tga", ".dds",
    # video
    ".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg",
    ".wmv", ".flv", ".ts", ".m2ts", ".3gp", ".ogv",
    # audio
    ".wav", ".mp3", ".flac", ".ogg", ".oga", ".opus", ".m4a", ".aac",
    ".wma", ".aif", ".aiff", ".alac",
})

# folder_paths.annotated_filepath() keeps these origin tags on the value
# (with the leading space). Strip one before looking at the extension.
_ANNOTATION_SUFFIXES = (" [output]", " [input]", " [temp]")


def _is_link(value) -> bool:
    """True for a ComfyUI API-format link ``[source_node_id, output_slot]``.

    Mirrors comfy_execution.graph_utils.is_link: a 2-element list whose first
    element is a node-id string and whose second is the numeric output slot.
    """
    return (
        isinstance(value, list)
        and len(value) == 2
        and isinstance(value[0], str)
        and isinstance(value[1], (int, float))
        and not isinstance(value[1], bool)
    )


def _looks_like_media_file(value) -> bool:
    if not isinstance(value, str):
        return False
    stem = value.strip()
    if not stem:
        return False
    for suffix in _ANNOTATION_SUFFIXES:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return os.path.splitext(stem)[1].lower() in _MEDIA_EXTENSIONS


def _walk_back_for_media_filename(prompt: dict, start_node_id: str):
    """Breadth-first walk backward from ``start_node_id``, returning the raw
    value of the nearest literal media-filename input, or ``None``.

    Breadth-first so "nearest" means fewest hops from the reference slot: a
    node's own literal inputs are checked before descending into its incoming
    links. Cycle-safe -- every node id is visited at most once -- so a graph
    with a loop terminates instead of recursing forever.
    """
    queue = deque([start_node_id])
    visited = {start_node_id}
    while queue:
        node = prompt.get(queue.popleft())
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            continue
        for candidate in inputs.values():
            if _looks_like_media_file(candidate):
                return candidate.strip()
        for candidate in inputs.values():
            if _is_link(candidate) and candidate[0] not in visited:
                visited.add(candidate[0])
                queue.append(candidate[0])
    return None


def _resolve_path(annotated: str):
    """``folder_paths.get_annotated_filepath(annotated)`` or ``None``.

    ``None`` -- so the caller drops the ``path`` field and keeps only
    ``annotated`` -- when folder_paths is unavailable or the getter rejects
    the value (e.g. a path-traversal guard trips on a crafted name).
    """
    try:
        import folder_paths
    except Exception:
        return None
    getter = getattr(folder_paths, "get_annotated_filepath", None)
    if getter is None:
        return None
    try:
        return getter(annotated)
    except Exception:
        return None


def collect_ref_sources(prompt, unique_id) -> dict:
    """Map each traceable ``ref_*`` slot of ``prompt[unique_id]`` to the file
    it came from.

    Returns ``{slot_name: {"annotated": <raw prompt value>[, "path": <abs>]}}``.
    ``annotated`` is the reference file's value exactly as it appears in the
    prompt (a stable identifier, origin tag and all); ``path`` is
    ``folder_paths.get_annotated_filepath(annotated)`` and is omitted when
    that cannot be resolved.

    A slot is left out of the result entirely when it is unconnected, its
    value is not a graph link, or the backward walk finds no literal media
    filename (the branch ends at something like ``VAEDecode`` or
    ``EmptyLatentImage``). An absent ``unique_id``, a prompt that is not a
    dict, or any other failure yields ``{}``.

    Never raises: provenance is best-effort context for the Cache Manager,
    never part of the HIT/MISS decision.
    """
    try:
        if not isinstance(prompt, dict) or unique_id is None:
            return {}
        node = prompt.get(str(unique_id))
        if not isinstance(node, dict):
            return {}
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            return {}

        sources = {}
        for key, value in inputs.items():
            if not key.startswith(_REF_INPUT_PREFIXES):
                continue
            if not _is_link(value):
                continue
            annotated = _walk_back_for_media_filename(prompt, value[0])
            if annotated is None:
                continue
            entry = {"annotated": annotated}
            path = _resolve_path(annotated)
            if path is not None:
                entry["path"] = path
            sources[key] = entry
        return sources
    except Exception as e:
        logger.warning(
            "reference-source provenance walk failed (%s); "
            "continuing without it", e,
        )
        return {}
