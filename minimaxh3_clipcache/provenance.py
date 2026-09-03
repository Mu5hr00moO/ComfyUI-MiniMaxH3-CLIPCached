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

The leaf rule: only a real loader's literal counts
--------------------------------------------------
A literal input is treated as "the file a reference came from" only when it
sits on a *leaf* node -- one none of whose inputs is a graph link, i.e. a
real loader (``LoadImage``, ``LoadAudio``, ``VHS_LoadVideo``, a custom
one). A media-looking literal on an *intermediate* node -- a text widget
holding ``"note.png"`` on a pass-through, a preview filename -- is ignored
and the walk descends through that node's incoming link instead. This is
deliberately a structural test, not a ``class_type`` allow-list: an
allow-list would have to be kept in sync with every loader in every custom
pack, and would silently stop tracing the ones it had not heard of.

``_MEDIA_EXTENSIONS`` is a secondary narrowing *within* a leaf, not the
primary filter: it keeps a leaf loader of something that is not media (an
upscale model's ``.pth`` name on ``UpscaleModelLoader``) from being read
as the reference source.

Nearest leaf wins; ties are all kept
------------------------------------
The walk is breadth-first and stops at the first depth that yields at least
one leaf media filename. When several loaders fan into the reference at
that same depth (two ``LoadImage`` nodes feeding one ``ImageBatch``), all
of them are recorded, in breadth-first discovery order -- guessing one or
skipping the slot would both be wrong. A loader that is only reachable via
a longer path than another is not reported: "nearest" is by hop count.

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

Return shape and the None / {} distinction
------------------------------------------
``collect_ref_sources()`` returns:

* ``None`` -- the walk could not run at all: no prompt graph, the graph is
  not a dict, no ``unique_id``, our own node is absent from the graph, or
  an unexpected error. The caller must leave any provenance already on the
  sidecar untouched -- an untraceable run says nothing about an earlier
  traceable one.
* ``{}`` -- the walk ran cleanly and found no traceable source for any
  slot (every branch ends at something like ``VAEDecode`` or
  ``EmptyLatentImage``). The caller must drop any stale provenance: this
  fingerprint's references genuinely have no on-disk trail this time.
* ``{slot_name: [ {"annotated": <raw value>[, "path": <abs>]}, ... ], ...}``
  -- one entry per traced loader. ``annotated`` is the reference file's
  value exactly as it appears in the prompt (origin tag and all); ``path``
  is ``folder_paths.get_annotated_filepath(annotated)`` and is omitted when
  that cannot be resolved.

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

# A leaf loader's literal input only counts as "the file a reference came
# from" when its extension names an image / video / audio container. This is
# a secondary narrowing within a leaf (see the module docstring), not the
# primary filter: it keeps a leaf loader of something that is not media --
# an upscale model's ".pth"/".safetensors" name -- from being mistaken for
# the reference source.
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


def _leaf_media_filename(inputs: dict):
    """The first media-looking literal among ``inputs``' values, or ``None``.

    Only meaningful for a leaf node (one with no link input); the caller
    checks that. Iteration order follows the prompt's own key order, so the
    pick is deterministic for a given prompt.
    """
    for candidate in inputs.values():
        if _looks_like_media_file(candidate):
            return candidate.strip()
    return None


def _walk_back_for_media_filenames(prompt: dict, start_node_id: str) -> list:
    """Breadth-first walk backward from ``start_node_id``; return the literal
    media filenames of the *nearest* leaf loaders, as a list (possibly
    empty).

    A leaf is a node none of whose inputs is a graph link. The walk checks a
    whole BFS depth at once: if any node at that depth is a leaf carrying a
    media filename, every such filename at that depth is returned and the
    walk stops. Intermediate nodes (those with a link input) never
    contribute a literal -- only their links are followed. Cycle-safe: every
    node id is enqueued at most once, so a graph with a loop terminates.
    """
    level = deque([start_node_id])
    visited = {start_node_id}
    while level:
        hits = []
        next_level = deque()
        for node_id in level:
            node = prompt.get(node_id)
            if not isinstance(node, dict):
                continue
            inputs = node.get("inputs")
            if not isinstance(inputs, dict):
                continue
            links = [v for v in inputs.values() if _is_link(v)]
            if links:
                # Intermediate node: ignore its literals, follow its links.
                for link in links:
                    if link[0] not in visited:
                        visited.add(link[0])
                        next_level.append(link[0])
                continue
            # Leaf node: its literals are eligible as the reference source.
            name = _leaf_media_filename(inputs)
            if name is not None:
                hits.append(name)
        if hits:
            return hits
        level = next_level
    return []


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


def _entries_for(annotated_names):
    """Turn a list of raw prompt filename values into the list of
    ``{"annotated": ...[, "path": ...]}`` dicts the sidecar stores."""
    entries = []
    for annotated in annotated_names:
        entry = {"annotated": annotated}
        path = _resolve_path(annotated)
        if path is not None:
            entry["path"] = path
        entries.append(entry)
    return entries


def collect_ref_sources(prompt, unique_id):
    """Map each traceable ``ref_*`` slot of ``prompt[unique_id]`` to the
    loader file(s) it came from.

    Returns (see the module docstring for the reasoning):

    * ``None`` when the walk cannot run at all -- ``prompt`` is not a dict,
      ``unique_id`` is ``None``, or our own node / its ``inputs`` block is
      absent from the graph. The caller must leave any existing provenance
      untouched.
    * ``{}`` when the walk ran cleanly but no slot traced back to a loader.
      The caller must drop any stale provenance.
    * ``{slot_name: [ {"annotated": <raw value>[, "path": <abs>]}, ... ]}``
      otherwise, one entry per nearest leaf loader for that slot.

    A slot is left out of the result when it is unconnected, its value is
    not a graph link, or the backward walk finds no leaf media filename.

    Never raises: provenance is best-effort context for the Cache Manager,
    never part of the HIT/MISS decision.
    """
    try:
        if not isinstance(prompt, dict) or unique_id is None:
            return None
        node = prompt.get(str(unique_id))
        if not isinstance(node, dict):
            return None
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            return None

        # From here the walk is possible: a clean run that finds nothing
        # returns {} so the caller can drop stale provenance.
        sources = {}
        for key, value in inputs.items():
            if not key.startswith(_REF_INPUT_PREFIXES):
                continue
            if not _is_link(value):
                continue
            names = _walk_back_for_media_filenames(prompt, value[0])
            if not names:
                continue
            sources[key] = _entries_for(names)
        return sources
    except Exception as e:
        logger.warning(
            "reference-source provenance walk failed (%s); "
            "continuing without it", e,
        )
        return None
