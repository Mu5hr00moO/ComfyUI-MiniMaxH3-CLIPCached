"""JPEG thumbnails of the first_frame / last_frame images behind a cache entry.

These are a pure UI aid for the Cache Manager -- a visual reminder of which
reference images produced an encode. They are NOT part of the source of
truth (that is store.py's ".json" + ".safetensors" pair) and NOT part of the
fingerprint: a missing, corrupt or stale thumbnail changes nothing about
HIT/MISS or about the conditioning that is served.

Because of that, the thumbnail is deliberately built from the RAW IMAGE
tensor handed to nodes.py's execute() -- the frame *before* the stock node's
internal _resize step -- not from the post-resize image that actually
reaches the encoder. Reconstructing the post-resize image would mean
duplicating stock H3 preprocessing (CLAUDE.md forbids that), and a
memory-jog thumbnail does not need to match the encoder input pixel for
pixel.

Writes are atomic (temp file + os.replace), the same pattern store.py and
verbose_store.py use. _tmp_name() is copied here rather than imported: these
small cache-layer modules stay independent of one another.
"""

import io
import os
import uuid
from pathlib import Path

from PIL import Image

THUMBNAILS_SUBDIR = "thumbnails"


def _tmp_name(path: Path) -> Path:
    # Same convention as store._tmp_name / verbose_store._tmp_name: the PID
    # keeps concurrent processes apart, the uuid4 keeps two writers of the
    # same file within one process off each other's temp path. Copied, not
    # imported -- these modules do not depend on each other.
    return path.with_name("{}.tmp-{}-{}".format(path.name, os.getpid(), uuid.uuid4().hex))


def tensor_to_jpeg_bytes(image_tensor, max_size=256, quality=85) -> bytes:
    """Encode one ComfyUI IMAGE tensor as JPEG bytes.

    image_tensor is (batch, H, W, C), float32 in [0, 1] -- the standard
    ComfyUI IMAGE layout. Only frame [0] of the batch is used: reference
    first_frame / last_frame images are batch=1 in practice, but a larger
    batch is accepted (first frame taken) rather than rejected.

    The image is downscaled so its longer side is <= max_size with aspect
    ratio preserved; an image already within max_size is left untouched
    (Image.thumbnail only shrinks).
    """
    frame = (image_tensor[0].clamp(0, 1) * 255).byte().cpu().numpy()
    channels = frame.shape[-1] if frame.ndim == 3 else 1
    mode = {1: "L", 3: "RGB", 4: "RGBA"}.get(channels)
    if mode is None:
        raise ValueError("unsupported IMAGE tensor with {} channels".format(channels))
    img = Image.fromarray(frame.squeeze(-1) if channels == 1 else frame, mode=mode).convert("RGB")
    img.thumbnail((max_size, max_size))

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def save_thumbnail(image_tensor, fingerprint, index, cache_dir, max_size=256) -> str:
    """Write "<cache_dir>/thumbnails/<fingerprint>_<index>.jpg" atomically.

    Returns the path RELATIVE to cache_dir ("thumbnails/<fp>_<index>.jpg"),
    which is what goes into the verbose sidecar: that JSON may later be read
    from a different working directory, so an absolute path would not
    travel.
    """
    cache_dir = Path(cache_dir)
    thumb_dir = cache_dir / THUMBNAILS_SUBDIR
    thumb_dir.mkdir(parents=True, exist_ok=True)

    filename = "{}_{}.jpg".format(fingerprint, index)
    final_path = thumb_dir / filename

    tmp_path = _tmp_name(final_path)
    try:
        tmp_path.write_bytes(tensor_to_jpeg_bytes(image_tensor, max_size=max_size))
        os.replace(tmp_path, final_path)
    except BaseException:
        # A failed encode/write must not leave a ".tmp-<pid>-<uuid>" file
        # behind in the thumbnails dir. The temp path is unique per call, so
        # unlinking it here can only ever remove our own partial file.
        try:
            tmp_path.unlink()
        except OSError:
            pass
        raise

    # Forward slash, not os.sep: this string is stored in JSON and read back
    # on any platform.
    return "{}/{}".format(THUMBNAILS_SUBDIR, filename)


def thumbnail_paths(fingerprint, cache_dir) -> list[Path]:
    """Every existing "thumbnails/<fingerprint>_*.jpg" of one cache entry.

    Unlike the core/sidecar path helpers this one has to look at the disk --
    the number of thumbnails is not derivable from the fingerprint alone --
    so it lists only files that exist, and returns an empty list when there
    is no thumbnails directory at all. delete_thumbnails() removes exactly
    this list, and the Cache Manager sizes an entry from it, so the two never
    disagree about which files belong to the entry.
    """
    thumb_dir = Path(cache_dir) / THUMBNAILS_SUBDIR
    return sorted(thumb_dir.glob("{}_*.jpg".format(fingerprint)))


def delete_thumbnails(fingerprint, cache_dir) -> None:
    """Remove every "thumbnails/<fingerprint>_*.jpg" for this cache entry.

    Idempotent: nothing to delete (no thumbnails dir, no matching files) is
    not an error, so a Delete can be retried on the same fingerprint.
    """
    for path in thumbnail_paths(fingerprint, cache_dir):
        try:
            path.unlink()
        except FileNotFoundError:
            pass
