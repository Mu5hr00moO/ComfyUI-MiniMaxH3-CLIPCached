"""Unit tests for minimaxh3_clipcache.thumbnails.

Synthetic torch.rand IMAGE tensors -- no GPU, no ComfyUI. Pillow only.
"""

import io

import pytest
import torch
from PIL import Image

from minimaxh3_clipcache.thumbnails import (
    delete_thumbnails,
    save_thumbnail,
    tensor_to_jpeg_bytes,
)

FINGERPRINT = "a" * 64


def _image(h, w, batch=1):
    return torch.rand(batch, h, w, 3, dtype=torch.float32)


def _open(data):
    img = Image.open(io.BytesIO(data))
    img.load()  # force a full decode so corruption would raise here
    return img


def test_a_landscape_jpeg_valid_bounded_and_ratio_preserved():
    src_w, src_h = 600, 300
    img = _open(tensor_to_jpeg_bytes(_image(src_h, src_w), max_size=256))

    assert img.format == "JPEG"
    assert max(img.size) <= 256
    out_w, out_h = img.size
    assert abs((out_w / out_h) - (src_w / src_h)) < 0.05


def test_a_portrait_ratio_preserved():
    src_w, src_h = 300, 600
    img = _open(tensor_to_jpeg_bytes(_image(src_h, src_w), max_size=256))

    assert max(img.size) <= 256
    out_w, out_h = img.size
    assert abs((out_w / out_h) - (src_w / src_h)) < 0.05


def test_a_square_maps_to_max_size_square():
    img = _open(tensor_to_jpeg_bytes(_image(500, 500), max_size=256))
    assert img.size == (256, 256)


def test_a_image_within_max_size_is_not_upscaled():
    img = _open(tensor_to_jpeg_bytes(_image(100, 120), max_size=256))
    assert img.size == (120, 100)  # PIL size is (width, height)


def test_b_save_thumbnail_writes_file_and_returns_relative_path(tmp_path):
    rel = save_thumbnail(_image(200, 200), FINGERPRINT, 0, tmp_path)

    assert rel == "thumbnails/{}_0.jpg".format(FINGERPRINT)
    written = tmp_path / "thumbnails" / "{}_0.jpg".format(FINGERPRINT)
    assert written.exists()
    Image.open(written).load()  # valid JPEG on disk


def test_b_index_appears_in_filename(tmp_path):
    rel = save_thumbnail(_image(64, 64), FINGERPRINT, 1, tmp_path)
    assert rel == "thumbnails/{}_1.jpg".format(FINGERPRINT)
    assert (tmp_path / "thumbnails" / "{}_1.jpg".format(FINGERPRINT)).exists()


def test_c_no_leftover_tmp_files_after_save(tmp_path):
    save_thumbnail(_image(200, 200), FINGERPRINT, 0, tmp_path)
    save_thumbnail(_image(200, 200), FINGERPRINT, 1, tmp_path)
    assert list((tmp_path / "thumbnails").glob("*.tmp-*")) == []


def test_c_failed_write_leaves_no_tmp_file_behind(tmp_path, monkeypatch):
    """A failed JPEG encode/write must propagate AND clean up the ".tmp-*"
    file it may have already created."""
    import minimaxh3_clipcache.thumbnails as th

    real_write_bytes = th.Path.write_bytes

    def boom(self, data):
        real_write_bytes(self, data)  # temp file now exists on disk...
        raise OSError("simulated disk full")  # ...then the write "fails"

    monkeypatch.setattr(th.Path, "write_bytes", boom)

    with pytest.raises(OSError, match="simulated disk full"):
        save_thumbnail(_image(64, 64), FINGERPRINT, 0, tmp_path)

    assert list((tmp_path / "thumbnails").glob("*.tmp-*")) == []
    assert not (tmp_path / "thumbnails" / "{}_0.jpg".format(FINGERPRINT)).exists()


def test_d_batch_larger_than_one_uses_frame_zero(tmp_path):
    batch = torch.empty(2, 16, 16, 3, dtype=torch.float32)
    batch[0] = 0.0  # frame 0 is black
    batch[1] = 1.0  # frame 1 is white

    img = _open(tensor_to_jpeg_bytes(batch, max_size=64)).convert("RGB")
    raw = img.tobytes()  # flat R,G,B,R,G,B,...
    avg = sum(raw) / len(raw)
    assert avg < 30  # near-black -> frame 0 was used, not frame 1


def test_d_single_channel_grayscale_tensor_becomes_rgb_jpeg():
    gray = torch.rand(1, 48, 64, 1, dtype=torch.float32)
    img = _open(tensor_to_jpeg_bytes(gray, max_size=256))
    assert img.format == "JPEG"
    assert img.convert("RGB").mode == "RGB"
    assert img.size == (64, 48)


def test_d_four_channel_rgba_tensor_becomes_rgb_jpeg():
    rgba = torch.rand(1, 48, 64, 4, dtype=torch.float32)
    img = _open(tensor_to_jpeg_bytes(rgba, max_size=256))
    assert img.format == "JPEG"
    assert img.convert("RGB").mode == "RGB"
    assert img.size == (64, 48)


def test_d_unusual_channel_count_still_rejected():
    two_channel = torch.rand(1, 16, 16, 2, dtype=torch.float32)
    with pytest.raises(ValueError):
        tensor_to_jpeg_bytes(two_channel)


# --- Phase 5: delete_thumbnails ---

FP2 = "b" * 64


def test_e_delete_thumbnails_removes_only_this_fingerprints_files(tmp_path):
    save_thumbnail(_image(32, 32), FINGERPRINT, 0, tmp_path)
    save_thumbnail(_image(32, 32), FINGERPRINT, 1, tmp_path)
    save_thumbnail(_image(32, 32), FP2, 0, tmp_path)

    delete_thumbnails(FINGERPRINT, tmp_path)

    thumbs = tmp_path / "thumbnails"
    assert not (thumbs / "{}_0.jpg".format(FINGERPRINT)).exists()
    assert not (thumbs / "{}_1.jpg".format(FINGERPRINT)).exists()
    assert (thumbs / "{}_0.jpg".format(FP2)).exists()  # untouched


def test_e_delete_thumbnails_is_idempotent(tmp_path):
    save_thumbnail(_image(32, 32), FINGERPRINT, 0, tmp_path)
    delete_thumbnails(FINGERPRINT, tmp_path)
    delete_thumbnails(FINGERPRINT, tmp_path)      # nothing left -> no raise
    delete_thumbnails(FINGERPRINT, tmp_path / "no-such-cache")  # no thumbnails dir -> no raise
