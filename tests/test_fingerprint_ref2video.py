"""Stress-test compute_fingerprint() against the exact shape a real
minimax_ref_items list has (confirmed by the R1 read of the stock
MiniMaxH3ReferenceToVideo node): a flat, ordered list of heterogeneous
dicts --

    {"type": "image", "data": <tensor>}
    {"type": "video", "data": <tensor>, "timestamps": [float, ...]}
    {"type": "audio"}                     # no "data" key at all

The audio entry carries no tensor -- it only exists so the tokenizer emits
an "<Audio j>" label -- but its presence and position still change the
prompt the encoder sees, so it must still change the fingerprint.

Pure torch.rand/torch.zeros stand-ins -- no GPU, no ComfyUI, no model load,
no disk I/O.
"""

import pytest
import torch

from minimaxh3_clipcache.fingerprint import compute_fingerprint

CLIP_NAME = "qwen3vl_32b_minimax_h3_int8_convrot.safetensors"
FILE_SIZE = 27141342152
MTIME_NS = 1712345678000000000
CTIME_NS = 1712345678999999999


ABI_ID = "test-abi-id"


def _fp(ref_items=None, kwargs=None, prompt="a ref2va prompt",
        clip_name=CLIP_NAME, file_size=FILE_SIZE, mtime_ns=MTIME_NS,
        ctime_ns=CTIME_NS, encoder_abi_id=ABI_ID):
    if kwargs is None:
        kwargs = {"minimax_ref_items": ref_items}
    return compute_fingerprint(prompt, kwargs, clip_name, file_size, mtime_ns, 1,
                               encoder_abi_id=encoder_abi_id, clip_ctime_ns=ctime_ns)


def _image_item(seed):
    g = torch.Generator().manual_seed(seed)
    return {"type": "image", "data": torch.rand(1, 64, 96, 3, generator=g)}


def _video_item(seed, timestamps=(0.0, 0.5, 1.0)):
    g = torch.Generator().manual_seed(seed)
    return {
        "type": "video",
        "data": torch.rand(3, 48, 48, 3, generator=g),
        "timestamps": list(timestamps),
    }


def _audio_item():
    return {"type": "audio"}


def _clone_items(items):
    out = []
    for it in items:
        c = dict(it)
        if "data" in c:
            c["data"] = c["data"].clone()
        if "timestamps" in c:
            c["timestamps"] = list(c["timestamps"])
        out.append(c)
    return out


# --- a) determinism ---------------------------------------------------------

def test_a_identical_ref_items_same_object_same_fingerprint():
    items = [_image_item(1), _video_item(2), _audio_item()]
    assert _fp(items) == _fp(items)


def test_a_identical_ref_items_cloned_tensors_same_fingerprint():
    items = [_image_item(1), _video_item(2), _audio_item()]
    assert _fp(items) == _fp(_clone_items(items))


# --- b) adding a second image ---------------------------------------------

def test_b_adding_second_image_changes_fingerprint():
    one = [_image_item(1)]
    two = [_image_item(1), _image_item(9)]
    assert _fp(one) != _fp(two)


# --- c) same two images, reversed order ----------------------------------

def test_c_reversed_image_order_changes_fingerprint():
    a, b = _image_item(1), _image_item(2)
    assert _fp([a, b]) != _fp([b, a])


# --- d) lone audio marker vs empty list ---------------------------------

def test_d_lone_audio_marker_differs_from_empty_list():
    assert _fp([_audio_item()]) != _fp([])


# --- e) audio/video ordering relative to each other --------------------

def test_e_audio_before_vs_after_video_changes_fingerprint():
    vid = _video_item(5)
    audio_first = [_audio_item(), _clone_items([vid])[0]]
    video_first = [_clone_items([vid])[0], _audio_item()]
    assert _fp(audio_first) != _fp(video_first)


# --- f) same video tensor, different timestamps ----------------------

def test_f_changing_only_timestamps_changes_fingerprint():
    base = _video_item(7, timestamps=(0.0, 0.5, 1.0))
    shifted = _clone_items([base])[0]
    shifted["timestamps"] = [0.0, 0.6, 1.0]
    assert _fp([base]) != _fp([shifted])


# --- g) minimax_ref_items vs images: different kwargs keys never collide -

def test_g_ref_items_vs_images_key_never_collide():
    g = torch.Generator().manual_seed(42)
    img = torch.rand(1, 64, 64, 3, generator=g)
    fp_ref = _fp(kwargs={"minimax_ref_items": [{"type": "image", "data": img}]})
    fp_img = _fp(kwargs={"images": [img.clone()]})
    assert fp_ref != fp_img


# --- h) control: real ref_items keys never trip the dotted-key guard ----

def test_h_ref_item_keys_do_not_raise_value_error():
    # The dotted-key ValueError lives in minimaxh3_clipcache.serialize
    # (the storage path), not in fingerprint.py -- "type" / "data" /
    # "timestamps" are all dot-free anyway. This guards against a future
    # change accidentally applying that validation on the fingerprint path.
    items = [_image_item(1), _video_item(2), _audio_item()]
    try:
        fp = _fp(items)
    except ValueError as e:  # pragma: no cover - only if a regression appears
        pytest.fail("compute_fingerprint raised ValueError on valid ref_items: {}".format(e))
    assert isinstance(fp, str) and len(fp) == 64
    int(fp, 16)


# --- i) change one image of two, same position -------------------------

def test_i_changing_one_image_of_two_leaves_the_other_changes_fingerprint():
    # Ref2VA analog of test_fingerprint.py's test_j/test_k: with two images
    # at fixed positions in the list, changing only index 0 must move the
    # digest even though index 1 is byte-identical. test_b (add) and test_c
    # (reorder) never exercised a same-length, same-position content swap.
    keep = _image_item(2)
    before = [_image_item(1), keep]
    after = [_image_item(7), _clone_items([keep])[0]]
    assert _fp(before) != _fp(after)


# --- j) same video, same timestamps, different frame pixels ------------

def test_j_changing_only_video_frames_changes_fingerprint():
    # Mirror of test_f (which changes only the timestamps): here the
    # timestamps and the item type are identical and only the frame tensor
    # differs, so the pixel content of a reference video is proven to matter
    # on its own.
    base = _video_item(7, timestamps=(0.0, 0.5, 1.0))
    other = _clone_items([base])[0]
    other["data"] = torch.rand(3, 48, 48, 3, generator=torch.Generator().manual_seed(31))
    assert _fp([base]) != _fp([other])


# --- k) add / remove a reference video --------------------------------

def test_k_adding_or_removing_a_reference_video_changes_fingerprint():
    # test_b covers add/remove for images; a video item carries both a
    # tensor and a timestamps list, so confirm its presence moves the digest
    # too. The assertion is symmetric: left->right reads as "add a video",
    # right->left as "remove a video".
    img = _image_item(1)
    assert _fp([img]) != _fp([img, _video_item(2)])


# --- l) clip identity on the minimax_ref_items path -------------------

def test_l_clip_identity_changes_fingerprint_on_the_ref_items_path():
    # test_fingerprint.py proves clip identity (name / file_size / mtime_ns)
    # drives the digest, but only on the "images" kwarg path. Clip identity
    # is hashed in the metadata block ahead of any kwargs, independent of the
    # kwargs shape, so it must behave identically for a request that carries
    # minimax_ref_items.
    items = [_image_item(1), _video_item(2), _audio_item()]
    base = _fp(items)
    assert _fp(items, clip_name="a_different_encoder.safetensors") != base
    assert _fp(items, file_size=FILE_SIZE + 1) != base
    assert _fp(items, mtime_ns=MTIME_NS + 1) != base
    assert _fp(items, ctime_ns=CTIME_NS + 1) != base


# --- m) control: seed / sampler / steps / scheduler cannot reach the key -

def test_m_compute_fingerprint_has_no_sampling_parameters():
    # seed / sampler / steps / scheduler are inputs to KSampler, downstream
    # of the CONDITIONING output -- neither node threads them into the proxy,
    # and compute_fingerprint() has no parameter to receive them. This is the
    # structural reason changing them is always a HIT (same as FL2VA). Guard
    # the signature so a future refactor can't quietly add one. encoder_abi_id
    # (plan audit point 1) is a legitimate, deliberate addition -- an encoder
    # identity component, not a sampling parameter.
    import inspect

    params = list(inspect.signature(compute_fingerprint).parameters)
    assert params == [
        "prompt", "tokenize_kwargs", "clip_name", "clip_file_size",
        "clip_mtime_ns", "cache_schema_version", "encoder_abi_id", "clip_ctime_ns",
    ]
