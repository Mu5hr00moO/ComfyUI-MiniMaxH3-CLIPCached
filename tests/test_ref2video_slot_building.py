"""Unit tests for nodes._build_ref_slot_dicts in isolation -- no node, no
proxy, no GPU, no ComfyUI startup. The helper is pure: it maps the four flat
lists of fixed optional Ref2VA slots to the dict-of-named-slots shape the
stock MiniMaxH3ReferenceToVideo.execute() expects.

The exact string keys it produces are load-bearing: the stock node pairs a
soundtrack to its video by matching the trailing "_<index>" of
"ref_video_<i>" against "ref_video_audio_<i>", so these tests assert the
literal key strings, not just value presence.
"""

import importlib.util
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_node_module():
    spec = importlib.util.spec_from_file_location(
        "minimaxh3clipcached_nodes_slotbuild", os.path.join(REPO_ROOT, "nodes.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_build_ref_slot_dicts = _load_node_module()._build_ref_slot_dicts


def _call(images=None, videos=None, video_audios=None, audios=None):
    return _build_ref_slot_dicts(
        images if images is not None else [None] * 9,
        videos if videos is not None else [None] * 3,
        video_audios if video_audios is not None else [None] * 3,
        audios if audios is not None else [None] * 3,
    )


def test_a_all_slots_none_gives_all_none():
    ref_images, ref_videos, ref_video_audios, ref_audios = _call()
    assert ref_images is None
    assert ref_videos is None
    assert ref_video_audios is None
    assert ref_audios is None


def test_b_sparse_image_slots_keep_their_own_index():
    img0, img3 = object(), object()
    images = [None] * 9
    images[0] = img0
    images[3] = img3

    ref_images, ref_videos, ref_video_audios, ref_audios = _call(images=images)

    assert ref_images == {"ref_image_0": img0, "ref_image_3": img3}
    assert list(ref_images.keys()) == ["ref_image_0", "ref_image_3"]
    assert ref_videos is None and ref_video_audios is None and ref_audios is None


def test_c_video_and_its_soundtrack_share_the_trailing_index():
    vid1, aud1 = object(), object()
    videos = [None, vid1, None]
    video_audios = [None, aud1, None]

    ref_images, ref_videos, ref_video_audios, ref_audios = _call(
        videos=videos, video_audios=video_audios)

    # literal key strings -- this is exactly what the stock node's
    # `"ref_video_audio_" + name.rsplit("_", 1)[-1]` pairing depends on
    assert ref_videos == {"ref_video_1": vid1}
    assert ref_video_audios == {"ref_video_audio_1": aud1}
    assert ref_images is None and ref_audios is None


def test_d_video_without_soundtrack_leaves_video_audios_none():
    vid0 = object()
    ref_images, ref_videos, ref_video_audios, ref_audios = _call(videos=[vid0, None, None])

    assert ref_videos == {"ref_video_0": vid0}
    assert ref_video_audios is None  # legal: a reference video with no audio
    assert ref_images is None and ref_audios is None


def test_e_standalone_audio_only():
    aud2 = object()
    ref_images, ref_videos, ref_video_audios, ref_audios = _call(audios=[None, None, aud2])

    assert ref_audios == {"ref_audio_2": aud2}
    assert ref_images is None and ref_videos is None and ref_video_audios is None


def test_f_full_house_all_four_groups_populated():
    img0, vid0, aud0, sa0 = object(), object(), object(), object()
    ref_images, ref_videos, ref_video_audios, ref_audios = _build_ref_slot_dicts(
        [img0] + [None] * 8,
        [vid0, None, None],
        [aud0, None, None],
        [sa0, None, None],
    )
    assert ref_images == {"ref_image_0": img0}
    assert ref_videos == {"ref_video_0": vid0}
    assert ref_video_audios == {"ref_video_audio_0": aud0}
    assert ref_audios == {"ref_audio_0": sa0}
