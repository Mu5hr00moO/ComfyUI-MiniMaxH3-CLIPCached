"""Unit tests for minimaxh3_clipcache.provenance.collect_ref_sources.

The prompt graph is a plain dict in ComfyUI's API format
(``{node_id: {"class_type": ..., "inputs": {...}}}``), built inline here --
no ComfyUI startup, no node execution, no model load. folder_paths is the
real module (conftest puts the ComfyUI root on sys.path); the one test that
needs get_annotated_filepath to fail monkeypatches it.
"""

import pytest

from minimaxh3_clipcache.provenance import collect_ref_sources

REF_NODE_ID = "10"


def _ref_node(**ref_inputs):
    """A cached-Ref2VA node dict with the given ref_* inputs wired in."""
    inputs = {"prompt": "a cat", "width": 1344, "height": 768}
    inputs.update(ref_inputs)
    return {"class_type": "MiniMaxH3CLIPCachedRef2VA", "inputs": inputs}


def test_direct_loadimage_is_traced_with_path():
    prompt = {
        "1": {"class_type": "LoadImage", "inputs": {"image": "foto.png", "upload": "image"}},
        REF_NODE_ID: _ref_node(ref_image_0=["1", 0]),
    }

    sources = collect_ref_sources(prompt, REF_NODE_ID)

    assert set(sources) == {"ref_image_0"}
    assert sources["ref_image_0"]["annotated"] == "foto.png"
    assert sources["ref_image_0"]["path"].endswith("/input/foto.png")


def test_chain_through_intermediate_node_is_followed():
    prompt = {
        "1": {"class_type": "LoadImage", "inputs": {"image": "portrait.jpg", "upload": "image"}},
        "2": {"class_type": "ImageResize", "inputs": {"image": ["1", 0], "width": 512, "height": 512}},
        "3": {"class_type": "ImageBlur", "inputs": {"image": ["2", 0], "blur_radius": 3}},
        REF_NODE_ID: _ref_node(ref_image_0=["3", 0]),
    }

    sources = collect_ref_sources(prompt, REF_NODE_ID)

    assert sources["ref_image_0"]["annotated"] == "portrait.jpg"


def test_branch_with_no_loader_yields_no_entry():
    prompt = {
        "1": {"class_type": "EmptyLatentImage", "inputs": {"width": 512, "height": 512, "batch_size": 1}},
        "2": {"class_type": "VAEDecode", "inputs": {"samples": ["1", 0], "vae": ["9", 0]}},
        REF_NODE_ID: _ref_node(ref_image_0=["2", 0]),
    }

    assert collect_ref_sources(prompt, REF_NODE_ID) == {}


def test_slot_numbering_gap_yields_exactly_the_connected_keys():
    prompt = {
        "1": {"class_type": "LoadImage", "inputs": {"image": "zero.png"}},
        "2": {"class_type": "LoadImage", "inputs": {"image": "two.png"}},
        REF_NODE_ID: _ref_node(ref_image_0=["1", 0], ref_image_2=["2", 0]),
    }

    sources = collect_ref_sources(prompt, REF_NODE_ID)

    assert set(sources) == {"ref_image_0", "ref_image_2"}
    assert sources["ref_image_0"]["annotated"] == "zero.png"
    assert sources["ref_image_2"]["annotated"] == "two.png"


def test_cycle_in_graph_terminates_and_still_finds_the_filename():
    # 2 and 3 point at each other; 2 also carries the real loader input.
    prompt = {
        "2": {"class_type": "WeirdMix", "inputs": {"a": ["3", 0], "image": "looped.png"}},
        "3": {"class_type": "WeirdMix", "inputs": {"b": ["2", 0]}},
        REF_NODE_ID: _ref_node(ref_image_0=["3", 0]),
    }

    sources = collect_ref_sources(prompt, REF_NODE_ID)

    assert sources["ref_image_0"]["annotated"] == "looped.png"


def test_pure_cycle_with_no_filename_terminates_with_no_entry():
    prompt = {
        "2": {"class_type": "WeirdMix", "inputs": {"a": ["3", 0]}},
        "3": {"class_type": "WeirdMix", "inputs": {"b": ["2", 0]}},
        REF_NODE_ID: _ref_node(ref_image_0=["3", 0]),
    }

    assert collect_ref_sources(prompt, REF_NODE_ID) == {}


def test_absent_unique_id_in_prompt_yields_empty():
    prompt = {"1": {"class_type": "LoadImage", "inputs": {"image": "foto.png"}}}

    assert collect_ref_sources(prompt, "999") == {}


def test_unique_id_none_yields_empty():
    assert collect_ref_sources({"1": {}}, None) == {}


def test_prompt_not_a_dict_yields_empty():
    assert collect_ref_sources(None, REF_NODE_ID) == {}
    assert collect_ref_sources("not a dict", REF_NODE_ID) == {}


def test_unconnected_and_literal_ref_slots_are_skipped():
    prompt = {
        "1": {"class_type": "LoadImage", "inputs": {"image": "foto.png"}},
        REF_NODE_ID: _ref_node(ref_image_0=["1", 0], ref_image_1="stray literal"),
    }

    sources = collect_ref_sources(prompt, REF_NODE_ID)

    assert set(sources) == {"ref_image_0"}


def test_get_annotated_filepath_failure_keeps_annotated_only(monkeypatch):
    def _boom(*args, **kwargs):
        raise ValueError("invalid file path")

    monkeypatch.setattr("folder_paths.get_annotated_filepath", _boom)

    prompt = {
        "1": {"class_type": "LoadImage", "inputs": {"image": "foto.png"}},
        REF_NODE_ID: _ref_node(ref_image_0=["1", 0]),
    }

    sources = collect_ref_sources(prompt, REF_NODE_ID)

    assert sources["ref_image_0"] == {"annotated": "foto.png"}


def test_input_origin_tag_is_preserved_in_annotated_and_resolved_for_path():
    prompt = {
        "1": {"class_type": "LoadImage", "inputs": {"image": "sub/foto.png [input]"}},
        REF_NODE_ID: _ref_node(ref_image_0=["1", 0]),
    }

    sources = collect_ref_sources(prompt, REF_NODE_ID)

    assert sources["ref_image_0"]["annotated"] == "sub/foto.png [input]"
    assert sources["ref_image_0"]["path"].endswith("/input/sub/foto.png")


def test_video_and_audio_slots_are_traced_like_images():
    prompt = {
        "1": {"class_type": "LoadVideo", "inputs": {"file": "clip.mp4"}},
        "2": {"class_type": "LoadAudio", "inputs": {"audio": "voice.wav"}},
        "3": {"class_type": "LoadAudio", "inputs": {"audio": "music.flac"}},
        REF_NODE_ID: _ref_node(
            ref_video_0=["1", 0],
            ref_video_audio_0=["2", 0],
            ref_audio_0=["3", 0],
        ),
    }

    sources = collect_ref_sources(prompt, REF_NODE_ID)

    assert sources["ref_video_0"]["annotated"] == "clip.mp4"
    assert sources["ref_video_audio_0"]["annotated"] == "voice.wav"
    assert sources["ref_audio_0"]["annotated"] == "music.flac"


def test_non_media_literal_on_the_path_is_not_mistaken_for_a_source():
    # LoadImage feeds the reference; an upscale-model loader with a ".pth"
    # name also sits on the traced path but must be ignored.
    prompt = {
        "1": {"class_type": "LoadImage", "inputs": {"image": "real.png"}},
        "2": {"class_type": "UpscaleModelLoader", "inputs": {"model_name": "4x-UltraSharp.pth"}},
        "3": {"class_type": "ImageUpscaleWithModel",
              "inputs": {"upscale_model": ["2", 0], "image": ["1", 0]}},
        REF_NODE_ID: _ref_node(ref_image_0=["3", 0]),
    }

    sources = collect_ref_sources(prompt, REF_NODE_ID)

    assert sources["ref_image_0"]["annotated"] == "real.png"


def test_nearest_media_filename_wins_over_a_deeper_one():
    prompt = {
        "1": {"class_type": "LoadImage", "inputs": {"image": "deep.png"}},
        "2": {"class_type": "ImagePassThrough", "inputs": {"image": ["1", 0], "note": "shallow.png"}},
        REF_NODE_ID: _ref_node(ref_image_0=["2", 0]),
    }

    sources = collect_ref_sources(prompt, REF_NODE_ID)

    assert sources["ref_image_0"]["annotated"] == "shallow.png"
