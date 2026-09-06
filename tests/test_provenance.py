"""Unit tests for minimaxh3_clipcache.provenance.collect_ref_sources.

The prompt graph is a plain dict in ComfyUI's API format
(``{node_id: {"class_type": ..., "inputs": {...}}}``), built inline here --
no ComfyUI startup, no node execution, no model load. folder_paths is the
real module (conftest puts the ComfyUI root on sys.path); the one test that
needs get_annotated_filepath to fail monkeypatches it.

Contract exercised here (see the module docstring for the rationale):

* the value for every traced slot is a *list* of ``{annotated[, path]}``
  entries -- one element for the common single-loader case, more when the
  reference fans in from several loaders anywhere in its backward subgraph;
* a literal only counts as a source on a *leaf* node (one with no incoming
  link) -- a media-looking literal sitting on an intermediate pass-through
  node is ignored and the walk descends through its link instead;
* ``collect_ref_sources`` returns ``None`` when the walk cannot run at all
  (no graph, graph not a dict, no unique_id, our node absent) and ``{}``
  when it runs cleanly but finds nothing traceable.
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
    assert sources["ref_image_0"][0]["annotated"] == "foto.png"
    assert sources["ref_image_0"][0]["path"].endswith("/input/foto.png")
    assert len(sources["ref_image_0"]) == 1


def test_chain_through_intermediate_node_is_followed():
    prompt = {
        "1": {"class_type": "LoadImage", "inputs": {"image": "portrait.jpg", "upload": "image"}},
        "2": {"class_type": "ImageResize", "inputs": {"image": ["1", 0], "width": 512, "height": 512}},
        "3": {"class_type": "ImageBlur", "inputs": {"image": ["2", 0], "blur_radius": 3}},
        REF_NODE_ID: _ref_node(ref_image_0=["3", 0]),
    }

    sources = collect_ref_sources(prompt, REF_NODE_ID)

    assert sources["ref_image_0"] == [
        {"annotated": "portrait.jpg", "path": sources["ref_image_0"][0]["path"]}
    ]


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
    assert [e["annotated"] for e in sources["ref_image_0"]] == ["zero.png"]
    assert [e["annotated"] for e in sources["ref_image_2"]] == ["two.png"]


def test_cycle_in_graph_terminates_and_still_finds_the_leaf_loader():
    # 2 and 3 point at each other; a real leaf loader (1) also feeds 2.
    prompt = {
        "1": {"class_type": "LoadImage", "inputs": {"image": "looped.png"}},
        "2": {"class_type": "WeirdMix", "inputs": {"a": ["3", 0], "src": ["1", 0]}},
        "3": {"class_type": "WeirdMix", "inputs": {"b": ["2", 0]}},
        REF_NODE_ID: _ref_node(ref_image_0=["3", 0]),
    }

    sources = collect_ref_sources(prompt, REF_NODE_ID)

    assert [e["annotated"] for e in sources["ref_image_0"]] == ["looped.png"]


def test_pure_cycle_with_no_loader_terminates_with_no_entry():
    prompt = {
        "2": {"class_type": "WeirdMix", "inputs": {"a": ["3", 0]}},
        "3": {"class_type": "WeirdMix", "inputs": {"b": ["2", 0]}},
        REF_NODE_ID: _ref_node(ref_image_0=["3", 0]),
    }

    assert collect_ref_sources(prompt, REF_NODE_ID) == {}


def test_absent_unique_id_in_prompt_yields_none():
    """Our own node is not in the graph: the walk cannot run, so the result
    is None (leave any existing provenance alone), not {} (drop it)."""
    prompt = {"1": {"class_type": "LoadImage", "inputs": {"image": "foto.png"}}}

    assert collect_ref_sources(prompt, "999") is None


def test_unique_id_none_yields_none():
    assert collect_ref_sources({"1": {}}, None) is None


def test_prompt_not_a_dict_yields_none():
    assert collect_ref_sources(None, REF_NODE_ID) is None
    assert collect_ref_sources("not a dict", REF_NODE_ID) is None


def test_node_present_but_inputs_missing_yields_none():
    prompt = {REF_NODE_ID: {"class_type": "MiniMaxH3CLIPCachedRef2VA"}}

    assert collect_ref_sources(prompt, REF_NODE_ID) is None


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

    assert sources["ref_image_0"] == [{"annotated": "foto.png"}]


def test_input_origin_tag_is_preserved_in_annotated_and_resolved_for_path():
    prompt = {
        "1": {"class_type": "LoadImage", "inputs": {"image": "sub/foto.png [input]"}},
        REF_NODE_ID: _ref_node(ref_image_0=["1", 0]),
    }

    sources = collect_ref_sources(prompt, REF_NODE_ID)

    assert sources["ref_image_0"][0]["annotated"] == "sub/foto.png [input]"
    assert sources["ref_image_0"][0]["path"].endswith("/input/sub/foto.png")


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

    assert [e["annotated"] for e in sources["ref_video_0"]] == ["clip.mp4"]
    assert [e["annotated"] for e in sources["ref_video_audio_0"]] == ["voice.wav"]
    assert [e["annotated"] for e in sources["ref_audio_0"]] == ["music.flac"]


def test_non_media_literal_on_a_leaf_loader_is_not_mistaken_for_a_source():
    # LoadImage feeds the reference; an upscale-model loader with a ".pth"
    # name is a leaf too, but its literal is not a media container.
    prompt = {
        "1": {"class_type": "LoadImage", "inputs": {"image": "real.png"}},
        "2": {"class_type": "UpscaleModelLoader", "inputs": {"model_name": "4x-UltraSharp.pth"}},
        "3": {"class_type": "ImageUpscaleWithModel",
              "inputs": {"upscale_model": ["2", 0], "image": ["1", 0]}},
        REF_NODE_ID: _ref_node(ref_image_0=["3", 0]),
    }

    sources = collect_ref_sources(prompt, REF_NODE_ID)

    assert [e["annotated"] for e in sources["ref_image_0"]] == ["real.png"]


def test_leaf_loader_wins_over_a_media_literal_on_an_intermediate_node():
    # A pass-through node carries a text widget that happens to hold a
    # ".png" string. Because that node also has a real incoming link, it is
    # NOT a leaf: its literal is ignored and the walk descends to the true
    # loader.
    prompt = {
        "1": {"class_type": "LoadImage", "inputs": {"image": "deep.png"}},
        "2": {"class_type": "ImagePassThrough", "inputs": {"image": ["1", 0], "note": "shallow.png"}},
        REF_NODE_ID: _ref_node(ref_image_0=["2", 0]),
    }

    sources = collect_ref_sources(prompt, REF_NODE_ID)

    assert [e["annotated"] for e in sources["ref_image_0"]] == ["deep.png"]


def test_several_leaf_loaders_at_the_same_depth_are_all_recorded():
    # Two LoadImage nodes fan into one ImageBatch that feeds the reference.
    # Neither is "the" source -- both are recorded, in breadth-first order.
    prompt = {
        "1": {"class_type": "LoadImage", "inputs": {"image": "a.png"}},
        "2": {"class_type": "LoadImage", "inputs": {"image": "b.png"}},
        "3": {"class_type": "ImageBatch", "inputs": {"image1": ["1", 0], "image2": ["2", 0]}},
        REF_NODE_ID: _ref_node(ref_image_0=["3", 0]),
    }

    sources = collect_ref_sources(prompt, REF_NODE_ID)

    assert [e["annotated"] for e in sources["ref_image_0"]] == ["a.png", "b.png"]
    assert all(e["path"].endswith(("/input/a.png", "/input/b.png"))
               for e in sources["ref_image_0"])


def test_asymmetric_fan_in_collects_a_leaf_from_every_depth():
    # ImageBatch pulls one input straight off a LoadImage and the other off
    # a LoadImage through an ImageScale. Both files are real sources of the
    # reference; the walk must not stop at the shallower one. Order is
    # breadth-first: the shallower leaf comes first.
    prompt = {
        "1": {"class_type": "LoadImage", "inputs": {"image": "SHALLOW.png"}},
        "2": {"class_type": "LoadImage", "inputs": {"image": "DEEPER.png"}},
        "3": {"class_type": "ImageScale", "inputs": {"image": ["2", 0], "width": 64, "height": 64}},
        "4": {"class_type": "ImageBatch", "inputs": {"image1": ["1", 0], "image2": ["3", 0]}},
        REF_NODE_ID: _ref_node(ref_image_0=["4", 0]),
    }

    sources = collect_ref_sources(prompt, REF_NODE_ID)

    assert [e["annotated"] for e in sources["ref_image_0"]] == ["SHALLOW.png", "DEEPER.png"]


def test_same_filename_on_two_branches_is_returned_once():
    # One loader wired into two inputs of the same downstream node: the file
    # is a single source and appears once, at its first occurrence.
    prompt = {
        "1": {"class_type": "LoadImage", "inputs": {"image": "same.png"}},
        "2": {"class_type": "LoadImage", "inputs": {"image": "same.png"}},
        "3": {"class_type": "ImageBatch", "inputs": {"image1": ["1", 0], "image2": ["2", 0]}},
        REF_NODE_ID: _ref_node(ref_image_0=["3", 0]),
    }

    sources = collect_ref_sources(prompt, REF_NODE_ID)

    assert [e["annotated"] for e in sources["ref_image_0"]] == ["same.png"]


def test_a_fileless_leaf_does_not_stop_a_deeper_leaf_on_another_branch():
    # Node 1 is a leaf with no media literal; it must not block the walk
    # from reaching the real loader two hops down the other branch.
    prompt = {
        "1": {"class_type": "UpscaleModelLoader", "inputs": {"model_name": "x.pth"}},
        "2": {"class_type": "LoadImage", "inputs": {"image": "wanted.png"}},
        "3": {"class_type": "ImageResize", "inputs": {"image": ["2", 0], "width": 8, "height": 8}},
        "4": {"class_type": "SomeMerge", "inputs": {"model": ["1", 0], "image": ["3", 0]}},
        REF_NODE_ID: _ref_node(ref_image_0=["4", 0]),
    }

    sources = collect_ref_sources(prompt, REF_NODE_ID)

    assert [e["annotated"] for e in sources["ref_image_0"]] == ["wanted.png"]


# --- FL2VA keyframe slots (first_frame / last_frame) ------------------------
#
# The walker is entirely key-name-agnostic; only the traced-key predicate
# knows about first_frame / last_frame. These tests pin that the two FL2VA
# keyframe slots are traced with the same {annotated[, path]} shape as the
# Ref2VA reference slots, keyed by their literal input name.

FL2VA_NODE_ID = "20"


def _fl2va_node(**frame_inputs):
    """A cached-FL2VA node dict with the given first_frame / last_frame wired."""
    inputs = {"prompt": "a cat", "width": 1344, "height": 768, "length": 124}
    inputs.update(frame_inputs)
    return {"class_type": "MiniMaxH3CLIPCachedFL2VA", "inputs": inputs}


def test_fl2va_first_and_last_frame_are_traced_with_paths():
    prompt = {
        "1": {"class_type": "LoadImage", "inputs": {"image": "start.png", "upload": "image"}},
        "2": {"class_type": "LoadImage", "inputs": {"image": "end.png", "upload": "image"}},
        FL2VA_NODE_ID: _fl2va_node(first_frame=["1", 0], last_frame=["2", 0]),
    }

    sources = collect_ref_sources(prompt, FL2VA_NODE_ID)

    assert set(sources) == {"first_frame", "last_frame"}
    assert sources["first_frame"][0]["annotated"] == "start.png"
    assert sources["first_frame"][0]["path"].endswith("/input/start.png")
    assert sources["last_frame"][0]["annotated"] == "end.png"


def test_fl2va_only_one_keyframe_connected_yields_exactly_that_key():
    prompt = {
        "1": {"class_type": "LoadImage", "inputs": {"image": "start.png", "upload": "image"}},
        FL2VA_NODE_ID: _fl2va_node(first_frame=["1", 0]),
    }

    assert set(collect_ref_sources(prompt, FL2VA_NODE_ID)) == {"first_frame"}


def test_fl2va_keyframe_chain_through_intermediate_node_is_followed():
    prompt = {
        "1": {"class_type": "LoadImage", "inputs": {"image": "portrait.jpg", "upload": "image"}},
        "2": {"class_type": "ImageResize", "inputs": {"image": ["1", 0], "width": 512, "height": 512}},
        FL2VA_NODE_ID: _fl2va_node(first_frame=["2", 0]),
    }

    sources = collect_ref_sources(prompt, FL2VA_NODE_ID)

    assert [e["annotated"] for e in sources["first_frame"]] == ["portrait.jpg"]


def test_fl2va_keyframe_off_a_non_loader_branch_yields_no_entry():
    prompt = {
        "1": {"class_type": "EmptyLatentImage", "inputs": {"width": 512, "height": 512}},
        "2": {"class_type": "VAEDecode", "inputs": {"samples": ["1", 0], "vae": ["9", 0]}},
        FL2VA_NODE_ID: _fl2va_node(first_frame=["2", 0]),
    }

    assert collect_ref_sources(prompt, FL2VA_NODE_ID) == {}


def test_non_keyframe_input_named_like_a_frame_is_not_traced():
    # The predicate is an exact match on first_frame / last_frame, not a
    # substring: a hypothetical "middle_frame" / "first_frames" input is not
    # a traced slot.
    prompt = {
        "1": {"class_type": "LoadImage", "inputs": {"image": "x.png", "upload": "image"}},
        FL2VA_NODE_ID: {"class_type": "SomeOtherNode",
                        "inputs": {"middle_frame": ["1", 0], "first_frames": ["1", 0]}},
    }

    assert collect_ref_sources(prompt, FL2VA_NODE_ID) == {}
