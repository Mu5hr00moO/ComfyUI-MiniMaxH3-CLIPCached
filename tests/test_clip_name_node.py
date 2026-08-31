"""Unit tests for MiniMaxH3CLIPName -- the standalone encoder-checkpoint
picker whose single COMBO output is meant to fan out to the ``clip_name``
input of any number of MiniMaxH3CLIPCachedFL2VA / MiniMaxH3CLIPCachedRef2VA
nodes.

The load-bearing test checks ``MiniMaxH3CLIPName.RETURN_TYPES[0]`` against
the REAL ``comfy_execution.validation.validate_node_input`` from the local
ComfyUI, not a reimplementation of it. That function is the exact gate
ComfyUI applies on Queue when it decides whether one node's output may feed
another node's input, so it is the only thing that actually proves the
``_ComboType`` trick works (and that a plain ``"COMBO"`` string would not).

nodes.py is loaded the same importlib way as in test_node.py -- never as a
bare ``import nodes``, which collides with ComfyUI's own top-level nodes.py.
"""

import importlib.util
import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# conftest.py already resolves and puts this on sys.path; mirror its default.
COMFYUI_ROOT = os.environ.get(
    "COMFYUI_ROOT",
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
)


def _load_node_module():
    spec = importlib.util.spec_from_file_location(
        "minimaxh3clipcached_nodes_clipname_test", os.path.join(REPO_ROOT, "nodes.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def node_module_with_real_comfy_nodes():
    """Yield a freshly loaded repo nodes.py whose ``import nodes`` resolves
    to ComfyUI's real top-level nodes module.

    Why this is needed: MiniMaxH3CLIPCachedFL2VA / Ref2VA read
    ``nodes.MAX_RESOLUTION`` inside INPUT_TYPES(). Under real ComfyUI
    ``import nodes`` picks up the already-loaded ComfyUI module. Under
    pytest it instead resolves to this repo's own nodes.py (the repo root
    precedes the ComfyUI root on sys.path), which has no MAX_RESOLUTION, so
    those two nodes' INPUT_TYPES() would raise AttributeError. We load
    ComfyUI's real nodes.py (~0.2 s, no server, no init_extra_nodes) into
    ``sys.modules["nodes"]`` for the duration of the test and restore the
    previous binding afterwards. MiniMaxH3CLIPName.INPUT_TYPES() itself does
    NOT need this -- it only has the clip_name widget -- but the equivalence
    and validation tests compare against FL2VA's INPUT_TYPES(), which does.
    """
    saved = sys.modules.get("nodes")
    real_spec = importlib.util.spec_from_file_location(
        "nodes", os.path.join(COMFYUI_ROOT, "nodes.py"))
    real_nodes = importlib.util.module_from_spec(real_spec)
    sys.modules["nodes"] = real_nodes
    try:
        real_spec.loader.exec_module(real_nodes)
        assert hasattr(real_nodes, "MAX_RESOLUTION"), (
            "ComfyUI's real nodes.py did not expose MAX_RESOLUTION -- the "
            "layout assumed by this fixture has changed")
        yield _load_node_module()
    finally:
        if saved is not None:
            sys.modules["nodes"] = saved
        else:
            sys.modules.pop("nodes", None)


# --- execute() -------------------------------------------------------------

@pytest.mark.parametrize("clip_name", [
    "qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
    "some_other_encoder.safetensors",
    "nested/dir/encoder.safetensors",
    "",
])
def test_execute_returns_the_selected_value_verbatim(clip_name):
    node_module = _load_node_module()
    node = node_module.MiniMaxH3CLIPName()
    assert node.execute(clip_name) == (clip_name,)


def test_execute_output_tuple_matches_return_names_arity():
    node_module = _load_node_module()
    assert len(node_module.MiniMaxH3CLIPName.RETURN_TYPES) == 1
    assert node_module.MiniMaxH3CLIPName.RETURN_NAMES == ("clip_name",)
    assert node_module.MiniMaxH3CLIPName.FUNCTION == "execute"
    assert node_module.MiniMaxH3CLIPName.CATEGORY == "model/conditioning/minimax/cached"
    # No IS_CHANGED on this node by design (see its docstring).
    assert not hasattr(node_module.MiniMaxH3CLIPName, "IS_CHANGED")


# --- shared clip_name INPUT_TYPES entry -----------------------------------

def test_clip_name_input_spec_is_shared_with_fl2va_and_ref2va(node_module_with_real_comfy_nodes):
    m = node_module_with_real_comfy_nodes

    fl2va = m.MiniMaxH3CLIPCachedFL2VA.INPUT_TYPES()["required"]["clip_name"]
    ref2va = m.MiniMaxH3CLIPCachedRef2VA.INPUT_TYPES()["required"]["clip_name"]
    clip_name_node = m.MiniMaxH3CLIPName.INPUT_TYPES()["required"]["clip_name"]

    # FL2VA and Ref2VA now build the entry from _clip_name_input_spec() with
    # the default tooltip -- the exact (options_list, options_dict) pair.
    assert fl2va == m._clip_name_input_spec()
    assert ref2va == m._clip_name_input_spec()
    assert fl2va == ref2va

    # The standalone picker uses the same options list (both go through
    # _clip_name_input_spec()); only its tooltip is tailored to its role.
    assert clip_name_node[0] == fl2va[0]
    assert clip_name_node[1]["tooltip"] != fl2va[1]["tooltip"]


# --- the load-bearing check: real comfy_execution.validation --------------

def test_comfy_execution_validation_is_importable():
    """Answer to the open question: comfy_execution.validation imports in
    the plain pytest environment with no extra process state -- conftest.py
    already puts the ComfyUI root on sys.path and that is all it needs
    (its only dependency is ``from comfy_api.latest import IO``)."""
    from comfy_execution.validation import validate_node_input  # noqa: F401


def test_return_type_validates_against_a_real_combo_options_list(node_module_with_real_comfy_nodes):
    """MiniMaxH3CLIPName.RETURN_TYPES[0] must satisfy ComfyUI's own
    output->input gate against the raw option list that a clip_name widget
    presents as its input_type on Queue."""
    from comfy_execution.validation import validate_node_input

    m = node_module_with_real_comfy_nodes
    return_type = m.MiniMaxH3CLIPName.RETURN_TYPES[0]
    combo_options_list = m.MiniMaxH3CLIPCachedFL2VA.INPUT_TYPES()["required"]["clip_name"][0]
    assert isinstance(combo_options_list, list)

    assert validate_node_input(return_type, combo_options_list) is True
    # Also matches the string "COMBO" itself (some code paths pass that).
    assert validate_node_input(return_type, "COMBO") is True
    # But must NOT accidentally match unrelated string types.
    assert validate_node_input(return_type, "STRING") is False
    assert validate_node_input(return_type, "CONDITIONING") is False


def test_plain_combo_string_would_fail_the_same_validation(node_module_with_real_comfy_nodes):
    """The naive ``RETURN_TYPES = ("COMBO",)`` (a plain str) is rejected by
    the same real gate against the option list -- this is why _ComboType
    exists and is not mere cosmetics."""
    from comfy_execution.validation import validate_node_input

    m = node_module_with_real_comfy_nodes
    combo_options_list = m.MiniMaxH3CLIPCachedFL2VA.INPUT_TYPES()["required"]["clip_name"][0]

    assert validate_node_input("COMBO", combo_options_list) is False
    # And the working value really is a str subclass carrying "COMBO", not a
    # list frozen at import time.
    return_type = m.MiniMaxH3CLIPName.RETURN_TYPES[0]
    assert isinstance(return_type, str)
    assert str(return_type) == "COMBO"
    assert type(return_type) is m._ComboType


# --- registration --------------------------------------------------------

def test_registered_in_node_mappings():
    spec = importlib.util.spec_from_file_location(
        "minimaxh3clipcached_package_clipname_test", os.path.join(REPO_ROOT, "__init__.py"))
    package = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = package
    spec.loader.exec_module(package)

    assert package.NODE_CLASS_MAPPINGS["MiniMaxH3CLIPName"].__name__ == "MiniMaxH3CLIPName"
    assert package.NODE_DISPLAY_NAME_MAPPINGS["MiniMaxH3CLIPName"] == "MiniMax H3 CLIP Name"
