"""Unit tests for the public MiniMaxH3CLIPCachedRef2VA node (nodes.py at repo
root). No GPU, no real encoder, no real ComfyUI startup -- the stock
MiniMaxH3ReferenceToVideo.execute(), the cache loader helpers, and
unload_model_and_clones() are all monkeypatched. Mirrors test_node.py's
FL2VA tests a/b/c/g/h, plus a mapping-regression guard.
"""

import importlib.util
import math
import os
import sys

import pytest
import torch

import comfy.model_management
from comfy_extras.nodes_minimax_h3 import MiniMaxH3ReferenceToVideo

from minimaxh3_clipcache.proxy import MINIMAX_H3_HIDDEN_DIM

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CLIP_NAME = "fake_clip.safetensors"
FAKE_FILE_SIZE = 111
FAKE_MTIME_NS = 222


def _load_node_module():
    spec = importlib.util.spec_from_file_location(
        "minimaxh3clipcached_nodes_ref2va_under_test", os.path.join(REPO_ROOT, "nodes.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FakeRealClip:
    def __init__(self):
        self.patcher = "fake_patcher"
        self.tokenize_calls = 0
        self.encode_calls = 0
        self.last_kwargs = None

    def tokenize(self, prompt, **kwargs):
        self.tokenize_calls += 1
        self.last_kwargs = kwargs
        return ("real_tokens", prompt, kwargs)

    def encode_from_tokens_scheduled(self, tokens):
        self.encode_calls += 1
        return [[torch.zeros(1, MINIMAX_H3_HIDDEN_DIM), {"pooled_output": None}]]


def _make_unload_counter():
    calls = {"count": 0, "args": []}

    def fake_unload(*args, **kwargs):
        calls["count"] += 1
        calls["args"].append((args, kwargs))

    return fake_unload, calls


def _patch_common(monkeypatch, node_module, tmp_path, fake_execute, real_clip):
    monkeypatch.setattr(MiniMaxH3ReferenceToVideo, "execute", classmethod(fake_execute))
    monkeypatch.setattr(node_module, "resolve_clip_stat",
                        lambda clip_name: (FAKE_FILE_SIZE, FAKE_MTIME_NS))
    monkeypatch.setattr(node_module, "build_clip_loader_fn",
                        lambda clip_name: (lambda: real_clip))
    monkeypatch.setattr(node_module, "CACHE_DIR", tmp_path)

    fake_unload, unload_calls = _make_unload_counter()
    monkeypatch.setattr(comfy.model_management, "unload_model_and_clones", fake_unload)
    return unload_calls


def _execute(node, **overrides):
    kwargs = dict(
        clip_name=CLIP_NAME, vae="fake_vae", audio_vae="fake_audio_vae",
        prompt="a ref2va prompt", width=1344, height=768, length=124,
    )
    kwargs.update(overrides)
    return node.execute(**kwargs)


def test_a_execute_not_touching_clip_never_unloads(monkeypatch, tmp_path):
    node_module = _load_node_module()
    real_clip = FakeRealClip()

    def fake_execute(cls, clip, vae, audio_vae, prompt, width, height, length,
                     ref_image_size="match", ref_images=None, ref_videos=None,
                     ref_video_audios=None, ref_audios=None):
        return ("cond_fake", "latent_fake")

    unload_calls = _patch_common(monkeypatch, node_module, tmp_path, fake_execute, real_clip)

    node = node_module.MiniMaxH3CLIPCachedRef2VA()
    cond, latent = _execute(node)

    assert (cond, latent) == ("cond_fake", "latent_fake")
    assert unload_calls["count"] == 0


def test_b_execute_touching_clip_unloads_exactly_once(monkeypatch, tmp_path):
    node_module = _load_node_module()
    real_clip = FakeRealClip()

    def fake_execute(cls, clip, vae, audio_vae, prompt, width, height, length,
                     ref_image_size="match", ref_images=None, ref_videos=None,
                     ref_video_audios=None, ref_audios=None):
        tokens = clip.tokenize(prompt, minimax_ref_items=[])
        cond = clip.encode_from_tokens_scheduled(tokens)
        return (cond, "latent_fake")

    unload_calls = _patch_common(monkeypatch, node_module, tmp_path, fake_execute, real_clip)

    node = node_module.MiniMaxH3CLIPCachedRef2VA()
    cond, latent = _execute(node)

    assert real_clip.tokenize_calls == 1
    assert real_clip.encode_calls == 1
    assert torch.equal(cond[0][0], torch.zeros(1, MINIMAX_H3_HIDDEN_DIM))
    assert unload_calls["count"] == 1
    assert unload_calls["args"][0][0] == (real_clip.patcher,)


def test_c_execute_raising_after_loading_clip_still_unloads_and_propagates(monkeypatch, tmp_path):
    node_module = _load_node_module()
    real_clip = FakeRealClip()

    def fake_execute(cls, clip, vae, audio_vae, prompt, width, height, length,
                     ref_image_size="match", ref_images=None, ref_videos=None,
                     ref_video_audios=None, ref_audios=None):
        tokens = clip.tokenize(prompt, minimax_ref_items=[])
        clip.encode_from_tokens_scheduled(tokens)
        raise RuntimeError("simulated ref2va failure")

    unload_calls = _patch_common(monkeypatch, node_module, tmp_path, fake_execute, real_clip)

    node = node_module.MiniMaxH3CLIPCachedRef2VA()
    with pytest.raises(RuntimeError, match="simulated ref2va failure"):
        _execute(node)

    assert real_clip.tokenize_calls == 1
    assert real_clip.encode_calls == 1
    assert unload_calls["count"] == 1
    assert unload_calls["args"][0][0] == (real_clip.patcher,)


def test_d_slot_args_reach_stock_execute_as_named_dicts(monkeypatch, tmp_path):
    """The node's flat optional slots must arrive at the stock execute() as
    the {name: tensor} dicts it expects, with a bare video's soundtrack slot
    collapsing to None."""
    node_module = _load_node_module()
    real_clip = FakeRealClip()
    seen = {}

    def fake_execute(cls, clip, vae, audio_vae, prompt, width, height, length,
                     ref_image_size="match", ref_images=None, ref_videos=None,
                     ref_video_audios=None, ref_audios=None):
        seen.update(ref_images=ref_images, ref_videos=ref_videos,
                    ref_video_audios=ref_video_audios, ref_audios=ref_audios,
                    ref_image_size=ref_image_size, audio_vae=audio_vae)
        return ("cond_fake", "latent_fake")

    _patch_common(monkeypatch, node_module, tmp_path, fake_execute, real_clip)

    img, vid = torch.zeros(1, 4, 4, 3), torch.zeros(6, 4, 4, 3)
    node = node_module.MiniMaxH3CLIPCachedRef2VA()
    _execute(node, ref_image_2=img, ref_video_0=vid, ref_image_size="max")

    assert list(seen["ref_images"].keys()) == ["ref_image_2"]
    assert seen["ref_images"]["ref_image_2"] is img
    assert list(seen["ref_videos"].keys()) == ["ref_video_0"]
    assert seen["ref_video_audios"] is None
    assert seen["ref_audios"] is None
    assert seen["ref_image_size"] == "max"
    assert seen["audio_vae"] == "fake_audio_vae"


def test_e_is_changed_refresh_forces_reexecution_every_call():
    node_module = _load_node_module()
    cls = node_module.MiniMaxH3CLIPCachedRef2VA

    first = cls.IS_CHANGED(cache_mode="refresh", clip_name=CLIP_NAME, prompt="p", width=1344)
    second = cls.IS_CHANGED(cache_mode="refresh", clip_name=CLIP_NAME, prompt="p", width=1344)

    assert isinstance(first, float) and math.isnan(first)
    assert isinstance(second, float) and math.isnan(second)
    assert not (first == second)


def test_f_is_changed_auto_is_stable_across_calls():
    node_module = _load_node_module()
    cls = node_module.MiniMaxH3CLIPCachedRef2VA

    assert cls.IS_CHANGED(cache_mode="auto", prompt="p") == \
           cls.IS_CHANGED(cache_mode="auto", prompt="p")
    assert cls.IS_CHANGED(prompt="p", width=1344) == cls.IS_CHANGED(prompt="p", width=1344)


def test_i_is_changed_auto_reflects_checkpoint_file_identity(monkeypatch):
    """cache_mode="auto" must change when the checkpoint file underneath
    an unchanged clip_name changes (swapped file, same filename) --
    otherwise ComfyUI's own execution cache could skip re-running this
    node entirely and our on-disk fingerprint (which does include file
    identity) would never even get computed. Same-stat calls must stay
    stable so an unchanged graph still hits ComfyUI's own execution
    cache."""
    node_module = _load_node_module()
    cls = node_module.MiniMaxH3CLIPCachedRef2VA

    monkeypatch.setattr(node_module, "resolve_clip_stat", lambda clip_name: (111, 222))
    before = cls.IS_CHANGED(cache_mode="auto", clip_name=CLIP_NAME, prompt="p")
    before_again = cls.IS_CHANGED(cache_mode="auto", clip_name=CLIP_NAME, prompt="p")
    assert before == before_again

    monkeypatch.setattr(node_module, "resolve_clip_stat", lambda clip_name: (333, 444))
    after = cls.IS_CHANGED(cache_mode="auto", clip_name=CLIP_NAME, prompt="p")
    assert after != before


def test_i_is_changed_returns_nan_when_checkpoint_file_missing(monkeypatch):
    """Same as FL2VA: a missing checkpoint file must make IS_CHANGED return
    NaN rather than raise from ComfyUI's scheduling layer, so execute() runs
    and surfaces the real FileNotFoundError as this node's own error."""
    node_module = _load_node_module()
    cls = node_module.MiniMaxH3CLIPCachedRef2VA

    def _missing(clip_name):
        raise FileNotFoundError("text_encoders file {!r} not found".format(clip_name))

    monkeypatch.setattr(node_module, "resolve_clip_stat", _missing)

    result = cls.IS_CHANGED(cache_mode="auto", clip_name=CLIP_NAME, prompt="p")
    assert isinstance(result, float) and math.isnan(result)
    assert result != result  # NaN != NaN -- forces re-execution instead of raising


def test_h_encoder_unloaded_before_stock_nodes_post_encode_work(monkeypatch, tmp_path):
    """Same guarantee as FL2VA: the real encoder must be released as soon as
    encode_from_tokens_scheduled() returns, before any work the stock node
    still does afterwards. Today's stock Ref2VA does its VAE ref-encoding
    BEFORE the CLIP encode, so this is currently a no-op for Ref2VA in
    practice -- this test guards the contract in case that ever changes."""
    node_module = _load_node_module()
    real_clip = FakeRealClip()

    def fake_execute(cls, clip, vae, audio_vae, prompt, width, height, length,
                     ref_image_size="match", ref_images=None, ref_videos=None,
                     ref_video_audios=None, ref_audios=None):
        tokens = clip.tokenize(prompt, minimax_ref_items=[])
        cond = clip.encode_from_tokens_scheduled(tokens)
        assert unload_calls["count"] == 1
        assert clip.real_clip is None
        return (cond, "latent_fake")

    unload_calls = _patch_common(monkeypatch, node_module, tmp_path, fake_execute, real_clip)

    node = node_module.MiniMaxH3CLIPCachedRef2VA()
    cond, latent = _execute(node)

    assert real_clip.tokenize_calls == 1
    assert real_clip.encode_calls == 1
    assert unload_calls["count"] == 1
    assert unload_calls["args"][0][0] == (real_clip.patcher,)


def test_j_finally_still_unloads_when_real_encode_itself_raises(monkeypatch, tmp_path):
    """If the real encoder's own encode_from_tokens_scheduled() raises before
    the proxy's new early-unload runs, nodes.py's outer finally must still be
    the safety net that releases the encoder -- exactly once."""
    node_module = _load_node_module()

    class FailingRealClip(FakeRealClip):
        def encode_from_tokens_scheduled(self, tokens):
            self.encode_calls += 1
            raise RuntimeError("simulated ref2va encoder failure")

    real_clip = FailingRealClip()

    def fake_execute(cls, clip, vae, audio_vae, prompt, width, height, length,
                     ref_image_size="match", ref_images=None, ref_videos=None,
                     ref_video_audios=None, ref_audios=None):
        tokens = clip.tokenize(prompt, minimax_ref_items=[])
        return clip.encode_from_tokens_scheduled(tokens)

    unload_calls = _patch_common(monkeypatch, node_module, tmp_path, fake_execute, real_clip)

    node = node_module.MiniMaxH3CLIPCachedRef2VA()
    with pytest.raises(RuntimeError, match="simulated ref2va encoder failure"):
        _execute(node)

    assert unload_calls["count"] == 1
    assert unload_calls["args"][0][0] == (real_clip.patcher,)


def test_k_build_reference_items_images_and_video_no_audio():
    node_module = _load_node_module()
    i0, i1 = torch.zeros(1, 2, 2, 3), torch.ones(1, 2, 2, 3)
    v0 = torch.zeros(4, 2, 2, 3)

    items = node_module._build_reference_items(
        {"ref_image_0": i0, "ref_image_1": i1}, {"ref_video_0": v0}, None, None)

    assert [t for t, _ in items] == ["image", "image", "video"]
    assert items[0][1] is i0 and items[1][1] is i1 and items[2][1] is v0


def test_l_build_reference_items_matched_audio_comes_before_its_video():
    """The stock node feeds a reference video's soundtrack to the encoder as
    an "<Audio N>" marker immediately before the video (CLAUDE.md R1). The
    audio entry carries no tensor -- the waveform never reaches the encoder."""
    node_module = _load_node_module()
    v1 = torch.zeros(4, 2, 2, 3)

    items = node_module._build_reference_items(
        None, {"ref_video_1": v1}, {"ref_video_audio_1": "wave"}, None)

    assert items == [("audio", None), ("video", v1)]


def test_m_build_reference_items_standalone_audio_only():
    node_module = _load_node_module()

    items = node_module._build_reference_items(
        None, None, None, {"ref_audio_0": "w0", "ref_audio_2": "w2"})

    assert items == [("audio", None), ("audio", None)]


def test_n_build_reference_items_full_mix_ordering():
    node_module = _load_node_module()
    i0, i2 = torch.zeros(1, 2, 2, 3), torch.ones(1, 2, 2, 3)
    v0, v1 = torch.zeros(4, 2, 2, 3), torch.ones(4, 2, 2, 3)

    items = node_module._build_reference_items(
        {"ref_image_0": i0, "ref_image_2": i2},
        {"ref_video_0": v0, "ref_video_1": v1},
        {"ref_video_audio_1": "wave"},          # only video 1 has a soundtrack
        {"ref_audio_0": "standalone"},
    )

    assert [t for t, _ in items] == [
        "image", "image", "video", "audio", "video", "audio"]
    assert items[2][1] is v0            # video 0: no matched audio, no marker before it
    assert items[3] == ("audio", None)  # video 1's soundtrack marker ...
    assert items[4][1] is v1            # ... immediately before video 1
    assert items[5] == ("audio", None)  # standalone audio last


def test_o_build_reference_items_orders_by_slot_number_not_dict_order():
    node_module = _load_node_module()
    i0, i2 = torch.zeros(1, 2, 2, 3), torch.ones(1, 2, 2, 3)

    items = node_module._build_reference_items(
        {"ref_image_2": i2, "ref_image_0": i0}, None, None, None)

    assert items[0][1] is i0 and items[1][1] is i2


def test_p_build_reference_items_all_groups_empty_is_empty_list():
    node_module = _load_node_module()
    assert node_module._build_reference_items(None, None, None, None) == []


def test_q_sync_verbose_fresh_miss_writes_ref2va_variant(monkeypatch, tmp_path):
    node_module = _load_node_module()
    monkeypatch.setattr(node_module, "CACHE_DIR", tmp_path)
    from minimaxh3_clipcache.verbose_store import load_verbose

    class _FakeProxy:
        last_fingerprint = "b" * 64
        last_hit = False
        last_core_cache_written = True

    items = [("image", torch.zeros(1, 2, 2, 3)), ("audio", None),
             ("video", torch.zeros(4, 2, 2, 3))]
    node_module._sync_verbose_metadata(
        _FakeProxy(), "ref2va", "a ref2va prompt", CLIP_NAME,
        FAKE_FILE_SIZE, FAKE_MTIME_NS, items)

    system = load_verbose("b" * 64, tmp_path)["system"]
    assert system["node_variant"] == "ref2va"
    assert [r["type"] for r in system["references"]] == ["image", "audio", "video"]
    assert [r["index"] for r in system["references"]] == [0, 1, 2]
    assert "label" not in system["references"][0]  # Ref2VA passes no labels


def test_r_sync_verbose_hit_without_sidecar_backfills(monkeypatch, tmp_path):
    node_module = _load_node_module()
    monkeypatch.setattr(node_module, "CACHE_DIR", tmp_path)
    from minimaxh3_clipcache.verbose_store import load_verbose

    class _FakeProxy:
        last_fingerprint = "b" * 64
        last_hit = True
        last_core_cache_written = None

    node_module._sync_verbose_metadata(
        _FakeProxy(), "ref2va", "a prompt", CLIP_NAME, FAKE_FILE_SIZE, FAKE_MTIME_NS, [])

    assert load_verbose("b" * 64, tmp_path)["system"]["node_variant"] == "ref2va"


def test_s_sync_verbose_hit_with_existing_sidecar_does_not_rewrite(monkeypatch, tmp_path):
    node_module = _load_node_module()
    monkeypatch.setattr(node_module, "CACHE_DIR", tmp_path)
    from minimaxh3_clipcache.verbose_store import save_verbose

    save_verbose("b" * 64, {"prompt": "original", "node_variant": "ref2va", "references": []}, tmp_path)
    before = (tmp_path / ("b" * 64 + ".verbose.json")).read_bytes()

    class _FakeProxy:
        last_fingerprint = "b" * 64
        last_hit = True
        last_core_cache_written = None

    node_module._sync_verbose_metadata(
        _FakeProxy(), "ref2va", "a changed prompt", CLIP_NAME, FAKE_FILE_SIZE, FAKE_MTIME_NS, [])

    assert (tmp_path / ("b" * 64 + ".verbose.json")).read_bytes() == before


def test_g_node_class_mappings_keeps_both_nodes():
    spec = importlib.util.spec_from_file_location(
        "minimaxh3clipcached_package_ref2va_under_test", os.path.join(REPO_ROOT, "__init__.py"))
    package = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = package
    spec.loader.exec_module(package)

    assert package.NODE_CLASS_MAPPINGS["MiniMaxH3CLIPCachedFL2VA"].__name__ == "MiniMaxH3CLIPCachedFL2VA"
    assert package.NODE_CLASS_MAPPINGS["MiniMaxH3CLIPCachedRef2VA"].__name__ == "MiniMaxH3CLIPCachedRef2VA"
    assert package.NODE_DISPLAY_NAME_MAPPINGS["MiniMaxH3CLIPCachedRef2VA"] == "MiniMax H3 CLIP-Cached Ref2VA"
