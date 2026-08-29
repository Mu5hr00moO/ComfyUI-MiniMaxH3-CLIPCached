"""R4 test: stock CLIP vs SpyClipProxy equivalence for MiniMaxH3ReferenceToVideo.

Full ref2va coverage in a single model session: one reference image (256x256) +
one reference video (22 frames, 128x128) with its soundtrack + one standalone
audio. Spatial dims are small on purpose -- see the REF_*_HW note below; they
control only the VAE activation footprint, not the CLIP path under test. The
stock ref2va node is driven twice with value-identical arguments:

  run_stock: clip = the real clip object, unwrapped
  run_proxy: clip = a fresh SpyClipProxy wrapping that same real clip object

The two outputs are then checked field by field with
minimaxh3_clipcache.comparison._tensors_equal -- torch.equal, exact, not
allclose: same loaded model, same deterministic tokenizer output, same forward
pass, exactly as the original phase 6 / phase 23 equivalence checks.

No cache logic here. Reuses SpyClipProxy / log_memory / CLIP constants from
test_proxy_gate.py and _tensors_equal from the clipcache package.

Reference video frame count (checked in source, not guessed)
-----------------------------------------------------------
comfy_extras/nodes_minimax_h3.py, MiniMaxH3ReferenceToVideo.execute:
    n = frames.shape[0]                      # after optional truncation to frame_count
    if n < 5:
        raise ValueError("MiniMax H3 reference videos need at least 5 frames ...")
    while n % 17 != 5:                        # snap down onto the 17k+5 clip grid
        n -= 1
    frames = frames[:n]
So T >= 5 is the hard minimum; T in {5, 22, 39, ...} sits exactly on the grid
with no silent frame drop. This test uses T = 22: the smallest on-grid count
above the bare minimum, which also exercises multi-frame Qwen video sampling
(frames are sub-sampled at FPS // 2 == 12 -> indices [0, 12] -> a 2-frame
vision block with a timestamp label).

AUDIO contract (checked in source, not guessed)
-----------------------------------------------
comfy_extras/nodes_audio.py + comfy/ldm/minimax/audio_vae.py + comfy/sd.py:
    AUDIO = {"waveform": Tensor [B, C, L], "sample_rate": int}
    MiniMax H3 audio VAE: audio_sample_rate = 32000, stereo (2 output channels),
    800 audio samples per latent frame (40 latent fps).
    _encode_ref_audio feeds waveform[:1].movedim(1, -1) to VAE.encode; a mono
    track is replicate-padded to stereo by vae_encode_crop_pixels.
This test builds a stereo [1, 2, 32000] waveform (1 s) at sample_rate = 32000,
so the torchaudio resample branch is not taken.

Run with the comfyenv conda environment from anywhere:
    conda run -n comfyenv python scripts/test_ref2video_equivalence.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_proxy_gate import (  # noqa: E402
    CLIP_NAME, CLIP_TYPE, SpyClipProxy, log_memory,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# append, NOT insert(0): REPO_ROOT holds a nodes.py of our own that would
# shadow ComfyUI's global `nodes` module (CLIPLoader / VAELoader live there).
# Appending still lets `minimaxh3_clipcache` resolve while ComfyUI's nodes win.
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from minimaxh3_clipcache.comparison import _tensors_equal  # noqa: E402

VAE_NAME = "minimax_h3_video_vae_int8_convrot.safetensors"
AUDIO_VAE_NAME = "minimax_h3_audio_vae_fp32.safetensors"

PROMPT = ("a cinematic shot combining <Picture 1>, the motion of <Video 1> with "
          "its sound <Audio 1>, over ambient <Audio 2>")
WIDTH = 1344
HEIGHT = 768
LENGTH = 124
REF_IMAGE_SIZE = "match"

# Spatial dims are deliberately small (option A): the first attempt at 512 /
# 256 OOM'd the 16 GB card during the Qwen encode -- the ~13 GB encoder plus
# both VAEs stay fully co-resident on the isolated-script path (no aimdo /
# DynamicVRAM streaming or eviction), and the VAE activations for the larger
# refs left no room for the encoder. Shrinking the refs only lowers the VAE
# activation footprint; it does not touch what the CLIP path computes.
REF_IMAGE_HW = 256     # square reference image, IMAGE convention [B, H, W, C]
REF_VIDEO_HW = 128     # square reference video frames
REF_VIDEO_T = 22       # smallest on-grid frame count above the 5-frame minimum;
                       # a deliberate 17k+5 grid value (-> one 2-frame Qwen
                       # video block), unrelated to memory -- kept unchanged
AUDIO_SR = 32000       # == MiniMax H3 audio VAE audio_sample_rate (no resample)
AUDIO_LEN = 32000      # 1 second

RESULTS = []  # (path, "PASS" | "FAIL", detail)


def _leaf_desc(x):
    import torch
    import comfy.nested_tensor

    if isinstance(x, comfy.nested_tensor.NestedTensor):
        return "NestedTensor(" + ", ".join(str(tuple(t.shape)) for t in x.tensors) + ")"
    if isinstance(x, torch.Tensor):
        return "Tensor shape={} dtype={} device={}".format(tuple(x.shape), x.dtype, x.device)
    if isinstance(x, dict):
        return "dict keys={}".format(sorted(x.keys()))
    if isinstance(x, (list, tuple)):
        return "{} len={}".format(type(x).__name__, len(x))
    return "{} = {!r}".format(type(x).__name__, x)


def check(path, a, b):
    """Deep torch.equal comparison of one field; records + prints PASS/FAIL."""
    try:
        _tensors_equal(path, a, b)
    except AssertionError as e:
        RESULTS.append((path, "FAIL", str(e)))
        print("  FAIL {}: {}".format(path, e))
        return False
    RESULTS.append((path, "PASS", _leaf_desc(a)))
    print("  PASS {}: {}".format(path, _leaf_desc(a)))
    return True


def check_eq(path, a, b):
    """Plain == comparison for scalar / key-set fields."""
    if a == b:
        RESULTS.append((path, "PASS", repr(a)))
        print("  PASS {}: {!r}".format(path, a))
        return True
    RESULTS.append((path, "FAIL", "stock={!r} proxy={!r}".format(a, b)))
    print("  FAIL {}: stock={!r} proxy={!r}".format(path, a, b))
    return False


def build_base_refs():
    """Deterministic reference tensors, built once and cloned per run."""
    import torch

    g = torch.Generator().manual_seed(0)
    image = torch.rand(1, REF_IMAGE_HW, REF_IMAGE_HW, 3, generator=g, dtype=torch.float32)
    video = torch.rand(REF_VIDEO_T, REF_VIDEO_HW, REF_VIDEO_HW, 3, generator=g, dtype=torch.float32)
    # AUDIO waveform is [B, C, L]; centre to [-1, 1] like a real track
    video_audio_wf = torch.rand(1, 2, AUDIO_LEN, generator=g, dtype=torch.float32) * 2.0 - 1.0
    standalone_audio_wf = torch.rand(1, 2, AUDIO_LEN, generator=g, dtype=torch.float32) * 2.0 - 1.0
    return {
        "image": image,
        "video": video,
        "video_audio": {"waveform": video_audio_wf, "sample_rate": AUDIO_SR},
        "standalone_audio": {"waveform": standalone_audio_wf, "sample_rate": AUDIO_SR},
    }


def fresh_refs(base):
    """A per-run copy so neither run can observe the other's in-place edits."""
    return {
        "image": base["image"].clone(),
        "video": base["video"].clone(),
        "video_audio": {"waveform": base["video_audio"]["waveform"].clone(),
                        "sample_rate": base["video_audio"]["sample_rate"]},
        "standalone_audio": {"waveform": base["standalone_audio"]["waveform"].clone(),
                             "sample_rate": base["standalone_audio"]["sample_rate"]},
    }


def _peak_vram(label, tag):
    import torch

    if not torch.cuda.is_available():
        return
    print("[{}] PEAK VRAM {}: allocated={:.2f}GiB reserved={:.2f}GiB".format(
        label, tag,
        torch.cuda.max_memory_allocated() / (1024 ** 3),
        torch.cuda.max_memory_reserved() / (1024 ** 3)))


def run_once(label, clip_obj, vae, audio_vae, refs):
    import torch
    from comfy_extras.nodes_minimax_h3 import MiniMaxH3ReferenceToVideo

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    print("=== {}: MiniMaxH3ReferenceToVideo.execute() ===".format(label))
    t0 = time.time()
    try:
        output = MiniMaxH3ReferenceToVideo.execute(
            clip=clip_obj,
            vae=vae,
            audio_vae=audio_vae,
            prompt=PROMPT,
            width=WIDTH,
            height=HEIGHT,
            length=LENGTH,
            ref_image_size=REF_IMAGE_SIZE,
            ref_images={"ref_image_0": refs["image"]},
            ref_videos={"ref_video_0": refs["video"]},
            ref_video_audios={"ref_video_audio_0": refs["video_audio"]},
            ref_audios={"ref_audio_0": refs["standalone_audio"]},
        )
    except BaseException:
        _peak_vram(label, "before failure")
        raise
    print("[{}] execute() finished in {:.1f}s".format(label, time.time() - t0))
    _peak_vram(label, "at success")
    return output


def compare(output_stock, output_proxy):
    import comfy.nested_tensor

    cond_s, latent_s = output_stock.args
    cond_p, latent_p = output_proxy.args

    print("--- cond: container ---")
    check_eq("cond: type", type(cond_s).__name__, type(cond_p).__name__)
    check_eq("cond: len", len(cond_s), len(cond_p))

    for i in range(min(len(cond_s), len(cond_p))):
        tensor_s, extra_s = cond_s[i][0], cond_s[i][1]
        tensor_p, extra_p = cond_p[i][0], cond_p[i][1]
        print("--- cond[{}] ---".format(i))
        check("cond[{}][0] (hidden states)".format(i), tensor_s, tensor_p)
        keys_ok = check_eq("cond[{}][1] extra keys".format(i),
                           sorted(extra_s.keys()), sorted(extra_p.keys()))
        shared = sorted(set(extra_s.keys()) & set(extra_p.keys())) if not keys_ok \
            else sorted(extra_s.keys())
        for k in shared:
            # _tensors_equal recurses through minimax_refs' nested list/dict/tensors
            check("cond[{}][1][{!r}]".format(i, k), extra_s[k], extra_p[k])

    print("--- latent['samples'] ---")
    samples_s = latent_s["samples"]
    samples_p = latent_p["samples"]
    check_eq("latent['samples']: type",
             type(samples_s).__name__, type(samples_p).__name__)
    if isinstance(samples_s, comfy.nested_tensor.NestedTensor) and \
       isinstance(samples_p, comfy.nested_tensor.NestedTensor):
        check_eq("latent['samples']: is_nested", samples_s.is_nested, samples_p.is_nested)
        check_eq("latent['samples']: tensor count",
                 len(samples_s.tensors), len(samples_p.tensors))
        names = ["video", "audio"]
        for i in range(min(len(samples_s.tensors), len(samples_p.tensors))):
            nm = names[i] if i < len(names) else "tensor[{}]".format(i)
            check("latent['samples'].{}".format(nm),
                  samples_s.tensors[i], samples_p.tensors[i])


def main():
    import torch  # noqa: F401
    import nodes
    import comfy.model_management

    print("=== Step 1: load real CLIP / VAE / audio_vae once (stock loaders) ===")
    log_memory("before-load")
    t0 = time.time()
    (clip,) = nodes.CLIPLoader().load_clip(CLIP_NAME, type=CLIP_TYPE)
    print("CLIP loaded in {:.1f}s".format(time.time() - t0))
    (vae,) = nodes.VAELoader().load_vae(VAE_NAME)
    (audio_vae,) = nodes.VAELoader().load_vae(AUDIO_VAE_NAME)
    print("vae is audio_vae? {}".format(vae is audio_vae))
    log_memory("after-load")

    print("=== Step 2: build deterministic references (image + video+audio + audio) ===")
    base = build_base_refs()
    print("  image:            {}".format(_leaf_desc(base["image"])))
    print("  video:            {}".format(_leaf_desc(base["video"])))
    print("  video_audio.wf:   {}  sample_rate={}".format(
        _leaf_desc(base["video_audio"]["waveform"]), base["video_audio"]["sample_rate"]))
    print("  standalone.wf:    {}  sample_rate={}".format(
        _leaf_desc(base["standalone_audio"]["waveform"]), base["standalone_audio"]["sample_rate"]))

    print("=== Step 3: run_stock (real clip, unwrapped) ===")
    output_stock = run_once("run_stock", clip, vae, audio_vae, fresh_refs(base))
    log_memory("after-run_stock")

    print("=== Step 4: run_proxy (fresh SpyClipProxy around the same clip) ===")
    proxy = SpyClipProxy(clip)
    output_proxy = run_once("run_proxy", proxy, vae, audio_vae, fresh_refs(base))
    log_memory("after-run_proxy")
    prompt_seen, kwargs_seen = proxy.last_call
    print("proxy.last_call: prompt={!r}".format(prompt_seen))
    print("proxy.last_call: kwargs keys={}".format(list(kwargs_seen.keys())))
    ref_items = kwargs_seen.get("minimax_ref_items")
    print("proxy.last_call: minimax_ref_items len={} types={}".format(
        len(ref_items) if ref_items is not None else None,
        [it.get("type") for it in ref_items] if ref_items else None))

    print("=== Step 5: field-by-field equivalence (_tensors_equal / torch.equal) ===")
    compare(output_stock, output_proxy)

    print("=== Step 6: targeted unload of the real clip ===")
    log_memory("before-unload")
    comfy.model_management.unload_model_and_clones(clip.patcher)
    log_memory("after-unload")

    print()
    passed = [r for r in RESULTS if r[1] == "PASS"]
    failed = [r for r in RESULTS if r[1] == "FAIL"]
    print("=== SUMMARY: {} field(s) checked, {} PASS, {} FAIL ===".format(
        len(RESULTS), len(passed), len(failed)))
    for path, status, detail in RESULTS:
        print("  {:4s} {}{}".format(status, path, ("  ->  " + detail) if detail else ""))

    if failed:
        print()
        print("=== EQUIVALENCE RESULT: FAIL ===")
        for path, _, detail in failed:
            print("  MISMATCH {}: {}".format(path, detail))
        sys.exit(1)
    print()
    print("=== EQUIVALENCE RESULT: PASS (stock == proxy, exact, all fields) ===")


if __name__ == "__main__":
    main()
