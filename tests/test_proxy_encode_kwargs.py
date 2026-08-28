"""CachedClipProxy.encode_from_tokens_scheduled() must reject the non-default
unprojected / add_dict / show_pbar kwargs of the real
comfy.sd.CLIP.encode_from_tokens_scheduled() rather than silently ignore them
(which would serve a cached result that does not match what was asked for).
The default call path -- the only one the stock MiniMax H3 nodes use -- must
keep working unchanged. No GPU, no ComfyUI, no real clip.
"""

import pytest
import torch

from minimaxh3_clipcache.proxy import MINIMAX_H3_HIDDEN_DIM, CachedClipProxy

CLIP_NAME = "fake_clip.safetensors"
CLIP_FILE_SIZE = 12345
CLIP_MTIME_NS = 67890


class FakeRealClip:
    def tokenize(self, prompt, **kwargs):
        return ("real_tokens", prompt, kwargs)

    def encode_from_tokens_scheduled(self, tokens):
        return [[torch.zeros(1, MINIMAX_H3_HIDDEN_DIM), {"pooled_output": None}]]


def _make_proxy(tmp_path):
    return CachedClipProxy(
        lambda: FakeRealClip(), CLIP_NAME, CLIP_FILE_SIZE, CLIP_MTIME_NS, tmp_path,
    )


@pytest.mark.parametrize("kwargs", [
    {"unprojected": True},
    {"add_dict": {"x": 1}},
    {"show_pbar": False},
])
def test_non_default_kwargs_raise_runtimeerror(tmp_path, kwargs):
    proxy = _make_proxy(tmp_path)
    tokens = proxy.tokenize("a prompt", images=[])
    with pytest.raises(RuntimeError, match="unprojected/add_dict/show_pbar"):
        proxy.encode_from_tokens_scheduled(tokens, **kwargs)


def test_default_kwargs_still_work(tmp_path):
    proxy = _make_proxy(tmp_path)
    tokens = proxy.tokenize("a prompt", images=[])

    # not passing them at all
    cond = proxy.encode_from_tokens_scheduled(tokens)
    assert torch.equal(cond[0][0], torch.zeros(1, MINIMAX_H3_HIDDEN_DIM))

    # passing them explicitly at their default values
    proxy2 = _make_proxy(tmp_path)
    tokens2 = proxy2.tokenize("another prompt", images=[])
    cond2 = proxy2.encode_from_tokens_scheduled(
        tokens2, unprojected=False, add_dict=None, show_pbar=True)
    assert torch.equal(cond2[0][0], torch.zeros(1, MINIMAX_H3_HIDDEN_DIM))
