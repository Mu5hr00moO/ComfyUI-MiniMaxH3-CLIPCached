"""End-to-end: CachedClipProxy folds the prompt's resolved embedding:
(textual inversion) content into the fingerprint, so a file swapped under an
unchanged name flips a HIT into a MISS -- and a failure in that auxiliary
layer never breaks a normal MISS. No GPU, no ~27 GB encoder, no real clip.
"""

import torch

import safetensors.torch

import comfy.text_encoders.minimax as minimax

from minimaxh3_clipcache import embeddings
from minimaxh3_clipcache.proxy import MINIMAX_H3_HIDDEN_DIM, CachedClipProxy

CLIP_NAME = "fake_clip.safetensors"


class FakeRealClip:
    def tokenize(self, prompt, **kwargs):
        return ("real_tokens", prompt, kwargs)

    def encode_from_tokens_scheduled(self, tokens):
        return [[torch.zeros(1, MINIMAX_H3_HIDDEN_DIM), {"pooled_output": None}]]


def _proxy(cache_dir):
    return CachedClipProxy(lambda: FakeRealClip(), CLIP_NAME, 1, 2, cache_dir)


def _write_ti(path, seed):
    g = torch.Generator().manual_seed(seed)
    safetensors.torch.save_file(
        {"qwen3vl_32b": torch.randn(5120, generator=g).contiguous()}, str(path))


def test_proxy_fingerprint_tracks_embedding_file_content(tmp_path, monkeypatch):
    emb_dir = tmp_path / "emb"
    emb_dir.mkdir()
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    ti_path = emb_dir / "tiX.safetensors"

    subtok = minimax.MiniMaxH3Tokenizer(
        embedding_directory=[str(emb_dir)], tokenizer_data={}).qwen3vl_32b
    monkeypatch.setattr(embeddings, "_build_minimax_tokenizer", lambda: subtok)
    embeddings._reset_for_tests()

    try:
        _write_ti(ti_path, seed=1)
        p1 = _proxy(cache_dir)
        p1.encode_from_tokens_scheduled(p1.tokenize("cat embedding:tiX", images=[]))
        fp1 = p1.last_fingerprint
        assert p1.last_hit is False

        # identical file -> HIT on the entry the first run wrote
        p2 = _proxy(cache_dir)
        p2.encode_from_tokens_scheduled(p2.tokenize("cat embedding:tiX", images=[]))
        assert p2.last_hit is True
        assert p2.last_fingerprint == fp1

        # swapped file content, unchanged name/prompt -> new fingerprint -> MISS
        _write_ti(ti_path, seed=2)
        p3 = _proxy(cache_dir)
        p3.encode_from_tokens_scheduled(p3.tokenize("cat embedding:tiX", images=[]))
        assert p3.last_fingerprint != fp1
        assert p3.last_hit is False
    finally:
        embeddings._reset_for_tests()


def test_proxy_miss_completes_when_subtokenizer_build_fails(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    def _boom():
        raise RuntimeError("simulated tokenizer build failure")

    monkeypatch.setattr(embeddings, "_build_minimax_tokenizer", _boom)
    embeddings._reset_for_tests()

    try:
        proxy = _proxy(cache_dir)
        tokens = proxy.tokenize("a cat embedding:some_ti", images=[])
        cond = proxy.encode_from_tokens_scheduled(tokens)  # must not raise

        assert torch.equal(cond[0][0], torch.zeros(1, MINIMAX_H3_HIDDEN_DIM))
        assert proxy.did_load_real_clip is True

        # fingerprint equals the no-embedding one for the same prompt string
        from minimaxh3_clipcache.fingerprint import CACHE_SCHEMA_VERSION, compute_fingerprint
        expected = compute_fingerprint(
            "a cat embedding:some_ti", {"images": []}, CLIP_NAME, 1, 2,
            CACHE_SCHEMA_VERSION, encoder_abi_id=proxy.encoder_abi_id,
            clip_ctime_ns=None, embedding_tensors=[],
        )
        assert proxy.last_fingerprint == expected
    finally:
        embeddings._reset_for_tests()
