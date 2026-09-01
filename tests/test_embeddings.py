"""Unit tests for minimaxh3_clipcache.embeddings.

resolve_prompt_embedding_tensors() runs the *real* stock MiniMax H3
sub-tokenizer (comfy.text_encoders.minimax.MiniMaxH3Tokenizer -> its
qwen3vl_32b SDTokenizer) so the `embedding:` split, name cleanup and
comfy.sd1_clip.load_embed file resolution are exercised without any
reimplementation. No GPU and no ~27 GB encoder -- only the bundled HF
tokenizer plus tiny textual-inversion safetensors written under a tmp
embeddings directory.
"""

import pytest
import torch

import safetensors.torch

import comfy.text_encoders.minimax as minimax

from minimaxh3_clipcache import embeddings


@pytest.fixture(scope="module")
def emb_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("embeddings")


@pytest.fixture(scope="module")
def _real_subtokenizer(emb_dir):
    """The genuine stock sub-tokenizer, built once, pointed at the tmp
    embeddings dir instead of ComfyUI's real models/embeddings."""
    tok = minimax.MiniMaxH3Tokenizer(embedding_directory=[str(emb_dir)], tokenizer_data={})
    return tok.qwen3vl_32b


@pytest.fixture(autouse=True)
def _wire_tokenizer(monkeypatch, _real_subtokenizer):
    """Point the process-wide cached getter at the module's pre-built stock
    sub-tokenizer and clear the cache before and after each test, so a
    real-models-dir instance built by another test file never leaks in or
    out. The build-failure test overrides this again with its own stub."""
    monkeypatch.setattr(embeddings, "_build_minimax_tokenizer", lambda: _real_subtokenizer)
    embeddings._reset_for_tests()
    yield
    embeddings._reset_for_tests()


def _write_ti(path, seed, rows=1):
    g = torch.Generator().manual_seed(seed)
    if rows == 1:
        t = torch.randn(5120, generator=g)
    else:
        t = torch.randn(rows, 5120, generator=g)
    safetensors.torch.save_file({"qwen3vl_32b": t.contiguous()}, str(path))


def test_no_embedding_reference_returns_empty():
    assert embeddings.resolve_prompt_embedding_tensors("a red cat on a bench") == []


def test_non_string_prompt_returns_empty():
    assert embeddings.resolve_prompt_embedding_tensors(None) == []
    assert embeddings.resolve_prompt_embedding_tensors(123) == []


def test_resolves_a_referenced_embedding(emb_dir):
    _write_ti(emb_dir / "ti_a.safetensors", seed=1)
    out = embeddings.resolve_prompt_embedding_tensors("a cat embedding:ti_a please")
    assert len(out) == 1
    assert tuple(out[0].shape) == (5120,)


def test_missing_embedding_file_returns_empty(emb_dir):
    # Stock logs "warning, embedding:X does not exist, ignoring" and resolves
    # nothing, so the encode behaves as if the reference were absent -- the
    # identity layer must match that.
    assert embeddings.resolve_prompt_embedding_tensors("a cat embedding:not_here at all") == []


def test_swapped_file_content_changes_resolved_tensor(emb_dir):
    p = emb_dir / "ti_swap.safetensors"
    _write_ti(p, seed=1)
    a = embeddings.resolve_prompt_embedding_tensors("x embedding:ti_swap")
    _write_ti(p, seed=2)
    b = embeddings.resolve_prompt_embedding_tensors("x embedding:ti_swap")
    assert len(a) == 1 and len(b) == 1
    assert not torch.equal(a[0], b[0])


def test_multi_vector_embedding_yields_one_tensor_per_vector(emb_dir):
    _write_ti(emb_dir / "ti_multi.safetensors", seed=7, rows=3)
    out = embeddings.resolve_prompt_embedding_tensors("x embedding:ti_multi")
    assert len(out) == 3
    assert all(tuple(t.shape) == (5120,) for t in out)


def test_order_of_appearance_is_preserved(emb_dir):
    _write_ti(emb_dir / "ti_one.safetensors", seed=10)
    _write_ti(emb_dir / "ti_two.safetensors", seed=20)
    first = embeddings.resolve_prompt_embedding_tensors("embedding:ti_one then embedding:ti_two")
    second = embeddings.resolve_prompt_embedding_tensors("embedding:ti_two then embedding:ti_one")
    assert len(first) == 2 and len(second) == 2
    assert torch.equal(first[0], second[1])
    assert torch.equal(first[1], second[0])
    assert not torch.equal(first[0], first[1])


def test_build_failure_yields_empty_and_never_raises(monkeypatch):
    def _boom():
        raise RuntimeError("simulated tokenizer build failure")

    monkeypatch.setattr(embeddings, "_build_minimax_tokenizer", _boom)
    embeddings._reset_for_tests()

    assert embeddings.resolve_prompt_embedding_tensors("a cat embedding:whatever") == []
    # Failure is cached: a second call also returns [] without another attempt.
    assert embeddings.resolve_prompt_embedding_tensors("embedding:foo") == []


def test_embedding_identity_digest_matches_manual_hash(emb_dir):
    from minimaxh3_clipcache.fingerprint import hash_embedding_tensors

    _write_ti(emb_dir / "ti_dig.safetensors", seed=3)
    prompt = "a dog embedding:ti_dig"
    tensors = embeddings.resolve_prompt_embedding_tensors(prompt)
    assert embeddings.embedding_identity_digest(prompt) == hash_embedding_tensors(tensors)


def test_embedding_identity_digest_is_none_without_embedding():
    assert embeddings.embedding_identity_digest("a plain prompt") is None
    assert embeddings.embedding_identity_digest(None) is None
