# HANDOFF

## Stan na: 2026-09-01 / branch master / commit 5512e57

## Ostatnio zrobione (luka tożsamości cache dla `embedding:` -- Codex audit MEDIUM #1, ZAMKNIĘTE)

Wariant A (zatwierdzony przez Chat): hash rozwiązanych tensorów textual
inversion z prawdziwego sub-tokenizera stocka, wpięty w fingerprint i
IS_CHANGED. Zero reimplementacji parsingu `embedding:`. Pełny suite:
379 passed (było 360). py_compile / `git diff --check` czyste. 6 commitów:

1. `c1ea2a5` -- nowy `minimaxh3_clipcache/embeddings.py`
2. `809c04e` -- `fingerprint.py`: param `embedding_tensors` + `hash_embedding_tensors`
3. `edee699` -- `proxy.py` + `nodes.py`: wpięcie
4. `5ee0cd1` -- testy (test_embeddings.py, test_proxy_embeddings.py, +test_fingerprint.py, +test_node.py)
5. `5512e57` -- README (nota o kluczu cache)

(HANDOFF = osobny commit, ten.)

### Jak to działa

- `minimaxh3_clipcache/embeddings.py`:
  - `_build_minimax_tokenizer()` -- buduje `comfy.text_encoders.minimax.MiniMaxH3Tokenizer(embedding_directory=folder_paths.get_folder_paths("embeddings"), tokenizer_data={})` i zwraca jego `.qwen3vl_32b` (stockowy `SDTokenizer`). Bez enkodera 27 GB.
  - `_get_minimax_tokenizer()` -- cache raz na proces, sukces i porażka osobno, WARNING raz na sesję (wzorzec `encoder_abi.get_encoder_abi_id`). Nigdy nie podnosi wyjątku.
  - `resolve_prompt_embedding_tensors(prompt) -> list[torch.Tensor]` -- woła realne `tokenize_with_weights(prompt, return_word_ids=False, disable_weights=True)` (dokładnie jak `minimax.py:155`), zbiera tokeny będące `torch.Tensor` w kolejności wystąpienia. `[]` gdy: brak `embedding:`, plik nie istnieje (stock loguje warning i ignoruje), prompt nie-string, tokenizer się nie zbudował. Nigdy nie podnosi.
  - `embedding_identity_digest(prompt) -> str | None` -- `hash_embedding_tensors(resolve_prompt_embedding_tensors(prompt))`, dla IS_CHANGED.
- `fingerprint.py`:
  - `compute_fingerprint(..., embedding_tensors=None)` (keyword-only). Pusta/None -> `_feed_embedding_tensors` nie dokłada NIC do hasza -> digest bit-w-bit jak przed zmianą (marker `b"E"` tylko gdy lista niepusta).
  - `hash_embedding_tensors(list) -> str | None` -- ten sam strumień jako samodzielny sha256, `None` dla pustej listy.
- `proxy.py` (`encode_from_tokens_scheduled`): `embedding_tensors = resolve_prompt_embedding_tensors(prompt)` (ma `prompt` z rozpakowania `tokens`), przekazane do `compute_fingerprint`.
- `nodes.py` (`_is_changed_common(clip_name, cache_mode, prompt=None)`): `embedding_digest = embedding_identity_digest(prompt)`; doklejane jako OSTATNI element krotki tylko gdy `is not None`. Wszystkie 4 `IS_CHANGED` przekazują `kwargs.get("prompt")`. Ścieżki NaN (refresh / brak ABI / brak pliku checkpointu) bez zmian. `prompt=None` = ta sama znana degradacja co `clip_name=None`, bez nowej klasy błędu.

## Ustalenia istotne dla Chat

- Jednorazowe unieważnienie cache: BRAK. Golden-digest test (`test_fingerprint.py::test_o_no_embedding_tensors_is_byte_for_byte_pre_embedding_format`) sprawdza dwie konkretne wartości sha256 policzone na commicie 993ac03 (przed zmianą) -- prompt bez embeddingu daje dokładnie te same digesty.
- Podmiana treści pliku pod tą samą nazwą: MISS. Udowodnione end-to-end przez proxy (`test_proxy_embeddings.py::test_proxy_fingerprint_tracks_embedding_file_content`: MISS -> HIT -> podmiana pliku -> MISS) i na IS_CHANGED (`test_node.py::test_i_is_changed_auto_reflects_embedding_file_content`).
- Awaria budowy sub-tokenizera: `resolve_prompt_embedding_tensors` zwraca `[]`, MISS przez proxy kończy się normalnie (`test_proxy_embeddings.py::test_proxy_miss_completes_when_subtokenizer_build_fails`).
- Koszt: budowa sub-tokenizera ~0.3 s raz na proces serwera (przy pierwszym encode; cache'owana). Nie dotyka ścieżki 27 GB.
- Zakres świadomie zawężony: hashujemy rozwiązane TENSORY (treść), nie `name+stat` -- bo stock (`comfy.sd1_clip.load_embed`) nie wystawia ścieżki pliku, a wariant name+stat wymagałby reimplementacji jego pętli katalogów. Tensor to i tak dokładna treść wchodząca do enkodera.
- Reszta pól fingerprintu / on-disk format / `CACHE_SCHEMA_VERSION` (dalej 2): bez zmian. Marker `b"E"` dołożony PO pętli kwargs, przed `hexdigest()`.

## Otwarte pytania

- brak

## Sugestie (nie polecenia)

- README ma sekcję "Cache key" która nie wymienia komponentu encoder-ABI id
  (dodanego we wcześniejszej rundzie audytu) -- nie ruszane teraz (dyscyplina
  zakresu). Gdyby robić przegląd README, warto dorównać.
