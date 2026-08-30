# HANDOFF

## Stan na: 2026-08-30 / commit 500d668 (branch master)

## Ostatnio zrobione
- Deduplikacja trzech identycznych bloków logiki między
  `MiniMaxH3CLIPCachedFL2VA` i `MiniMaxH3CLIPCachedRef2VA` w `nodes.py`.
  Wydzielone trzy funkcje modułowe (obok `_build_references` /
  `_sync_verbose_metadata` / `_record_last_used`):
  - `_is_changed_common(clip_name, cache_mode)` — całe ciało `IS_CHANGED`
    (NaN przy refresh / niedostępnym ABI / brakującym pliku, inaczej
    krotka `(cache_mode, clip_name, file_size, mtime_ns, ctime_ns, abi_id)`).
  - `_build_cached_proxy(clip_name, cache_mode) -> (proxy, file_size,
    mtime_ns, ctime_ns)` — `resolve_clip_stat` + `build_clip_loader_fn` +
    `get_encoder_abi_id` + konstrukcja `CachedClipProxy`.
  - `_release_real_clip_safety_net(proxy)` — blok `finally` z targeted
    unload + `del proxy` / `gc.collect()` / `soft_empty_cache()`.
- `IS_CHANGED` każdej klasy pozostaje classmethodem (testy wołają
  `cls.IS_CHANGED(...)` bezpośrednio); ciało to jednolinijkowa delegacja
  `return _is_changed_common(clip_name, cache_mode)`.
- Czysta relokacja kodu, bez zmiany zachowania. Komentarze „dlaczego"
  przeniesione do nowych funkcji, nie zduplikowane w obu klasach.

## Ustalenia istotne dla Chat
- `git diff nodes.py` = tylko przeniesienie kodu do wspólnych funkcji
  (122 wstawień / 148 usunięć, per-klasowe bloki zwinięte do wywołań).
- `python -m py_compile nodes.py` — OK.
- `pytest tests/test_node.py tests/test_node_ref2va.py -v` — 65 passed
  (37 FL2VA + 28 Ref2VA), 0 SKIP, 0 FAIL. Pełny pakiet: 260 passed.
- Pełny output pytest: scratchpad `pytest_refactor.txt` / `pytest_full.txt`.
- Testy patchują `node_module.resolve_clip_stat` /
  `node_module.build_clip_loader_fn` / `node_module.get_encoder_abi_id` /
  `node_module.CachedClipProxy` / `node_module.CACHE_DIR`. Nowe funkcje
  odwołują się do tych nazw jako globali modułu, więc monkeypatch dalej
  je łapie — `nodes.py:_is_changed_common` / `nodes.py:_build_cached_proxy`.
- Jedyna teoretyczna różnica semantyczna: `del proxy` żyje teraz w
  `_release_real_clip_safety_net` i usuwa parametr funkcji, nie lokalną
  `proxy` w `execute()`. `CachedClipProxy` nie ma `__del__` ani cyklu
  referencji, `unload_model_and_clones` i `gc.collect()` /
  `soft_empty_cache()` wołane bez zmian, a obiekt proxy (bez referencji do
  odładowanego już encodera na normalnej ścieżce) i tak jest zwalniany
  przez refcounting przy wyjściu z `execute()` mikrosekundy później —
  brak konsekwencji dla VRAM/RAM, żaden test tego nie pokrywa.
  `nodes.py:_release_real_clip_safety_net`.

## Otwarte pytania
- brak
