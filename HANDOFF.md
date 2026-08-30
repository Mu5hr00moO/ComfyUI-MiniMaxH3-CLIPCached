# HANDOFF

## Stan na: 2026-08-30 / branch master

## Ostatnio zrobione
- `minimaxh3_clipcache/comparison.py` — przeredagowano akapit docstringa
  (linie 4-8), który wciąż odwoływał się do usuniętego
  `scripts/test_cache_roundtrip.py` (usunięty w `e53d3a0`). Nowa treść
  nazywa dwa pliki, które REALNIE importują `_tensors_equal` (zweryfikowane
  przez `grep -l _tensors_equal scripts/*.py`):
  `scripts/test_stock_vs_cache.py` (faza 23: stock == cached-MISS ==
  cached-HIT) oraz `scripts/test_ref2video_equivalence.py` (R4: stock-CLIP
  vs SpyClipProxy dla MiniMaxH3ReferenceToVideo). Rozszerzono też
  uzasadnienie „torch.equal, nie allclose" o przypadek przezroczystego
  proxy. Commit `aa1c24c`, tylko ten jeden akapit.
- `python -m py_compile minimaxh3_clipcache/comparison.py` — OK.
- `pytest tests/test_comparison.py -v` — 7 passed.
- `git diff` ograniczony do tego jednego akapitu.

## Ustalenia istotne dla Chat
- `_tensors_equal` z `minimaxh3_clipcache/comparison.py:20` jest
  importowane przez dokładnie dwa skrypty:
  `scripts/test_stock_vs_cache.py:70` i
  `scripts/test_ref2video_equivalence.py:67`. Żaden inny plik w repo go
  nie importuje.
- Pozostałe tekstowe wzmianki o usuniętych skryptach (poza zakresem tego
  zlecenia, wciąż otwarte):
  - `scripts/test_server_memory_trend_phase17.py:12,16` — komentarze
    narracyjne odwołujące się do usuniętego `test_server_memory_trend.py`.
  - `HANDOFF.md` — ten plik (nadpisywany co sesję).

## Otwarte pytania
- Czy komentarze w `scripts/test_server_memory_trend_phase17.py`
  (odwołania do usuniętego poprzednika harnessu) warto przeredagować przy
  najbliższej okazji dotykającej tego pliku.
- Los 2 „niepewnych" skryptów (`test_clip_unload_isolation.py`,
  `test_vae_memory_isolation.py`) — nadal otwarte, świadomie zostawione.
