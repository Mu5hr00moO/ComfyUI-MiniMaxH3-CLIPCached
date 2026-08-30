# HANDOFF

## Stan na: 2026-08-30 / branch master

## Ostatnio zrobione
- Usunięto 5 skryptów z `scripts/` zidentyfikowanych w audycie jako
  zastąpione przez późniejsze fazy — 5 osobnych commitów (`3e63257`,
  `e53d3a0`, `dd6c0af`, `e4b5f2e`, `f0f16a8`), po jednym na skrypt, każdy
  z ewentualną edycją pliku sprzężonego w tym samym commicie:
  1. `test_proxy_equivalence.py` + reformułowanie wzmianki w `README.md`
     (usunięty martwy link, zdanie o wcześniejszym allclose-checku
     zostało).
  2. `test_cache_roundtrip.py` (bez towarzyszącej edycji wg zlecenia).
  3. `test_ref2video_gate.py` (bez towarzyszącej edycji).
  4. `test_server_memory_trend.py` + usunięcie wpisu z `SERVER_SCRIPTS`
     w `tests/test_server_script_safety.py`.
  5. `test_clip_unload_isolation_aimdo.py` (self-declared BROKEN).
- Pozostałe 9 skryptów (w tym oba "niepewne": `test_clip_unload_isolation.py`,
  `test_vae_memory_isolation.py`) bez zmian.
- Pełny pytest: 261 passed, 0 fail, 0 skip (było 262 — spadek o 1 to
  parametr `test_server_memory_trend.py` usunięty z listy w kroku 4).
  `test_server_script_safety.py`: 4 parametry, wszystkie PASS
  (`test_server_memory_trend_phase17.py`, `test_ref2video_memory_trend.py`,
  `test_ref2video_server_e2e.py`, `test_ref2video_server_hit.py`).

## Ustalenia istotne dla Chat
- `scripts/test_proxy_gate.py` zostaje — pełni podwójną rolę (historyczna
  bramka faza 4-5 + współdzielony fixture `SpyClipProxy`/stałe loadera/
  `log_memory`), importowany przez pozostałe skrypty GPU.
- Trzy pliki wciąż zawierają tekstowe wzmianki o usuniętych skryptach,
  poza zakresem tego zlecenia (grep repo-wide je pokazuje):
  - `minimaxh3_clipcache/comparison.py:4` — docstring "Shared between
    scripts/test_cache_roundtrip.py ...". Zlecenie wskazywało krok 2 jako
    "bez sprzężeń zewnętrznych" / "bez towarzyszącej edycji", więc nie
    ruszane. Drugi człon tego samego zdania ("phase 23 ... test")
    wskazuje na wciąż istniejący `test_stock_vs_cache.py`.
  - `scripts/test_server_memory_trend_phase17.py:12,16` — komentarze
    narracyjne ("extends the phase 24 step 3c harness
    (test_server_memory_trend.py)"). `scripts/` jawnie wyłączone ze
    zmian w zleceniu.
  - `HANDOFF.md` — ten plik (bieżący, nadpisywany co sesję).
- `README.md` linkuje już tylko `scripts/test_stock_vs_cache.py` (2x);
  martwy link do `test_proxy_equivalence.py` usunięty.
- `CLAUDE.md` i `CACHE_MANAGER_PLAN.md` — czyste, nigdy nie odnosiły się
  do żadnego z 5 usuniętych skryptów.

## Otwarte pytania
- Czy `minimaxh3_clipcache/comparison.py:4` (docstring) ma zostać
  poprawiony osobnym commitem, żeby nie odwoływał się do usuniętego
  `test_cache_roundtrip.py` — decyzja Chat/Kamila. Zlecenie tego nie
  obejmowało.
- Czy komentarze w `test_server_memory_trend_phase17.py` (odwołania do
  usuniętego poprzednika harnessu) warto przeredagować przy najbliższej
  okazji dotykającej tego pliku.
- Los 2 "niepewnych" skryptów (`test_clip_unload_isolation.py`,
  `test_vae_memory_isolation.py`) — nadal otwarte, tym razem świadomie
  zostawione.
