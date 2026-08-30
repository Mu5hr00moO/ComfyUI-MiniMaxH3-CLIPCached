# HANDOFF

## Stan na: 2026-08-30 / branch master

## Ostatnio zrobione
- Audyt `scripts/` (14 ręcznych skryptów GPU): rozpoznanie, które są
  nieaktualne / zastąpione, bez usuwania czegokolwiek. Raport w
  `/tmp/scripts_audit.md` (artefakt tymczasowy, nie w repo). Wynik:
  7 zachować, 5 kandydatów do usunięcia jako zastąpione, 2 niepewne
  (diagnostyka fazy 24, dochodzenie zamknięte). Szczegóły w sekcji
  "Ustalenia" niżej. Zero zmian w kodzie/repo w tej części sesji.
- (wcześniej w tej sesji) Konsolidacja `buildLegacyRow` /
  `buildInconsistentRow` w `web/main.js` -> `buildSimpleRow`
  (`web/main.js:473`). Commit `5eb87b1`. Weryfikacja bez przeglądarki OK,
  render w żywym ComfyUI wciąż do sprawdzenia przez użytkownika.

## Ustalenia istotne dla Chat
- `scripts/test_proxy_gate.py` pełni podwójną rolę: historyczna bramka
  faza 4-5 ORAZ współdzielony moduł-fixture (`SpyClipProxy`, stałe
  loadera, `log_memory`) importowany przez 7 innych skryptów. Nie da się
  go usunąć dopóki żyje którykolwiek importer.
- `tests/test_server_script_safety.py` (pytest) czyta `read_text()`
  źródło 5 skryptów serwerowych: `test_server_memory_trend.py`,
  `test_server_memory_trend_phase17.py`, `test_ref2video_memory_trend.py`,
  `test_ref2video_server_e2e.py`, `test_ref2video_server_hit.py`.
  Usunięcie któregokolwiek wymaga też edycji tego pliku pytest.
- `README.md` linkuje `scripts/test_stock_vs_cache.py` (2x) i
  `scripts/test_proxy_equivalence.py` (1x).
- `test_ref2video_server_e2e.py` pisze `/tmp/r7_last_fingerprint.txt`,
  `test_ref2video_server_hit.py` go czyta — para, trzymać/usuwać razem.
- KEEP (7): test_proxy_gate.py, test_stock_vs_cache.py (kanoniczny
  bit-exact FL2VA), test_ref2video_equivalence.py (kanoniczny exact
  Ref2VA), test_ref2video_server_e2e.py + test_ref2video_server_hit.py
  (e2e Ref2VA MISS/HIT), test_server_memory_trend_phase17.py (trend RAM
  FL2VA, 10 iter), test_ref2video_memory_trend.py (trend RAM Ref2VA,
  jedyny z drugim VAE/audio_vae przez unload).
- USUŃ jako zastąpione (5): test_proxy_equivalence.py (-> stock_vs_cache),
  test_cache_roundtrip.py (-> stock_vs_cache + pytest laziness/
  invalidation), test_ref2video_gate.py (-> ref2video_equivalence, R4
  jest nadzbiorem R3), test_server_memory_trend.py (-> _phase17, ten sam
  harness 3->10 iter), test_clip_unload_isolation_aimdo.py (plik sam
  deklaruje `STATUS: BROKEN, DO NOT RUN AS-IS`, martwa gałąź
  comfy_aimdo).
- NIEPEWNE, decyzja Kamila (2): test_clip_unload_isolation.py,
  test_vae_memory_isolation.py — diagnostyka fazy 24, dochodzenie
  zamknięte w CLAUDE.md (wyciek był na sztucznej ścieżce aimdo=False bez
  main.py; produkcja płaska). Trzymać tylko jako gotowe sondy / zapis
  rozumowania.

## Otwarte pytania
- Czy usuwać 5 kandydatów i 2 niepewne — decyzja Kamila. Audyt nic nie
  usunął.
