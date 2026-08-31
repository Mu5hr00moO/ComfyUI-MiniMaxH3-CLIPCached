# HANDOFF

## Stan na: 2026-08-31 / branch master / commit cf731ba

## Ostatnio zrobione
- **Dual-resolution pairing dostał jawną flagę `is_upscale_target`**
  (commit cf731ba). Wcześniej (commit 5286ce9) parowanie zapisywało do
  bloku `system` każdego sidecara tylko symetryczne
  `paired_fingerprint` / `paired_width` / `paired_height` - front-end
  musiałby zgadywać, który z dwóch wpisów jest bazą, a który celem
  upscalingu, po `paired_width * paired_height`. To zawodzi, bo node
  nie waliduje, że `width_upscale`/`height_upscale` jest faktycznie
  większe niż `width`/`height`.
- `verbose_store.add_pairing()`: nowy **wymagany** parametr
  `is_upscale_target: bool` (bez wartości domyślnej), zapisywany jako
  `system["is_upscale_target"]`. Docstring zaktualizowany ("four keys").
- `nodes._pair_verbose_entries()`: nowy parametr `b_is_upscale_target`
  (domyślnie `True`). Strona `a` (fp_a / width_a / height_a) dostaje
  `is_upscale_target = not b_is_upscale_target`, strona `b` dostaje
  `b_is_upscale_target`. a/b to nadal czysta symetria FS - który bok
  jest celem upscalingu określa wywołujący.
- Oba wywołania w `MiniMaxH3CLIPCachedFL2VADualRes.execute()` i
  `MiniMaxH3CLIPCachedRef2VADualRes.execute()`: przekazują bazę jako
  `fp1/width/height` i upscale jako `fp2/width_upscale/height_upscale`,
  z jawnym `b_is_upscale_target=True`.
- Testy zaktualizowane / dodane: `tests/test_verbose_store.py`
  (nowy param we wszystkich wywołaniach `add_pairing` + nowy test
  `test_h_add_pairing_records_the_upscale_target_role_on_both_sides`),
  `tests/test_pair_verbose_entries.py` (asercja `is_upscale_target` w
  cross-link i skip-direction + nowy
  `test_b_is_upscale_target_false_swaps_the_role_flags`),
  `tests/test_node_fl2va_dual.py` i `tests/test_node_ref2va_dual.py`
  (asercja `by_res[(1344,768)] is False`, `by_res[(1920,1088)] is True`).
- Weryfikacja: `python -m py_compile` OK; pełny pytest
  **303 passed, 0 skipped, 0 failed** (było 289 przed dual-res pairing,
  wzrost przez commity 5286ce9 + cf731ba). Output w scratchpadzie sesji:
  `pytest_full.txt`.

## Ustalenia istotne dla Chat

### Finalne sygnatury
- `verbose_store.add_pairing(fingerprint: str, cache_dir: Path,
  paired_fingerprint: str, paired_width: int, paired_height: int,
  is_upscale_target: bool) -> None`
  (`minimaxh3_clipcache/verbose_store.py:141`)
- `nodes._pair_verbose_entries(fp_a, width_a, height_a, fp_b, width_b,
  height_b, b_is_upscale_target=True)`
  (`nodes.py:201`)

### Semantyka zapisu
- `system["is_upscale_target"]` jest zapisywane obok trzech dotychczasowych
  kluczy `paired_*` w tym samym wywołaniu `save_verbose()`
  (`minimaxh3_clipcache/verbose_store.py:182`).
- `False` = wpis bazowej rozdzielczości (bok `width`/`height` node'a),
  `True` = wpis rozdzielczości upscale (bok `width_upscale`/
  `height_upscale`). Front-end czyta flagę wprost, bez porównywania
  powierzchni.
- Przy `fp_a == fp_b` (obie rozdzielczości spadły na jeden fingerprint)
  `_pair_verbose_entries` nadal robi natychmiastowy no-op - żaden z kluczy
  `paired_*` ani `is_upscale_target` nie jest zapisywany
  (`nodes.py:242`).
- Nieudany `add_pairing` nadal jest łykany (`try/except` + WARNING
  "VERBOSE PAIRING FAILED"), rdzeń cache obu rozdzielczości pozostaje
  ważny (`nodes.py:250`).

### Zakres zmiany
- `git diff` (commit cf731ba) ogranicza się do dodania `is_upscale_target`
  w `add_pairing`, `_pair_verbose_entries` i dwóch wywołaniach w
  `execute()` + odpowiadające asercje w 4 plikach testowych. Zero innej
  zmiany logiki parowania.
- Brak innych wywołań `add_pairing` / `_pair_verbose_entries` w kodzie
  (sprawdzone grep-em po całym repo).

## Otwarte pytania
- brak (KRYTERIUM_DONE ZLECENIA spełnione: pełny pytest 303/0/0;
  git diff ograniczony do `is_upscale_target`; output zapisany).
- Do sprawdzenia przez użytkownika w żywym UI (CC nie może): czy
  front-end (kolejna faza) faktycznie odczytuje `system.is_upscale_target`
  i renderuje parę bazową + "+ rescaled to WxH" na wpisie upscale.

## Sugestie (nie polecenia)
- Istniejące wpisy cache sprzed commita cf731ba, sparowane przez 5286ce9,
  nie mają pola `is_upscale_target`. Front-end powinien traktować brak
  tego pola jako "nieznana rola" i ewentualnie wrócić do dawnego
  zgadywania po powierzchni tylko dla takich starych par - albo
  zignorować parowanie bez flagi. Świadomie nie dodano tu migracji ani
  backfillu (poza zakresem ZLECENIA).
