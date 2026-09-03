# HANDOFF

## Stan na: 2026-09-03 / branch fix/verbose-hit-generation-size / PR (do otwarcia)

## Ostatnio zrobione

Sidecar verbose odświeża rozmiar generacji przy cache HIT, a Cache Manager
tłumaczy semantykę tego pola. Gałąź `fix/verbose-hit-generation-size`
odcięta od `origin/master` (`94e2044`, po merge PR #11).

### Commit 1 — backend HIT refresh (`448ef9b`)

- `nodes.py` `_sync_verbose_metadata()`: normalny HIT wpisu z kompletnym
  sidecarem nie jest już czystym early-returnem. Odświeża **wyłącznie**
  `system.width` / `system.height` / `system.megapixels`, i tylko gdy
  podany rozmiar różni się od zapisanego:
  * `width`/`height` nie podane -> no-op,
  * rozmiar identyczny -> brak zapisu na dysk,
  * rozmiar inny -> shallow copy istniejącego `system`, nadpisanie tylko
    tych trzech kluczy, `save_verbose()`.
- `prompt`, `created_at`, `references`, `clip_*`, klucze pairingu,
  `comfyui_version` nie są ruszane na HIT. Fingerprint, HIT/MISS i logika
  generacji bez zmian. Cała decyzja + zapis pod istniejącym
  per-fingerprint lockiem, po ponownym sprawdzeniu istnienia core
  `<fp>.json` pod lockiem. Kontrakt bez zmian: never raises.
- Testy w `tests/test_node.py`:
  `test_oa_sync_verbose_hit_refreshes_stale_generation_size_only`,
  `test_ob_sync_verbose_hit_same_generation_size_writes_nothing`.

### Commit 2 — DualRes finalizacja rozmiaru (`94be4da`)

- Commit 1 tworzy regresję dla DualRes ze współdzielonym fingerprintem:
  upscale pass jest HIT-em i przesuwa rozmiar wpisu na stronę B.
- `nodes.py`: nowy helper `_finalize_shared_fingerprint_size()`.
  `_pair_verbose_entries()` w gałęzi `fp_a == fp_b` woła ten helper zamiast
  czystego `return` — re-stampuje trio rozmiaru na BASE (`width_a` /
  `height_a`, `megapixels` przeliczone), zachowuje resztę `system`, nie
  pisze żadnych metadanych pairingu (jeden fingerprint). Te same guardy co
  ścieżka pairingu: `get_lock(fp_a)`, tylko gdy core `<fp_a>.json` istnieje,
  brak zapisu gdy rozmiar już się zgadza, never raises.
- Jeden helper obsługuje FL2VA i Ref2VA DualRes (oba już współdzielą
  `_pair_verbose_entries()`).
- Testy: rozszerzone
  `test_dual_resolution_independent_input_writes_no_pairing` w
  `tests/test_node_fl2va_dual.py` i `tests/test_node_ref2va_dual.py`
  (sprawdzają `(width, height) == (1344, 768)` i `megapixels == 1.03`);
  `tests/test_pair_verbose_entries.py`:
  `test_noop_when_fingerprints_equal` przemianowany na
  `test_noop_when_fingerprints_equal_finalizes_to_base_resolution` +
  nowy `test_shared_fingerprint_finalize_skipped_when_core_entry_gone`.

### Commit 3 — tooltip + docs (`e444182`)

- `web/main.js`: nowy eksportowany helper `generationSizeTooltip(system)`,
  ustawiany jako `title` pola rozmiaru w wierszu listy
  (`h3cm-row-created`) i w panelu szczegółów (`[data-h3cm-detail-created]`).
  Zwraca pusty string (brak tooltipa) gdy wpis nie ma rozmiaru, zgodnie z
  `formatGenerationSize()`. Obliczanie MP po stronie frontendu bez zmian.
  Treść tooltipa: "Resolution of the most recent run that used this entry.
  One cached encode serves every resolution when no keyframes are connected
  -- the encode itself does not depend on width/height."
- `docs/CACHE_MANAGER.md`: nowa sekcja "Generation resolution" — to samo
  zachowanie, plus fakt że data utworzenia i rozmiar pochodzą z dwóch
  różnych momentów, plus że DualRes ze współdzielonym fingerprintem trzyma
  rozmiar BASE.

### Commit 4 — HANDOFF.md (osobno, w tym samym PR)

## Weryfikacja

- Pełny `python -m pytest -q` w comfyenv: **402 passed / 0 failed /
  0 skipped** (4 `DeprecationWarning` z `transformers`, niezwiązane).
  Baseline przed zmianą: 399 passed.
- FAIL-przed / PASS-po dla nowych testów:
  * `test_oa_sync_verbose_hit_refreshes_stale_generation_size_only` —
    przed commit 1 FAIL (`assert 544 == 768`), po commit 1 PASS.
  * `test_ob_sync_verbose_hit_same_generation_size_writes_nothing` —
    guard: przechodzi też na master (stary kod też jest no-op tu),
    dalej PASS po commit 1; łapie regresję "bezwarunkowy zapis na HIT".
  * `test_node_fl2va_dual` / `test_node_ref2va_dual`
    `..._writes_no_pairing` (rozszerzone) — przed commit 2 FAIL
    (`assert (1920, 1088) == (1344, 768)`), po commit 2 PASS.
  * `test_noop_when_fingerprints_equal_finalizes_to_base_resolution` —
    przed commit 2 FAIL (`(1920, 1088) == (1344, 768)`), po commit 2 PASS.
  * `test_shared_fingerprint_finalize_skipped_when_core_entry_gone` —
    guard: przechodzi też przed commit 2 (stary kod to czysty `return`),
    dalej PASS po commit 2.
- `node --check` na kopii `.mjs` z `web/main.js`: czysty.
- Scratchpad harness Node (loader hook stubuje `/scripts/app.js` i
  `/scripts/api.js`, minimalny `document`): moduł importuje się bez
  wyjątku; `generationSizeTooltip` zwraca dokładny tekst tooltipa gdy jest
  rozmiar, `""` gdy brak `width`/`height`; `formatGenerationSize`
  niezmienione (9/9 asercji). Harness w scratchpadzie, niescommitowany.
- `git diff --check` czysty.

## Ustalenia istotne dla Chat

- `system.width` / `.height` / `.megapixels` w sidecarze są czysto
  informacyjne — nie wchodzą do `compute_fingerprint()` ani do decyzji
  HIT/MISS. Bez podpiętych keyframe'ów/referencji jeden fingerprint (jeden
  cache'owany conditioning) obsługuje każdą rozdzielczość.
- Po tej zmianie te trzy pola śledzą **ostatni** run który użył wpisu,
  a `system.created_at` dalej wskazuje moment pierwszego zapisu wpisu —
  dwa różne momenty, świadomie.
- `_sync_verbose_metadata()` (`nodes.py`) — gałąź `not (fresh_miss_written
  or hit_needs_backfill)` robi teraz warunkowy zapis rozmiaru zamiast
  bezwarunkowego `return`.
- `_finalize_shared_fingerprint_size()` (`nodes.py`) — nowy helper,
  wołany tylko z `_pair_verbose_entries()` gałąź `fp_a == fp_b`.
- `generationSizeTooltip()` w `web/main.js` jest eksportowany (jak reszta
  czystych helperów w tym pliku).

## NIE zweryfikowane (do sprawdzenia przez Kamila w żywym ComfyUI)

- Realny render tooltipa po najechaniu na pole rozmiaru w wierszu listy i
  w panelu szczegółów; brak błędów w konsoli przeglądarki.
- End-to-end na żywym serwerze + GPU: DualRes ze współdzielonym
  fingerprintem faktycznie zostawia wpis z rozmiarem BASE po drugim
  (upscale) passie; pojedynczy HIT przy innej rozdzielczości faktycznie
  przesuwa rozmiar w UI po ponownym Check.

## Otwarte pytania

- brak

## Sugestie (nie polecenia)

- Świadomy skutek: pierwszy Check po wdrożeniu może pokazać zmienione
  rozmiary przy istniejących wpisach, jeśli były reużywane w innych MP —
  to oczekiwane, nie bug.
