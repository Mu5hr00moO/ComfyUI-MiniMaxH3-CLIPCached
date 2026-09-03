# HANDOFF

## Stan na: 2026-09-04 / branch feat/reference-slot-names / PR (do otwarcia)

## Ostatnio zrobione

Backend: nazwa slotu wejściowego dla każdej referencji Ref2VA trafia do
`system.references` w sidecarze `<fp>.verbose.json`, jako pole `"slot"`
obok istniejącego `"index"`. Cel: Cache Manager złącza referencję z
`system.ref_sources` (PR #14, kluczowane nazwą slotu) po tej nazwie,
zamiast odtwarzać kompakcję pozycyjnie w JS. Zero zmian w mechanice H3,
w `compute_fingerprint()`, w decyzji HIT/MISS. UI to osobna, późniejsza
faza — tu wyłącznie zapis do sidecara.

### Commit 1 — kod + testy (`745cb40`)

- `_build_reference_items()` (`nodes.py`) zwraca teraz krotki
  3-elementowe `(type, tensor_or_None, slot)` zamiast 2-elementowych.
  Nazwa slotu jest brana z klucza, po którym funkcja i tak iteruje
  (`sorted(... , key=_slot_index)`).
- Marker ścieżki dźwiękowej wideo dostaje nazwę SWOJEGO wejścia AUDIO
  (`ref_video_audio_<N>`), nie `ref_video_<N>` wideo — to dwa osobne
  wejścia na węźle.
- `_build_references()` zapisuje `"slot"` obok `"index"`. Komentarz WHY:
  `index` = pozycja w spłaszczonym batchu widzianym przez encoder, PO
  wycięciu pustych slotów; `slot` = nazwa wejścia na węźle, jedyny
  wspólny klucz z `system.ref_sources`.
- FL2VA bez zmian: jego itemy zostają 2-krotkami (`first_frame` /
  `last_frame` to stałe nazwane wejścia, już zapisane jako `"label"`, a
  jego sidecary nigdy nie mają `system.ref_sources`), więc `"slot"` nie
  powstaje. `_build_references()` znosi obie długości krotki
  (`item[2] if len(item) > 2 else None`).

### Commit 2 — HANDOFF.md (osobno, w tym samym PR)

## Weryfikacja

- Pełny `conda run -n comfyenv python -m pytest -q`: **438 passed / 0
  failed / 0 skipped** (4 `DeprecationWarning` z `transformers`,
  niezwiązane). Baseline przed zmianą: 435.
- `python -m py_compile nodes.py tests/test_node.py
  tests/test_node_ref2va.py`: czysto.
- `git diff --check`: czysto.
- Nowe / zmienione testy (`tests/test_node_ref2va.py`,
  `tests/test_node.py`):
  - `test_k`/`test_l`/`test_m`/`test_n`/`test_o` — dostrojone do
    3-krotki, asertują też nazwę slotu; marker audio ma
    `ref_video_audio_1`, nie `ref_video_1`; standalone audio ma
    `ref_audio_<N>`.
  - `test_o2` — luka w numeracji slotów (`ref_image_0` + `ref_image_2` +
    `ref_image_5` → dokładnie te trzy nazwy przy indeksach 0,1,2).
  - `test_q` — `_sync_verbose_metadata` przepisuje `slot` do sidecara.
  - `test_q2` — item bez slotu (stary kształt / FL2VA) → deskryptor bez
    klucza `"slot"`, odczyt się nie wywraca.
  - `test_z5` — end-to-end `execute()` z luką w slotach obrazów →
    `references` niosą poprawny `slot` obok `index`.
  - `test_l` (FL2VA, `test_node.py`) — asertuje brak `"slot"` w
    referencjach FL2VA.

## Ustalenia istotne dla Chat

- `system.references[i]` w sidecarze dostaje NOWE, opcjonalne pole
  `"slot"` (string, np. `ref_image_0`, `ref_video_1`,
  `ref_video_audio_2`, `ref_audio_0`) — TYLKO dla wpisów Ref2VA. FL2VA
  wpisy go nie dostają (mają `"label"` = `first_frame`/`last_frame`).
  — `nodes.py` `_build_references` / `_build_reference_items`.
- `"slot"` != `"index"`. `"index"` = pozycja po kompakcji pustych slotów
  (co widzi encoder). `"slot"` = nazwa wejścia na węźle. Rozjeżdżają się
  gdy slot w środku jest pusty. `"slot"` to jedyny klucz wspólny z
  `system.ref_sources` (też kluczowanym nazwą slotu).
- Marker ścieżki dźwiękowej wideo (`{"type": "audio", ...}` wstawiany
  PRZED wideo) ma `slot` = `ref_video_audio_<N>` (nazwa wejścia AUDIO),
  nie `ref_video_<N>`.
- Kompatybilność wstecz: istniejące sidecary nie mają `"slot"`, nic nie
  jest migrowane. `scanner.py` (`scan_cache`) i `routes.py` (`/get`,
  `/check`) przepuszczają cały blok `verbose` bez introspekcji
  pojedynczych wpisów `references` — brak pola `"slot"` jest po stronie
  odczytu backendu nie-zdarzeniem. Zweryfikowane przez przegląd
  `scanner.py:119-147` i `routes.py:138-146` (oba tylko `load_verbose` /
  `web.json_response(verbose)`).
- `_build_references` nadal znosi wejście 2-krotkowe (FL2VA + starsze
  testy) — pole `"slot"` powstaje tylko gdy item ma 3. element != None.

## NIE zweryfikowane (do sprawdzenia przez Kamila w żywym ComfyUI)

- Realny przebieg przez `/prompt` API: czy po MISS z podpiętymi
  `LoadImage`/`LoadAudio` w slotach `ref_*` (z luką) `<fp>.verbose.json`
  ma `system.references[i].slot` zgodne z kluczami
  `system.ref_sources`.

## Otwarte pytania

- brak

## Sugestie (nie polecenia)

- Faza UI: w `web/main.js` złączyć `system.ref_sources[ref.slot]` z
  wierszem/panelem referencji po `ref.slot` (fallback dla starych wpisów
  bez `slot`: dotychczasowe zachowanie pozycyjne albo pominięcie tropu).
  To domyka pierwotny powód istnienia pola.
