# HANDOFF

## Stan na: 2026-09-03 / branch feat/ref-provenance-from-graph / PR do master

## Ostatnio zrobione

Backend proweniencji referencji dla obu węzłów Ref2VA
(`MiniMaxH3CLIPCachedRef2VA`, `MiniMaxH3CLIPCachedRef2VADualRes`). Gałąź
odcięta od `origin/master` (`653707e`, po merge PR #13). Zero zmian w
mechanice H3, w `compute_fingerprint()` ani w decyzji HIT/MISS. UI to
osobna, późniejsza faza — tu wyłącznie zapis do sidecara.

### Commit 1 — kod + testy (`b88890d`)

- Nowy moduł `minimaxh3_clipcache/provenance.py`:
  `collect_ref_sources(prompt, unique_id)` przechodzi graf prompta
  (format API) wstecz od każdego wejścia `ref_*` węzła i zwraca
  `{nazwa_slotu: {"annotated": <surowa wartość z prompta>[, "path":
  <abs>]}}`. Przejście jest BFS ("najbliższy" = najmniej skoków),
  odporne na cykl (każdy node id odwiedzany raz). Literał liczy się jako
  źródło tylko gdy jego rozszerzenie należy do listy kontenerów
  obraz/wideo/audio — to odcina np. nazwę modelu upscalera `.pth` leżącą
  na tej samej ścieżce grafu. `path` =
  `folder_paths.get_annotated_filepath(annotated)`; gdy rzuci wyjątkiem
  albo `folder_paths` niedostępne — pole `path` pomijane, `annotated`
  zostaje. Slot bez trafienia (gałąź kończy się na
  `VAEDecode`/`EmptyLatentImage`) jest pomijany w całości. Funkcja nigdy
  nie rzuca.
- Klucz wyniku to NAZWA SLOTU, nie pozycja — świadomie, bo lista
  `system.references` w sidecarze jest w kolejności montażu stockowego
  węzła i zipowanie pozycyjne rozjechałoby się przy pustym slocie w
  środku. Późniejsza faza UI łączy po kluczu slotu.
- Oba węzły Ref2VA dostały blok `"hidden"` z `"PROMPT"` i `"UNIQUE_ID"`.
  Klucz to `prompt_graph`, NIE `prompt` — węzły mają już wymagane wejście
  tekstowe `prompt`, a ComfyUI podaje wejścia (w tym hidden) po nazwie
  przez `f(**inputs)`, więc `"prompt": "PROMPT"` nadpisałoby tekst
  prompta dictem grafu. FL2VA i FL2VADualRes NIE ruszane (nie mają
  referencji) — jest test-strażnik.
- `nodes._sync_ref_sources(proxy, prompt_graph, unique_id)` woła helper i
  dopisuje wynik do `system.ref_sources` sidecara, tuż po
  `_sync_verbose_metadata`, z tą samą dyscypliną: całość pod
  `get_lock(fingerprint)`, re-check że `<fp>.json` nadal istnieje, każdy
  wyjątek połknięty jako WARNING, nigdy nie rzuca. Pusty wynik helpera →
  pole nie jest zapisywane. Brak bloku `system` (np. czysty HIT sidecara
  sprzed tej funkcji) → no-op (tworzenie bloku to rola
  `_sync_verbose_metadata`). Wszystkie 3 call-site'y
  `_execute_ref2va_once` przekazują `prompt_graph`/`unique_id` (single +
  base + upscale w dual).

### Commit 2 — HANDOFF.md (osobno, w tym samym PR)

## Weryfikacja

- Pełny `conda run -n comfyenv python -m pytest -q`: **427 passed / 0
  failed / 0 skipped** (4 `DeprecationWarning` z `transformers`,
  niezwiązane). Baseline przed zmianą: 402. Nowe: 15 w
  `tests/test_provenance.py` (bezpośredni LoadImage; łańcuch przez node
  pośredni; gałąź bez loadera → brak wpisu; luka w numeracji slotów;
  cykl; brak `unique_id` w prompcie; `get_annotated_filepath` rzuca →
  zostaje `annotated`; wideo/audio jak obrazy; literał `.pth` na ścieżce
  ignorowany), 8 w `test_node_ref2va.py` (`_sync_ref_sources` +
  end-to-end przez `execute()` z hidden), 1 w `test_node_ref2va_dual.py`
  (oba węzły Ref2VA mają hidden `prompt_graph`/`unique_id`), 1 w
  `test_node_fl2va_dual.py` (węzły FL2VA nie mają bloku `hidden`).
- `python -m py_compile` na wszystkich zmienionych plikach: czysto.
- `git diff --check`: czysto.

## Ustalenia istotne dla Chat

- `MiniMaxH3CLIPCachedRef2VA` / `...Ref2VADualRes` — `INPUT_TYPES()` ma
  teraz `"hidden": {"unique_id": "UNIQUE_ID", "prompt_graph": "PROMPT"}`
  (`nodes.py` `_ref2va_hidden_input_spec`). Deklaracja `UNIQUE_ID`
  włącza `include_unique_id_in_input()` w ComfyUI → node_id wchodzi do
  sygnatury cache'a WYKONANIA w RAM. Zaakceptowane: nasz cache dyskowy
  jest kluczowany fingerprintem enkodowania, więc przebudowany graf i
  tak trafia HIT z dysku i enkoder (~27 GB) nadal się nie ładuje.
- `system.ref_sources` w sidecarze — nowe, opcjonalne pole. Dict
  kluczowany nazwą slotu (`ref_image_0`, `ref_video_2`,
  `ref_video_audio_1`, `ref_audio_0`, …), wartość
  `{"annotated": str, "path"?: str}`. Wyłącznie informacyjne; nie w
  fingerprincie. `scanner.py`/`routes.py` przepuszczają je bez zmian
  (cały obiekt `verbose` idzie do API).
- `provenance.collect_ref_sources` nigdy nie rzuca; `_sync_ref_sources`
  nigdy nie rzuca — awaria warstwy proweniencji nie może zepsuć
  zwróconego cond/latent.

## Odchylenie od zlecenia

- ZLECENIE mówiło dosłownie `"hidden": {..., "prompt": "PROMPT"}`.
  Użyto `prompt_graph` zamiast `prompt`, bo `"prompt"` kolidowałoby z
  istniejącym wymaganym wejściem tekstowym `prompt` (ComfyUI: hidden
  przypisywane po required w `input_data_all`, potem `f(**inputs)` —
  tekst prompta zostałby nadpisany dictem grafu). Zmiana konieczna dla
  poprawności.

## NIE zweryfikowane (do sprawdzenia przez Kamila w żywym ComfyUI)

- Realny przebieg przez `/prompt` API: czy `prompt_graph`/`unique_id`
  faktycznie docierają jako hidden do `execute()` obu węzłów i czy
  `system.ref_sources` pojawia się w `<fp>.verbose.json` po MISS z
  podpiętym `LoadImage`/`LoadAudio`/`VHS_LoadVideo`.
- Czy realne węzły ładujące w tym środowisku (nazwa wejścia z nazwą
  pliku) są łapane przez listę rozszerzeń mediów w `provenance.py`.

## Otwarte pytania

- brak

## Sugestie (nie polecenia)

- Faza UI: w Cache Managerze złączyć `system.ref_sources[slot]` z
  wierszem referencji po kluczu slotu i pokazać `annotated` + (jeśli
  jest) `path` jako trop do odnalezienia oryginału.
