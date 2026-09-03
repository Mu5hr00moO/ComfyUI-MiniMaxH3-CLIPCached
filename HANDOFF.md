# HANDOFF

## Stan na: 2026-09-03 / branch feat/ref-provenance-from-graph / PR #14 (otwarty)

## Ostatnio zrobione

Backend proweniencji referencji dla obu węzłów Ref2VA
(`MiniMaxH3CLIPCachedRef2VA`, `MiniMaxH3CLIPCachedRef2VADualRes`) + trzy
poprawki po review botach na PR #14 + poprawka pokrycia reguły liściowej.
Zero zmian w mechanice H3, w
`compute_fingerprint()` ani w decyzji HIT/MISS. UI to osobna, późniejsza
faza — tu wyłącznie zapis do sidecara.

### Commit 1 — pierwotny backend (`b88890d`)

- Nowy moduł `minimaxh3_clipcache/provenance.py`:
  `collect_ref_sources(prompt, unique_id)` przechodzi graf prompta
  (format API) wstecz od każdego wejścia `ref_*` węzła.
- Oba węzły Ref2VA dostały blok `"hidden"` z `"PROMPT"` i `"UNIQUE_ID"`.
  Klucz to `prompt_graph`, NIE `prompt` (kolizja z wymaganym wejściem
  tekstowym `prompt`; ComfyUI podaje hidden po nazwie przez `f(**inputs)`).
- `nodes._sync_ref_sources(proxy, prompt_graph, unique_id)` dopisuje wynik
  do `system.ref_sources` sidecara, tuż po `_sync_verbose_metadata`, pod
  `get_lock(fingerprint)`, z re-checkiem `<fp>.json`, każdy wyjątek
  połknięty jako WARNING.

### Commit 2 — hardening po review (`c2c0100`)

Trzy niezależnie potwierdzone znaleziska greptile / CodeRabbit:

1. **Reguła liściowa** zastąpiła "najbliższy literał media wygrywa".
   Literał liczy się jako źródło referencji TYLKO na liściu grafu — node
   bez żadnego wejścia będącego linkiem (prawdziwy loader). Literał
   wyglądający na plik media na node'ie pośrednim (widget tekstowy
   `note="shallow.png"` na pass-through) jest ignorowany, walk schodzi
   dalej przez link. To test strukturalny, nie biała lista `class_type` —
   działa dla `VHS_LoadVideo` i customowych loaderów.
   `_MEDIA_EXTENSIONS` zostało jako wtórne zawężenie w obrębie liścia
   (odcina np. `.pth` upscalera na `UpscaleModelLoader`), nie jako główny
   filtr.

2. **Wartość zawsze listą.** `collect_ref_sources()` zwraca
   `{slot: [ {annotated[, path]}, ... ]}`. Gdy kilka loaderów wpada do
   jednej referencji na tym samym poziomie grafu (dwa `LoadImage` ->
   `ImageBatch` -> ref) — zapisywani są WSZYSCY, w kolejności BFS. Walk
   zatrzymuje się na pierwszej głębokości, która daje trafienie liściowe;
   głębszy loader z dłuższej ścieżki nie jest raportowany.

3. **None vs {}.** `collect_ref_sources()` zwraca `None` gdy walk nie mógł
   się wykonać (brak grafu, brak `unique_id`, nasz node nieobecny,
   wyjątek) i `{}` gdy wykonał się czysto i nic nie znalazł.
   `_sync_ref_sources()`:
   - `None` → no-op, istniejące `system.ref_sources` zostaje nietknięte;
   - `{}` → pod tą samą blokadą USUWA `system.ref_sources` jeśli jest
     (i tylko wtedy zapisuje sidecar);
   - niepusty → jak dotąd, zapis o ile się różni.
   Naprawia: ten sam fingerprint (identyczne tensory refów) raz z
   `LoadImage`, raz z nieprześledzalnego grafu — sidecar nie pokazuje już
   dalej starej nazwy pliku.

### Commit 3 — HANDOFF.md (osobno, w tym samym PR, `7fd5c97`)

### Commit 4 — pokrycie reguły liściowej (`82a44cd`)

`_walk_back_for_media_filenames()` (provenance.py) zatrzymywał się na
pierwszej głębokości BFS z trafionym liściem. Przy asymetrycznym
fan-inie gubiło to realne źródła:
`ImageBatch(a <- LoadImage "SHALLOW.png", b <- ImageScale <- LoadImage
"DEEPER.png")` zwracał dziś tylko `SHALLOW.png`.

Walk przechodzi teraz CAŁY osiągalny podgraf wstecz i zwraca nazwę pliku
z KAŻDEGO napotkanego liścia-loadera. Bez zmian: reguła liściowa,
cycle-safety przez `visited`, kolejność BFS (płycej przed głębiej, w
obrębie poziomu wg kolejności kluczy w prompcie). Dedup: ta sama nazwa
pliku z kilku liści → zwracana raz, pierwsze wystąpienie.

Uzasadnienie: nadmiarowi kandydaci w obrębie jednego poziomu byli już
świadomie zaakceptowani (composite z maską → DEST/SRC/MASK razem);
ograniczanie do jednego poziomu było niekonsekwentne.

**Zmiana kontraktu testu:** `test_nearer_leaf_stops_the_walk_before_a_
deeper_leaf` asertował stare zachowanie "stop na pierwszej głębokości" i
został zastąpiony przez `test_asymmetric_fan_in_collects_a_leaf_from_
every_depth` (oczekiwane obie nazwy, SHALLOW przed DEEPER). Żaden inny
test nie zmienił kontraktu.

### Commit 5 — HANDOFF.md (osobno, w tym samym PR)

## Weryfikacja

- Pełny `conda run -n comfyenv python -m pytest -q`: **435 passed / 0
  failed / 0 skipped** (4 `DeprecationWarning` z `transformers`,
  niezwiązane). Baseline przed hardeningiem: 427.
- `test_provenance.py` (20 testów): reguła liściowa, listy wartości,
  `None`/`{}`, wiele liści, asymetryczny fan-in zbiera z każdej
  głębokości, dedup nazwy, liść bez pliku nie blokuje głębszego, cykl
  kończy się.
- `test_node_ref2va.py`: `test_w2` (czysty pusty walk kasuje stare
  `ref_sources`), `test_w3` (`prompt_graph=None` zostawia pole),
  `test_w4` (wiele liści → wielopozycyjna lista); `test_v`/`test_z3`
  pod listę.
- `python -m py_compile` na zmienionych plikach: czysto.
- `git diff --check`: czysto.

## Ustalenia istotne dla Chat

- `system.ref_sources` w sidecarze — nowe, opcjonalne pole. Dict
  kluczowany nazwą slotu (`ref_image_0`, `ref_video_2`,
  `ref_video_audio_1`, `ref_audio_0`, …). Wartość: **lista**
  `[{"annotated": str, "path"?: str}, ...]` (min. 1 element) — po jednym
  wpisie na KAŻDY osiągalny liść-loader w podgrafie danego refa, w
  kolejności BFS, zdeduplikowane po nazwie pliku. Wyłącznie informacyjne;
  nie w fingerprincie.
  `provenance.py:collect_ref_sources` — `None` = walk niemożliwy,
  `{}` = walk czysty ale pusty, dict = trafienia.
- `provenance.collect_ref_sources` nigdy nie rzuca; `_sync_ref_sources`
  nigdy nie rzuca — awaria warstwy proweniencji nie może zepsuć
  zwróconego cond/latent.
- `MiniMaxH3CLIPCachedRef2VA` / `...Ref2VADualRes` — `INPUT_TYPES()` ma
  `"hidden": {"unique_id": "UNIQUE_ID", "prompt_graph": "PROMPT"}`
  (`nodes.py` `_ref2va_hidden_input_spec`). Deklaracja `UNIQUE_ID`
  włącza `include_unique_id_in_input()` → node_id wchodzi do sygnatury
  cache'a WYKONANIA w RAM. Zaakceptowane: cache dyskowy jest kluczowany
  fingerprintem enkodowania, przebudowany graf i tak trafia HIT z dysku.

## NIE zweryfikowane (do sprawdzenia przez Kamila w żywym ComfyUI)

- Realny przebieg przez `/prompt` API: czy `prompt_graph`/`unique_id`
  docierają jako hidden do `execute()` obu węzłów i czy
  `system.ref_sources` pojawia się w `<fp>.verbose.json` po MISS z
  podpiętym `LoadImage`/`LoadAudio`/`VHS_LoadVideo` — w nowym kształcie
  (lista) i z regułą liściową.
- Czy realne węzły ładujące w tym środowisku są liśćmi grafu (brak
  wejść-linków) i czy nazwa pliku ma rozszerzenie z `_MEDIA_EXTENSIONS`.

## Otwarte pytania

- brak

## Sugestie (nie polecenia)

- Faza UI: w Cache Managerze złączyć `system.ref_sources[slot]` (lista) z
  wierszem referencji po kluczu slotu i pokazać `annotated` + (jeśli
  jest) `path` jako trop do oryginału; obsłużyć >1 wpis na slot
  (asymetryczny fan-in, composite z maską — realnie kilka plików na
  jeden ref).
