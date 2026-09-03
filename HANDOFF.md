# HANDOFF

## Stan na: 2026-09-03 / branch feat/ref-provenance-from-graph / PR #14 (otwarty)

## Ostatnio zrobione

Backend proweniencji referencji dla obu węzłów Ref2VA
(`MiniMaxH3CLIPCachedRef2VA`, `MiniMaxH3CLIPCachedRef2VADualRes`) + trzy
poprawki po review botach na PR #14. Zero zmian w mechanice H3, w
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

### Commit 3 — HANDOFF.md (osobno, w tym samym PR)

## Weryfikacja

- Pełny `conda run -n comfyenv python -m pytest -q`: **433 passed / 0
  failed / 0 skipped** (4 `DeprecationWarning` z `transformers`,
  niezwiązane). Baseline przed hardeningiem: 427.
- `test_provenance.py` przepisane pod nowy kontrakt (18 testów): reguła
  liściowa, listy wartości, `None`/`{}`, wiele liści na tym samym
  poziomie, bliższy liść zatrzymuje walk przed głębszym.
- `test_node_ref2va.py`: nowe `test_w2` (czysty pusty walk kasuje stare
  `ref_sources`), `test_w3` (`prompt_graph=None` zostawia pole),
  `test_w4` (wiele liści → wielopozycyjna lista); `test_v`/`test_z3`
  zaktualizowane pod listę.
- `python -m py_compile` na zmienionych plikach: czysto.
- `git diff --check`: czysto.

## Ustalenia istotne dla Chat

- `system.ref_sources` w sidecarze — nowe, opcjonalne pole. Dict
  kluczowany nazwą slotu (`ref_image_0`, `ref_video_2`,
  `ref_video_audio_1`, `ref_audio_0`, …). Wartość: **lista**
  `[{"annotated": str, "path"?: str}, ...]` (min. 1 element).
  Wyłącznie informacyjne; nie w fingerprincie.
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
  jest) `path` jako trop do oryginału; obsłużyć >1 wpis na slot.
