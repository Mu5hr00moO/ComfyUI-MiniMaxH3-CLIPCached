# Handoff — feat/fl2va-file-provenance

- Branch: `feat/fl2va-file-provenance` (baza: `origin/master` @ 9620500)
- HEAD: 9620500 (brak commita — drzewo brudne, zmiany niezacommitowane)
- Dirty: tak — 5 plików zmienionych, 0 commitów

## Stan bieżący

Faza 1 zaimplementowana. Ten sam ślad loadera co Ref2VA (annotated + path,
klucz = nazwa slotu) działa teraz dla `first_frame` / `last_frame` na obu
node'ach FL2VA. Jeden collector, jeden walker, jeden hidden-input-spec helper.

### provenance.py
- Nagłówek modułu i docstring `collect_ref_sources` zaktualizowane —
  obejmują teraz slot y FL2VA, nie mówią już "Ref2VA only".
- Nowy `_FRAME_INPUT_KEYS = {"first_frame", "last_frame"}` + predykat
  `_is_traced_input_key(key)` (prefiks dla `ref_*`, exact-match dla keyframe).
- `collect_ref_sources` używa predykatu zamiast gołego `key.startswith(...)`.
- `_walk_back_for_media_filenames` NIETKNIĘTY (generyczny, nie zna nazw kluczy).

### nodes.py
- `_ref2va_hidden_input_spec` → `_provenance_hidden_input_spec` (rename, nie
  duplikat). Docstring rozszerzony; dodane wprost, że `compute_fingerprint()`
  nie bierze node id, więc uzasadnienie "harmless RAM-cache only" przenosi
  się 1:1 na FL2VA. Zweryfikowane w kodzie: `fingerprint.py:124` —
  brak parametru unique_id / node id.
- `_sync_ref_sources` docstring: usunięte "on the Ref2VA path only".
- `_build_references` docstring + komentarz (nodes.py:59-77): usunięte
  kłamstwo "FL2VA sidecars never hold system.ref_sources". Teraz: rekordy
  reference FL2VA nie mają `slot`, a `system.ref_sources` dla FL2VA jest
  kluczowane nazwami label `first_frame`/`last_frame` — Cache Manager łączy
  FL2VA po `label`, Ref2VA po `slot`.
- `_execute_fl2va_once`: nowe opcjonalne kwargs `prompt_graph=None,
  unique_id=None`; wywołanie `_sync_ref_sources(proxy, prompt_graph,
  unique_id)` tuż po `_sync_verbose_metadata`, analogicznie do
  `_execute_ref2va_once`.
- `MiniMaxH3CLIPCachedFL2VA` i `MiniMaxH3CLIPCachedFL2VADualRes`:
  `"hidden": _provenance_hidden_input_spec()` w INPUT_TYPES; `execute()`
  przyjmuje `prompt_graph`/`unique_id` i przekazuje do `_execute_fl2va_once`
  (dual — do obu wywołań).

### Testy
- `tests/test_provenance.py`: +6 testów sekcji FL2VA (first/last frame z
  path, jeden keyframe, chain przez node pośredni, gałąź bez loadera → `{}`,
  predykat exact-match odrzuca `middle_frame`/`first_frames`).
- `tests/test_node.py`: +3 (end-to-end MISS zapisuje `ref_sources` pod
  `first_frame`/`last_frame`; brak hidden → brak `ref_sources`; node
  deklaruje wspólny hidden block). Zaktualizowany nieaktualny komentarz w
  `test_l` (mówił, że sidecar FL2VA "never carries system.ref_sources").
- `tests/test_node_fl2va_dual.py`: +2 (dual wpisuje `ref_sources` do obu
  sidecarów; brak hidden → brak). Przepisany
  `test_fl2va_nodes_declare_no_hidden_inputs` → teraz sprawdza, że oba
  node'y FL2VA deklarują wspólny blok (poprzednia wersja kłamałaby).

## Nieuruchomione

`pytest` nie jest zainstalowany w Pythonie tej sesji, a `torch` jest
niedostępny (cała sucia testowa importuje pakiet → `nodes.py` →
`comfy.model_management` → `torch`). Logika `collect_ref_sources` dla slotów
FL2VA zweryfikowana osobnym importem modułu (5 scenariuszy, wszystkie
przechodzą). `py_compile` czysty dla wszystkich zmienionych plików.
Pełny `pytest tests/` do odpalenia w środowisku z torch.

## Następne kroki

- Odpalić `pytest tests/test_provenance.py tests/test_node.py
  tests/test_node_fl2va_dual.py tests/test_node_ref2va.py
  tests/test_node_ref2va_dual.py` (regresja Ref2VA zielona = brak zmiany
  zachowania).
- Commit + `HANDOFF` commit na tej gałęzi, push, PR.
- Faza 2 (poza zakresem): web/ / Cache Manager UI — join `first_frame` /
  `last_frame` z `system.ref_sources` po stronie frontendu.
