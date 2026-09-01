# HANDOFF

## Stan na: 2026-09-01 / branch master / commit 94130b0

## Ostatnio zrobione (store.py hardening przed v0.1.0 — z ZLECENIA, audyt Grok+Codex)

Dwa niezależnie zweryfikowane znaleziska audytu, referencja master@83629c7.
Trzy commity:

### 1. `_SAFETENSOR_READ_ERRORS` zawężone (commit 3b7e95e)
- Przed: `(SafetensorError, OSError, RuntimeError)`. `RuntimeError` był
  niezweryfikowany — test `test_d_load_file_runtimeerror_is_a_clean_miss`
  explicite asercjował, że dowolny `RuntimeError` z `load_file()` staje
  się cichym MISS-em.
- Zbadano empirycznie (skrypt diagnostyczny w scratchpadzie, NIE
  scommitowany — jednorazowy probe, nie test regresyjny): 13 scenariuszy
  korupcji `.safetensors` (pusty plik, obcięcie na każdym etapie nagłówka,
  absurdalny prefiks długości nagłówka, niepoprawny JSON w nagłówku,
  nieznany dtype, niezgodność shape/data_offsets, ujemny wymiar shape,
  odwrócone data_offsets, brakujące pole dtype, bit-flip w danych tensora,
  brakujący plik, katalog zamiast pliku) — zawsze `SafetensorError` albo
  `OSError`, nigdy `RuntimeError`. Inspekcja źródeł zainstalowanego
  `safetensors/torch.py` (0.8.0) potwierdza: `load_file()` to cienki
  wrapper `safe_open(...).get_tensors()`; jedyne trzy miejsca rzucające
  `RuntimeError` w tym module są w `load_model()`/`save_file()` (ścieżki
  nieużywane przez nasz kod odczytu).
- `_SAFETENSOR_READ_ERRORS = (SafetensorError, OSError)` + obszerny
  komentarz WHY przy stałej (`store.py:54-77`).
- Stary test zastąpiony (nie tylko dopisany obok):
  `test_d_load_file_safetensorerror_is_a_clean_miss` (potwierdzona
  korupcja nadal MISS-em) + 3 nowe testy że NIEPOWIĄZANY `RuntimeError`
  PROPAGUJE się (nie staje MISS-em) z trzech miejsc dzielących tę stałą:
  `load_file()`, `safe_open()` w `load_conditioning()`, `safe_open()` w
  `inspect_conditioning_pair()`.

### 2. Walidacja formatu `generation_id` przed porównaniem (commit b36c1bb)
- Przed: `load_conditioning()` (~linia 249) i `inspect_conditioning_pair()`
  (~linia 175) sprawdzały tylko obecność klucza `"generation_id" in
  payload`, potem porównywały `!=` wprost. Spreparowany JSON
  `{"generation_id": null}` + `.safetensors` bez klucza
  `cache_generation_id` w metadata (więc `.get()` też zwraca `None`)
  przechodziły jako dopasowanie (`None != None -> False`).
- Dodano `_is_valid_generation_id()`: wymaga niepustego stringa
  pasującego do `^[0-9a-f]{32}$` (dokładny kształt `uuid.uuid4().hex`).
  Obie strony (JSON i metadata safetensors) walidowane PRZED
  porównaniem, w obu funkcjach. Niepoprawna wartość (None, brak pola,
  pusty string, nie-string, zły format/case/długość/myślniki) -> jawny
  MISS z konkretnym logiem w `load_conditioning()`, `"invalid_json_envelope"`
  (strona JSON) lub `"generation_mismatch"` (strona safetensors) w
  `inspect_conditioning_pair()` — reużyte istniejące kody powodów, ŻADNYCH
  zmian w Cache Managerze/web/main.js w tym zadaniu (poza kolateralną
  poprawką testu, patrz niżej).
- ~15 nowych testów parametryzowanych (`store.py:54` docstring) w
  `test_store.py`: None/None (dokładny scenariusz z audytu), brak pola,
  pusty string, nie-string (int/list), malformed-niepusty, non-hex chars,
  za krótki, uppercase, prawdziwy uuid4 z myślnikami — dla obu funkcji,
  po obu stronach (JSON i safetensors metadata).

### 3. Kolateralna poprawka testu (commit 94130b0)
- `tests/test_scanner.py::test_i_generation_mismatch_is_reported_as_inconsistent`
  używał `"torn-refresh-generation"` (niepoprawny format) do
  przetestowania ścieżki `generation_mismatch` przez `scan_cache()` ->
  `inspect_conditioning_pair()`. Po zmianie #2 ta wartość jest łapana
  wcześniej jako `invalid_json_envelope`. Podmieniono na
  poprawny-formatowo-ale-inny uuid4 hex (ten sam wzorzec co analogiczna
  poprawka w `test_store.py`). Jedyny plik poza `store.py`/
  `tests/test_store.py` dotknięty w tej sesji — konieczna konsekwencja
  zmiany #2, nie scope creep.

## Ustalenia istotne dla Chat
- `python -m py_compile` na wszystkich zmienionych plikach — OK.
- `git diff --check` — czysty, po każdym z 3 commitów.
- Pełny pytest: **349 passed, 0 skipped, 0 failed** (przed sesją: 348,
  jeden test naprawiony zamiast dodany netto +1 zbiorczo licząc nowe
  testy minus brak zmiany liczby plików testowych — patrz commity dla
  dokładnych liczb per plik).
- `git diff --stat` całej sesji (83629c7..HEAD): `minimaxh3_clipcache/store.py`
  (+80/-2), `tests/test_scanner.py` (+8/-1), `tests/test_store.py`
  (+197/-4) — dokładnie w zakresie store.py + jego testy, plus jeden
  wymuszony plik testowy.
- Diagnostyczne skrypty probe (`probe_safetensors_errors.py`,
  `probe_safetensors_errors2.py`) zostały w scratchpadzie sesji, NIE
  scommitowane — jednorazowe narzędzie do zbadania faktycznych wyjątków
  safetensors 0.8.0, nie część projektu.
- Commity tej sesji: 3b7e95e (zawężenie wyjątków), b36c1bb (walidacja
  generation_id), 94130b0 (poprawka test_scanner.py).

## Otwarte pytania
- brak.

## Sugestie (nie polecenia)
- brak nowych w tej sesji.
