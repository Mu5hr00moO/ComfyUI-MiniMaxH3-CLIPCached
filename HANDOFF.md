# HANDOFF

## Stan na: 2026-09-01 / branch master / commit c161d9c

## Ostatnio zrobione (audyt runda 2 -- warstwa trwałości, dwie niezależne poprawki poprawności)

Dwa osobne commity, dwie różne przyczyny, ten sam gatunek co pierwsza sesja
hardeningowa store.py. Pełny suite: 397 passed (było 379). py_compile /
`git diff --check` czyste.

### Część A -- `6e0bf8a` store: zawężenie OSError w read-gate do FileNotFoundError

Codex MEDIUM #2. `_SAFETENSOR_READ_ERRORS` oraz dwa miejsca odczytu JSON
(`load_conditioning` + `inspect_conditioning_pair`) łapały goły `OSError`,
więc EMFILE / EACCES / EIO / ENOSPC zamieniały się w cichy cache MISS i
zbędny ~27 GB re-encode -- ta sama klasa za-szerokiego łapania co
`RuntimeError` usunięty w rundzie 1, tylko dla innego typu.

- Empiryczne re-probowanie zainstalowanego safetensors 0.8.0 (skrypt w
  scratchpadzie, nie w repo): jedyny `OSError` znaczący "plik tego wpisu
  naprawdę zniknął, re-encode to naprawia" to `FileNotFoundError` --
  podnoszony jednolicie przez `open()` (przez `read_bytes()`),
  `safe_open()` i `load_file()` gdy plik znika między strażnikiem
  `.exists()` a odczytem (race z Cache Manager Delete albo przerwany
  zapis).
- Każda awaria zasobowa/uprawnieniowa ma inny typ: goły `OSError` dla
  EMFILE/EIO/ENOSPC oraz `safe_open()`-na-katalogu ("No such device"),
  `PermissionError` dla EACCES/EPERM, `IsADirectoryError` z `read_bytes()`
  na katalogu. Dopasowanie na poziomie klasy wystarcza -- zero inspekcji
  errno.
- Korupcja treści bez zmian: to `SafetensorError` albo `ValueError` (zły
  JSON / nie-UTF-8), nie `OSError`, dalej czysty MISS.
- `_SAFETENSOR_READ_ERRORS = (SafetensorError, FileNotFoundError)`.
- 15 nowych testów w `tests/test_store.py`: symulowany
  EMFILE/EACCES/EIO/ENOSPC/katalog `OSError` propaguje się ze wszystkich
  czterech miejsc odczytu; prawdziwy race zniknięcia pliku (unlink między
  `.exists()` a `read_bytes()`) nadal daje czysty, zalogowany MISS.

### Część B -- `c161d9c` nodes: backfill verbose nie może kasować obcych kluczy `system`

Grok finding. `_sync_verbose_metadata()` (`nodes.py`) budował blok `system`
od zera z pustego literału i oddawał do `save_verbose()`, która podmienia
cały blok. `add_pairing()` (`verbose_store.py:141`) zapisuje
`paired_fingerprint` / `paired_width` / `paired_height` /
`is_upscale_target` wprost w `system`; gdy PÓŹNIEJ HIT tego samego fp
trafiał na ścieżkę backfillu (sidecar obecny, ale bez `created_at` --
starszy/obcięty wpis), te cztery klucze znikały po cichu. HIT/MISS bez
zmian -- czysta utrata danych w indeksie Cache Managera.

- Fix: `system` startuje od PŁYTKIEJ KOPII istniejącego bloku
  (`dict(existing_system) if isinstance(existing_system, dict) else {}`),
  potem `.update()` nadpisuje tylko klucze, które ta funkcja posiada.
  NIE hardkodujemy czterech kluczy pairingu po nazwie -- ogólne "zachowaj
  to, czego sam nie ustawiam" jest solidniejsze.
- 3 nowe testy w `tests/test_node.py`: dokładny zgłoszony scenariusz
  (`add_pairing` -> backfill wpisu bez `created_at` -> klucze pairingu
  przeżywają); świeży MISS bez istniejącego sidecaru bez zmian; zwykły
  backfill legacy bez obcych kluczy bez zmian.

## Ustalenia istotne dla Chat

- `store._SAFETENSOR_READ_ERRORS` to teraz `(SafetensorError,
  FileNotFoundError)` -- `store.py:92`. Oba miejsca odczytu JSON łapią
  `(FileNotFoundError, ValueError)` -- `store.py:218`, `store.py:278`.
- `_sync_verbose_metadata()` seeduje `system` z płytkiej kopii
  `existing_system` -- `nodes.py:152-163`. Funkcja "posiada": prompt,
  clip_name, clip_file_size, clip_mtime_ns, cache_schema_version,
  node_variant, created_at, references, (width/height/megapixels gdy
  podane), (clip_ctime_ns gdy podane), (comfyui_version best-effort).
  Wszystko inne w `system` jest zachowywane.
- Efekt uboczny Części B: wartości width/height/megapixels/clip_ctime_ns
  z wcześniejszego sidecaru są teraz zachowywane przy backfillu, który
  ich nie dostał. Dla danego wariantu węzła są stałe (FL2VA zawsze podaje
  width/height, Ref2VA nigdy), więc to nie wprowadza niespójności.
- Skrypt probujący OSError: `scratchpad/probe_oserror.py` (poza repo,
  artefakt diagnostyczny, nie commitowany).

## Otwarte pytania

- brak

## Sugestie (nie polecenia)

- brak
