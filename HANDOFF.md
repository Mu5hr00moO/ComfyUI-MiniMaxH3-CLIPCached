# HANDOFF

## Stan na: 2026-09-04 / branch feat/entry-size-in-meta-line / PR (do otwarcia)

## Ostatnio zrobione

Cache Manager pokazuje teraz rozmiar KAŻDEGO wpisu, dopisany po myślniku
do istniejącej linii meta:

    04.09.2026, 01:53 · 768×1376 (1.06 MP) - 1.0 GB

Gałąź ma dwa commity:

1. `2de30b5` — backend (`size_bytes` per wpis) + frontend (formatowanie i
   sumowanie przy dual-res) + testy.
2. ten commit — HANDOFF.md (osobno, ten sam PR).

### Backend — `size_bytes` w `scan_cache()`

Każdy wpis zwracany przez `scan_cache()` niesie `size_bytes`. Dotyczy to
wszystkich trzech klasyfikacji, także `legacy` i `inconsistent`: zepsuta
para dalej zajmuje dysk i dalej ma w UI ten sam przycisk Delete.

Kluczowe wymaganie: liczba na ekranie ma odpowiadać temu, co faktycznie
zwolni Delete. Zamiast powtarzać listę sufiksów w drugim miejscu, każdy
moduł-właściciel wystawia teraz listę ścieżek, którą konsumuje jego własny
krok kasujący:

- `store.conditioning_paths()` — `<fp>.json`, `<fp>.safetensors` (w
  kolejności kasowania), używane przez `delete_conditioning()`;
- `verbose_store.verbose_paths()` — `<fp>.verbose.json`, używane przez
  `delete_verbose()`;
- `thumbnails.thumbnail_paths()` — `thumbnails/<fp>_*.jpg` (glob, więc
  tylko pliki istniejące), używane przez `delete_thumbnails()`.

`scanner.entry_file_paths()` skleja te trzy. Zachowanie samego kasowania
NIE zmienione — każda funkcja po prostu iteruje teraz po swojej liście.
Brakujący plik (wpis legacy nie ma sidecara, większość wpisów nie ma
miniaturek) nie dokłada bajtów i nie jest błędem; ta sama tolerancja
obejmuje plik znikający w trakcie skanu.

### Jedno przejście vs dwa (punkt z zlecenia)

`_entry_size_bytes()` robi WŁASNE `stat()` po `entry_file_paths()`, a nie
wycinek z przejścia `_dir_size_bytes()`. Świadoma decyzja: żeby wyciągnąć
rozmiary wpisów z tamtego przejścia, `scanner.py` musiałby sam odtworzyć
regułę „który plik należy do którego wpisu” (w tym wzorzec nazw
miniaturek) — czyli dokładnie to drugie zapisanie konwencji nazw, którego
`entry_file_paths()` ma unikać. Koszt tej decyzji to garść dodatkowych
`stat()` na katalogu, po którym i tak chodzimy. Uzasadnienie jest w
docstringu `_entry_size_bytes()`.

`total_size_bytes` bez zmian — dalej liczy CAŁY katalog, więc suma
`size_bytes` po wpisach nie równa się totalowi (sieroty i pliki obce nie
należą do żadnego wpisu). Sprawdzone na realnym `cache/`: 29 wpisów, suma
per-wpis 963 082 337 B vs katalog 963 084 230 B (różnica 1 893 B to pliki
spoza wpisów).

### Frontend — `web/main.js`

- `formatEntryMetaLine(system, sizeBytes = 0)` — rozmiar jest ARGUMENTEM,
  funkcja tylko formatuje i nigdy nie sięga po parowanie. Używa
  istniejącego `formatBytes()` (ten sam helper co nagłówek „Cache: N
  entries / size”), więc format jest spójny z resztą panelu. Brak
  rozmiaru albo 0 → linia zostaje dokładnie jak dziś.
- `entryOwnSizeBytes(entry)` — własne bajty wpisu; wszystko, co nie jest
  dodatnią skończoną liczbą (odpowiedź `/check` sprzed tej zmiany, 0,
  string), daje 0.
- `entryDisplaySizeBytes(entry, pairing)` — SUMA obu stron tylko przy
  `pairing.status === "valid"`; każdy inny status (`none` / `orphaned` /
  `inconsistent-pair` / `role-unknown`) → własny rozmiar.
- Trzy miejsca wywołania: wiersz listy i panel szczegółów dostają sumę
  przez `entryDisplaySizeBytes()`; pasek partnera przy dual-res dostaje
  `entryOwnSizeBytes(partner)` — suma pary na obu liniach naraz wyglądała
  na zdublowaną.
- `populateDetail()` liczy teraz `resolvePairing()` RAZ i podaje wynik i
  do linii meta, i do listy fingerprintów (wcześniej liczyło dwa razy).

## Weryfikacja

- Pełny pytest: **446 passed**, 0 skipped (przed zmianą 438 — doszło 8
  nowych testów w `tests/test_scanner.py`).
- `python -m py_compile` na wszystkich zmienionych plikach `.py`: czysto.
- `node --check` na kopii `.mjs` z `web/main.js`: czysto.
- `git diff --check`: czysto.
- `scan_cache()` uruchomione na REALNYM `cache/` (29 wpisów) — liczby
  wyżej.
- Scratchpadowy harness ESM (loader podstawia `/scripts/app.js` i
  `/scripts/api.js`, minimalne `document`; harness i loader NIE
  commitowane; testowany jest REALNY `web/main.js`, nie kopia) — **16
  asercji PASS**, w tym: progi formatowania (`700 B`, `1.5 MB`, `1.0 GB`),
  każdy wariant „nie ma czego pokazać” (brak pola, 0, ujemne, `NaN`,
  string), rozmiar doklejony też gdy wpis nie ma rozdzielczości, oraz
  suma-vs-własny dla wszystkich statusów parowania — z przejściem
  end-to-end przez prawdziwe `resolvePairing()` (para bazowa 3 MB +
  upscale 1 MB → wiersz `- 4.0 MB`, pasek partnera `- 1.0 MB`).

## NIE zweryfikowane (do sprawdzenia przez Kamila w żywym ComfyUI)

- Realny render linii meta z rozmiarem w wierszu listy, w panelu
  szczegółów i w rozwiniętym pasku partnera.
- Czy dopisany rozmiar nie rozpycha wiersza listy w poziomie przy długich
  nazwach/promptach.
- Czy przy realnej parze dual-res wiersz bazowy pokazuje sumę, a pasek
  partnera własny rozmiar partnera.
- Brak błędów w konsoli przeglądarki.

## Ustalenia istotne dla Chat

- `scan_cache()` zwraca teraz `size_bytes` per wpis obok istniejącego
  `total_size_bytes` — `minimaxh3_clipcache/scanner.py:173-206`.
- Zestaw plików wpisu ma jedno źródło prawdy per artefakt:
  `store.conditioning_paths()` (`store.py:182`),
  `verbose_store.verbose_paths()` (`verbose_store.py:233`),
  `thumbnails.thumbnail_paths()` (`thumbnails.py:99`), sklejone w
  `scanner.entry_file_paths()` (`scanner.py:81`). Kasowanie i liczenie
  rozmiaru czytają tę samą listę.
- Zbiór trzech artefaktów wpisu jest wymieniony w DWÓCH miejscach:
  `routes._delete_entry_files()` (`routes.py:103`) i
  `scanner.entry_file_paths()`. Nie dało się tego zejść do jednego bez
  przebudowy `delete` (zlecenie tego zabraniało), więc oba miejsca mają
  komentarz wskazujący na drugie. Czwarty artefakt trzeba by dodać w obu.
- `formatEntryMetaLine()` jest teraz eksportowana i przyjmuje rozmiar jako
  drugi argument — `web/main.js:158`. Granica jest celowa: rozmiar wylicza
  wywołujący, funkcja tylko formatuje.
- Sumowanie przy dual-res działa wyłącznie dla `resolvePairing()` o
  statusie `"valid"` (czyli tylko dla pary faktycznie złożonej w jeden
  wiersz).

## Otwarte pytania

- brak

## Sugestie (nie polecenia)

- `docs/CACHE_MANAGER.md` wylicza, co pokazuje wpis („creation date,
  generation resolution”) — ta lista jest teraz niepełna. Świadomie
  nietknięte: ZAKRES zlecenia obejmował kod i testy, a ostatnie PR-y
  (np. #16) też nie ruszały tego pliku. Warto dopisać przy najbliższej
  okazji, razem z notką o sumowaniu przy dual-res.
- `entryMetaTooltip()` tłumaczy dziś rozjazd „data vs rozdzielczość”, ale
  nic nie mówi o rozmiarze. Przy złożonej parze dual-res wiersz pokazuje
  sumę obu stron, co bez podpowiedzi może zaskoczyć — jednozdaniowe
  rozszerzenie tooltipa byłoby tanie.
- `CHANGELOG.md` ma sekcję `[Unreleased]` bez pozycji `Added` — ta zmiana
  jest widoczna dla użytkownika i pasowałaby tam przy następnym wydaniu.
