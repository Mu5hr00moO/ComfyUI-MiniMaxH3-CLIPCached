# HANDOFF

## Stan na: 2026-09-02 / branch feature/benchmark-schema-version-3 / commit 0829299

## Ostatnio zrobione

Domknięte pierwsze otwarte pytanie z audytu P1: `schema_version` raportu
`scripts/benchmark_conditioning.py` bumpnięty **2 -> 3**.

Powód: zmiana "force-cold eviction" (PR #5, zmergowany jako 84b32e4)
zostawiła `schema_version` na 2, mimo że Native/Cached MISS mierzą teraz
zimny odczyt encodera zamiast prewarmowanego. `--rerun` na starym
raporcie v2 rozszerzałby go w miejscu, mieszając w jednym
`markdown_summary` czasy prewarmowane z wymuszonymi na zimno.

- `_initial_report` zapisuje `"schema_version": 3` (z komentarzem DLACZEGO,
  `benchmark_conditioning.py:1673-1676`).
- Guard `_load_rerun_report` wymaga `== 3` (`:1903`); komunikat błędu przy
  niezgodności jawnie tłumaczy, że to zmiana METODOLOGII (encoder cold
  zamiast prewarmu), nie sam numer wersji, i że stary raport trzeba
  wygenerować od zera, nie da się go rozszerzyć.
- `schema_version` w tym pliku występuje tylko w tych 2 miejscach + string
  błędu; docstring / help `--rerun` go nie wymieniają. `cache_schema_version`
  w testach/README to inny byt (fingerprint węzła) -- nietknięty.

PR #5 (force-cold) był OPEN na starcie tego zlecenia; odczekany do mergu
(polling `origin/master` na obecność `_evict_encoder_file`), dopiero potem
gałąź `feature/benchmark-schema-version-3` odcięta od świeżego
`origin/master` (84b32e4).

- Commit 1 (0829299): `scripts/benchmark_conditioning.py`.
- Commit 2: ten plik.

### Weryfikacja (BEZ ComfyUI, BEZ serwera, BEZ GPU)

- `python -m py_compile scripts/benchmark_conditioning.py` -- OK.
- Pełny pytest w comfyenv: **399 passed / 0 failed / 0 skipped**.
- Runtime check komunikatu guardu: fake raport `{"schema_version": 2}`
  podany do `_load_rerun_report` -> `RuntimeError` z pełnym tekstem
  ("methodology change, not just a version number ... run a fresh full
  benchmark instead").

## Ustalenia istotne dla Chat

- `schema_version` raportu benchmarku = **3** (`benchmark_conditioning.py:1676`).
  Guard `--rerun`: `_load_rerun_report` odrzuca wszystko != 3
  (`benchmark_conditioning.py:1903`).
- Reszta kontraktu rerun (15 runów / 5 case'ów / kształt `statistics` /
  `status in ("complete", "rerun_failed")`) bez zmian.
- Stan P1 (z PR #5, teraz na master): encoder w Native/Cached MISS
  wymuszany na zimno przez `os.posix_fadvise(fd,0,0,POSIX_FADV_DONTNEED)`
  + weryfikacja `mincore()` <= 1% (`ENCODER_EVICT_MAX_RESIDENCY_FRACTION`),
  retry 5x/0.2s. VAE dalej prewarmowany do warm we wszystkich ścieżkach.
  Cached HIT bez zmian. Czas eviction poza mierzonym wall-time.
- `_evict_encoder_file()` `benchmark_conditioning.py:397`;
  `_prepare_filesystem_cache_for_run()` (dawne `_prewarm_files_for_run`)
  `:465`; wspólny helper `_resident_pages_via_mincore` `:302`.
- Klucze JSON raportu z P1: `filesystem_prewarm*` ->
  `filesystem_cache_preparation*` (były tylko zapisywane, nigdzie
  nieczytane).

## Otwarte pytania

- Osierocony fragment README z PR #4 (`git stash@{0}` w repo tego node'a,
  "WIP: README benchmark_conditioning.py section") opisuje encoder jako
  "explicitly prewarms ... not a cold-disk benchmark" -- po zmianie
  force-cold częściowo NIEAKTUALNE dla encodera w Native/MISS. Świadomie
  odłożone do zamknięcia całego wątku benchmarku (README = ostatnia faza),
  NIE ruszane w tym zleceniu.

## Sugestie (nie polecenia)

- Przy pierwszym live smoke-teście schematu 3: istniejący
  `benchmark_results/conditioning_benchmark.json` (jeśli jest, schema 2)
  nie będzie już akceptowany przez `--rerun` -- to oczekiwane, trzeba
  pełny przebieg od zera.
