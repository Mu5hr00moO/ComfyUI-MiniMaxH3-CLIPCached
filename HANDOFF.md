# HANDOFF

## Stan na: 2026-09-02 / branch feature/benchmark-cold-encoder-eviction / commit 6c1893d

## Ostatnio zrobione

Redesign P1 dla `scripts/benchmark_conditioning.py`: encoder w ścieżkach
Native i Cached MISS jest teraz aktywnie EWAKUOWANY z page cache tuż przed
pomiarem, zamiast prewarmowany do pełnej rezydencji.

Poprzednio `_prewarm_file` robił mmap + touch wszystkich stron ~27 GB
encodera i weryfikował mincore >= 99% -- chwilę przed tym, jak ComfyUI
ładował ten sam plik do własnego procesu. Na maszynie 54 GB RAM to
żądanie podwójnej rezydencji połowy pamięci; w smoke-teście Native był
~6x wolniejszy (thrashing), a Cached MISS został ręcznie zabity w nvitop
przy eksplozji RAM.

Nowe podejście (`_evict_encoder_file`): `open(path)` ->
`os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)` na całym pliku ->
weryfikacja przez `mincore()`, że rezydencja spadła <= 1%
(`ENCODER_EVICT_MAX_RESIDENCY_FRACTION`), z retry (5 prób, 0.2 s odstęp)
gdyby kernel potrzebował chwili. Nieudana weryfikacja rzuca `RuntimeError`
tym samym wzorcem co nieudany prewarm. Eviction NIGDY nie podnosi
rezydencji ponad baseline.

VAE dalej prewarmowany do warm we wszystkich ścieżkach. Cached HIT bez
żadnej zmiany (encoder tam nigdy nie jest czytany; VAE + pliki wpisu
cache prewarmowane jak dotąd). Czas eviction poza mierzonym wall-time
dokładnie tak jak wcześniej czas prewarmu.

Setup `mincore()` wyciągnięty do wspólnego helpera
`_resident_pages_via_mincore` (prewarm i eviction weryfikują tak samo);
metadane prewarmu VAE bajt-w-bajt bez zmian.

- Commit 1 (6c1893d): `scripts/benchmark_conditioning.py`.
- Commit 2: ten plik.

### Weryfikacja (BEZ ComfyUI, BEZ serwera, BEZ żadnego case'u benchmarku)

- `python -m py_compile scripts/benchmark_conditioning.py` -- OK.
- Pełny pytest w comfyenv: **399 passed / 0 failed / 0 skipped**
  (`test_server_script_safety.py` -- ochrona reguły port-ownership
  `benchmark_conditioning.py` dalej przechodzi; string
  "refusing to adopt or stop it" nietknięty).
- Izolowany probe `POSIX_FADV_DONTNEED` + `mincore()` na PRAWDZIWYM pliku
  encodera (`qwen3vl_32b_minimax_h3_int8_convrot.safetensors`, 27 141 342 152 B,
  6 626 305 stron 4 KiB, ext4 na /dev/sdc, 54 GB RAM):
  - naturalna rezydencja przed: **0.000%** (0 stron) -- cache był zimny
  - po odczycie prefiksu 512 MiB: **2.009%** (133 120 stron, ~0.51 GiB)
  - `os.posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED)`: 0.0027 s
  - po fadvise: **0.000%** (0 stron)
- Ten sam mechanizm przez FAKTYCZNY kod modułu (`_evict_encoder_file`,
  `_measure_residency`): warm 256 MiB -> **0.989%** -> po fadvise ->
  **0.000%** (0 stron), `verified=True` na 1. próbie. `_prewarm_file` na
  małym pliku tmp: metadane VAE-path -- zestaw kluczy identyczny jak
  przed zmianą.

STOP zgodnie ze zleceniem po weryfikacji mechanizmu. Nie uruchamiano
ComfyUI ani żadnego case'u benchmarku. Decyzja o live smoke-teście
należy do następnej, osobnej wiadomości.

## Ustalenia istotne dla Chat

- Nowe stałe w `scripts/benchmark_conditioning.py:91-96`:
  `ENCODER_EVICT_MAX_RESIDENCY_FRACTION = 0.01`,
  `ENCODER_EVICT_VERIFY_ATTEMPTS = 5`,
  `ENCODER_EVICT_RETRY_DELAY_SECONDS = 0.2`.
- `_evict_encoder_file()` (`benchmark_conditioning.py:397`) -- eviction +
  weryfikacja mincore; zwraca dict z `eviction_method`,
  `filesystem_state: "forced_cold_read"`, `verify_attempts`, `verified`.
- `_prewarm_files_for_run` przemianowane na
  `_prepare_filesystem_cache_for_run` (`benchmark_conditioning.py:465`);
  zwraca płaską listę -- najpierw wpisy eviction (encoder), potem prewarm
  (VAE, pliki cache). Encoder w native/cached_miss idzie do eviction,
  VAE zawsze do prewarm, HIT bez encodera.
- Klucze w JSON raportu zmienione: `filesystem_prewarm` ->
  `filesystem_cache_preparation`, `filesystem_prewarm_seconds` ->
  `filesystem_cache_preparation_seconds`, `filesystem_prewarm_included` ->
  `filesystem_cache_preparation_included`. Te pola były tylko zapisywane,
  nigdzie nieczytane (grep potwierdzony) -- rerun/markdown/statistics ich
  nie dotykają.
- `schema_version` raportu NIE zmienione (dalej 2). Kształt kontraktu
  rerun (15 runów / 5 case'ów / statistics) bez zmian; zmienił się tylko
  audyt filesystem-cache w pojedynczym runie.
- Czas eviction jest strukturalnie poza wall-time: prewarm/eviction
  dzieje się w `_run_one` PRZED `_execute_until_fl2va_complete`, a
  `started = time.perf_counter()` (start pomiaru) jest dopiero tuż przed
  POST `/prompt`.
- W tej sesji encoder file NIE był rezydentny w page cache na starcie
  (`free -g` -> buff/cache ~0), fadvise DONTNEED sprowadza go do dokładnie
  0 stron na tym ext4/WSL2.

## Otwarte pytania

- Czy `schema_version` raportu bumpnąć do 3? Argument za: `--rerun` na
  starym raporcie v2 (prewarmed encoder) wymieszałby rekordy o różnej
  semantyce cold/warm w jednym `markdown_summary`. `_load_rerun_report`
  pilnuje tylko `schema_version == 2` i `server_environment`, nie
  metodologii. Argument przeciw: małe ryzyko w praktyce (rerun i tak
  regeneruje run od zera), a bump ma własny ripple. Zostawione do decyzji.
- Osierocony fragment README z PR #4 (`git stash@{0}` w repo tego node'a,
  "WIP: README benchmark_conditioning.py section") opisuje encoder jako
  "explicitly prewarms ... verifies them with mincore" i "not a cold-disk
  benchmark". Po tej zmianie to jest częściowo NIEAKTUALNE dla encodera
  w native/miss. Fragment świadomie nietknięty (poza zakresem P1) -- do
  poprawienia zanim trafi na master.

## Sugestie (nie polecenia)

- Przed live smoke-testem warto potwierdzić `free -g` / `nvidia-smi`
  (GPU idle, brak innego serwera na 8188), bo eviction odsłania pełny
  koszt zimnego odczytu 27 GB i pierwszy Native/MISS będzie wyraźnie
  wolniejszy niż w poprzednich (prewarmowanych) przebiegach -- to
  oczekiwane, nie regresja.
