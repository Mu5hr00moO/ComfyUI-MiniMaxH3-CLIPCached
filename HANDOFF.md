# HANDOFF

## Stan na: 2026-09-01 / branch master / commit 0572f00

## Ostatnio zrobione (2 poprawki dev-tooling z audytu: PID reuse + CPU-only pytest)

Obie dotyczą wyłącznie narzędzi deweloperskich — nie kodu wykonywanego
podczas normalnego użycia węzła. Dwa commity, jeden na punkt. Pełny pytest
349 -> 360 passed (11 nowych testów regresyjnych, reszta bez zmian).

### 1. PID reuse w skryptach live-server (commit 0582ef8)

- Cztery skrypty-orkiestratory (`test_ref2video_server_e2e.py`,
  `test_ref2video_server_hit.py`, `test_server_memory_trend_phase17.py`,
  `test_ref2video_memory_trend.py`) trzymały goły int PID serwera i
  eskalowały SIGINT -> SIGTERM -> SIGKILL przez `os.kill(server_pid, ...)`
  + pętle `psutil.pid_exists(server_pid)`. Dwa problemy: (a) po wyjściu
  dziecka OS może nadać ten PID innemu procesowi -> późna eskalacja trafia
  nie w ten proces; (b) pętla `pid_exists()` NIGDY nie reapuje zombie, więc
  przy każdym czystym zamknięciu kręci się do swojego deadline'u i kończy
  raportem "exited: False". Potwierdzone empirycznie
  (`psutil.pid_exists(zombie)` == True, `os.kill(zombie, 0)` bez błędu;
  po `Popen.poll()` == reap -> `pid_exists` == False, `os.kill` ->
  ProcessLookupError).
- Nowy `scripts/_live_server.py` (współdzielony helper — świadoma decyzja,
  patrz SUGESTIE): `stop_live_server()` eskaluje przez
  `proc.send_signal()/terminate()/kill()` i czeka przez
  `proc.wait(timeout=...)`; `forward_termination()` to bezpieczny z
  poziomu signal handlera SIGTERM. CPython `Popen.send_signal()` sam
  robi `poll()` i odmawia sygnalizowania procesu, którego status wyjścia
  już zebrano (bpo-38630 / bpo-40550) — recyklowany PID nigdy nie jest
  trafiony.
- Watchdog: `__init__` bierze teraz `server_proc`, łapie
  `psutil.Process(pid)` od razu i bramkuje odczyt RSS oraz SIGTERM przez
  `proc.poll() is None and self._ps.is_running()` (is_running() = kontrola
  tożsamości po create_time — to jest to, co ZLECENIE nazwało
  "porównaniem create_time"). `send_signal` zamiast `os.kill`.
- Każdy skrypt czyści swój moduł-level uchwyt cleanup
  (`_server_proc_for_cleanup = None`) po potwierdzeniu wyjścia dziecka.
- Grace timery zachowane per skrypt (45s e2e/hit, 60s oba memory-trend).
- `tests/test_live_server_stop_pid_reuse.py`: FakePopen mimikuje guard w
  send_signal + monkeypatch `os.kill`/`psutil.pid_exists` (fail-loud).
  Asercje: brak eskalacji po czystym wyjściu na SIGINT; poprawna
  eskalacja SIGTERM/SIGKILL gdy SIGINT/SIGTERM ignorowane; `skip_sigint`
  omija SIGINT; already-exited dziecko nie dostaje żadnego sygnału;
  `forward_termination(None)` nie wywala. Plus statyczny parametryzowany
  test 4 skryptów (brak `os.kill(` / `psutil.pid_exists(` /
  `_server_pid_for_cleanup`; obecność `from _live_server import` +
  `stop_live_server(`).
- Smoke test na prawdziwych subprocessach (poza repo): graceful SIGINT
  0.01s rc=0 bez eskalacji; stubborn (SIGINT ignorowany) -> SIGTERM po
  1s rc=-15; already-dead wykryte natychmiast bez kręcenia się.

### 2. CPU-only pytest na hoście bez sterownika NVIDIA (commit 0572f00)

- Dokładny punkt inicjalizacji CUDA: `comfy/model_management.py:~363`
  `total_vram = get_total_memory(get_torch_device())` na poziomie modułu.
  Przy `cpu_state == CPUState.GPU` (domyślne) `get_torch_device()` woła
  `torch.device(torch.cuda.current_device())` -> `torch._C._cuda_init()`
  -> `RuntimeError: No CUDA GPUs are available`. To przerywa
  `import comfy.sd` / `import comfy.model_management` -> 7 modułów testowych
  nie zbiera się (test_node, test_loader, test_node_fl2va_dual,
  test_node_ref2va, test_node_ref2va_dual,
  test_ref2video_invalidation_integration, test_ref2video_slot_building).
- Jedyny hook przed linią 363: `if args.cpu: cpu_state = CPUState.CPU`
  (linia 157). `args` z `comfy.cli_args`; `comfy.options.args_parsing`
  domyślnie False -> `parse_args([])`, więc `import comfy.cli_args` z
  conftest jest bezpieczny i nie ciągnie `model_management`.
- Poprawka (najmniejsza wystarczająca, reużywa własnej ścieżki `--cpu`
  ComfyUI, guideline #31): `tests/conftest.py` po wstawieniu
  `COMFYUI_ROOT` do `sys.path` robi
  `if not torch.cuda.is_available(): comfy.cli_args.args.cpu = True`
  w `try/except Exception`. Na hoście z GPU `is_available()` == True ->
  `args` NIETKNIĘTE -> zero zmiany zachowania (zweryfikowane: `cpu_state`
  zostaje `GPU`, `args.cpu` zostaje `False`, pełny suite 360 passed
  identycznie).
- Symulacja driverless przez `CUDA_VISIBLE_DEVICES=""`: przed poprawką
  7 błędów zbierania, po poprawce 360 passed.

## Ustalenia istotne dla Chat

- `python -m py_compile` na wszystkich zmienionych plikach — OK po każdym
  commicie. `git diff --check` — czysto po obu.
- `_live_server.py` importuje się poprawnie w OBU kontekstach:
  `import _live_server` gdy skrypt uruchamiany bezpośrednio
  (`sys.path[0]` == katalog `scripts/`) oraz `import scripts._live_server`
  jako namespace package pod `python -m pytest` z korzenia repo.
  `scripts/` nie ma `__init__.py` i nie jest zbierany (pytest
  `testpaths = tests`).
- `Popen.send_signal()` w tej wersji CPython: woła `self.poll()`, potem
  `if self.returncode is not None: return`, a `os.kill` opakowany w
  `except ProcessLookupError: pass`. Bezpieczne dla bezpośredniego
  dziecka względem recyklingu PID.
- `psutil.pid_exists()` i `os.kill(pid, 0)` zwracają True/sukces dla
  zombie — dlatego stara pętla `pid_exists` nie kończyła się wcześnie.
- Zakres zmian: `scripts/_live_server.py` (nowy), 4 skrypty live-server,
  `tests/conftest.py`, `tests/test_live_server_stop_pid_reuse.py` (nowy).
  Nic poza tym.
- Commity: 0582ef8 (PID reuse), 0572f00 (CPU-only pytest).

## Otwarte pytania

- brak.

## Sugestie (nie polecenia)

- `scripts/_live_server.py` to pierwszy współdzielony moduł między
  skryptami live-server (dotąd każdy był samodzielny, z duplikowanymi
  `Watchdog` / `wait_for_server_ready` / `_read_proc_field_kb`). Wybrano
  współdzielony helper zamiast 4 równoległych kopii `stop_server`
  kierując się guideline #31 (reuse-first, "small helper" zamiast
  recreate parallel version). ZLECENIE mówiło "nic więcej" niż
  scripts/ + conftest + nowy plik testu — helper mieści się w `scripts/`,
  ale to szersza decyzja niż inline. Jeśli preferencja jest inna
  (4 lokalne kopie), łatwo cofnąć — logika jest w jednym miejscu.
- Reszta duplikacji w tych 4 skryptach (`Watchdog`,
  `wait_for_server_ready`, `_read_proc_field_kb`, handlery sygnałów)
  NIE była ruszana — poza zakresem tego zlecenia.
