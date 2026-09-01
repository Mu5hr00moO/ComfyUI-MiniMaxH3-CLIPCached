# HANDOFF

## Stan na: 2026-09-01 / branch master / commit cdbde7e

## Ostatnio zrobione (Codex LOW #3 + #4 -- domknięcie luki w zamykaniu serwera dev-scriptów)

Jeden commit kodu (`cdbde7e`), zakres: 4 skrypty orchestratora +
`scripts/_live_server.py` + `tests/test_live_server_stop_pid_reuse.py`.
Pełny suite: 398 passed (było 397). `py_compile` / `git diff --check` czyste.

### LOW #3 -- handler sygnału omijał eskalację/reap z `stop_live_server()`

Cztery skrypty live-server (`test_ref2video_server_e2e.py`,
`test_ref2video_server_hit.py`, `test_server_memory_trend_phase17.py`,
`test_ref2video_memory_trend.py`) instalowały handler SIGINT/SIGTERM
(`_forwarding_signal_handler` / `_sig`), który robił jeden best-effort
SIGTERM do dziecka przez `forward_termination()` i natychmiast `os._exit(1)`.
To pomijało eskalację SIGINT -> SIGTERM -> SIGKILL i `proc.wait()`-reap
dodane w `0582ef8`: dziecko ignorujące ten jeden SIGTERM (albo tylko
potrzebujące chwili) było porzucane jako sierota.

- Nowy `install_shutdown_signal_handler()` w `scripts/_live_server.py`
  kieruje oba sygnały w `OrchestratorShutdownSignal` -- podklasa
  `BaseException` (jak `KeyboardInterrupt`, więc `except Exception` w środku
  runu jej nie połknie). Handler NIE robi nic poza `raise` -- zero I/O, zero
  wywołań subprocess, zero czekania.
- Każdy `main()` łapie ten wyjątek w nowej klauzuli `except
  OrchestratorShutdownSignal` (przed istniejącym `finally:`), zapisuje
  `signum`, pozwala `finally:` wykonać tę samą `stop_live_server()` co
  normalne zamknięcie, a potem `sys.exit(128 + signum)` przed sekcją
  raportu. Jedna ścieżka zamknięcia, nie druga równoległa.
- Usunięte: `_server_proc_for_cleanup` (moduł-level uchwyt istniejący
  wyłącznie po to, żeby handler miał co złapać), `forward_termination()`
  z `_live_server.py` (jedyny użytkownik to była ścieżka bare-exit), oraz
  martwe po tej zmianie importy `os` (4 skrypty) i `signal` (tylko
  `test_ref2video_server_hit.py` -- nie ma watchdoga).

### LOW #4 -- resztkowy wyścig w `Popen.send_signal()`: udokumentowany, NIE naprawiany

Zbadane; zgadzam się z rekomendacją Codex (dokumentować, nie wdrażać pidfd).
Dopisane wprost w docstringu modułu `scripts/_live_server.py` obok
odwołania do bpo-38630/40550:

- CPython `Popen.send_signal()` po swoim `poll()`-guardzie i tak kończy
  gołym `os.kill(self.pid, sig)` (ostatni akapit komentarza w subprocess.py
  CPythona to wprost mówi). Jeśli dziecko zakończy się I OS zrecykluje jego
  dokładny PID w sub-milisekundowym oknie między guardem a `os.kill()`,
  sygnał trafi w nowego właściciela PID.
- Dlaczego to zaakceptowane, nie naprawiane: każdy `waitpid` w tych
  skryptach idzie przez metodę `Popen`, wszystkie serializowane na
  `Popen._waitpid_lock` i wszystkie re-sprawdzające `returncode` pod nim,
  więc jedyny aktor mogący w ogóle dojść do tego okna to równoległy
  `poll()`/`wait()` ścigający się z jednym `send_signal()`. Linux alokuje
  PID-y sekwencyjnie do `pid_max` (~4M), więc trafienie w ten sam numer w
  oknie mikrosekundowym nie jest realnym zdarzeniem na maszynie
  deweloperskiej. `os.pidfd_open` nie jest nawet wyeksponowane w
  interpreterze tego projektu (Python 3.14 / comfyenv) -- naprawa
  wymagałaby ominięcia `Popen.send_signal()` i własnego pidfd.
  Guideline #34: brak spekulatywnej złożoności bez potwierdzonego
  praktycznego wpływu.

## Ustalenia istotne dla Chat

- `OrchestratorShutdownSignal(BaseException)` w `scripts/_live_server.py:48`,
  `install_shutdown_signal_handler()` w `scripts/_live_server.py:65`.
  Handler to `_raise_shutdown(signum, frame): raise
  OrchestratorShutdownSignal(signum)` -- nic więcej.
- Wzorzec w każdym `main()`: `shutdown_signum = None` przed `try`,
  `except OrchestratorShutdownSignal as sig: shutdown_signum = sig.signum`
  przed `finally`, `finally` woła `stop_live_server(...)` bez zmian, po
  bloku `if shutdown_signum is not None: sys.exit(128 + shutdown_signum)`.
- `forward_termination()` NIE ISTNIEJE już w `_live_server.py`. Eksporty
  modułu: `OrchestratorShutdownSignal`, `install_shutdown_signal_handler`,
  `stop_live_server`.
- `stop_live_server()` (eskalacja + reap) bez zmian merytorycznych --
  `scripts/_live_server.py:89`.
- Watchdog (3 skrypty z nim) bez zmian: dalej `self.server_proc.send_signal(
  signal.SIGTERM)` + flaga `triggered` + `stopped_by_watchdog` ->
  `skip_sigint`. Ścieżka sygnału i ścieżka watchdoga schodzą się w tym
  samym `finally`.
- Testy: `tests/test_live_server_stop_pid_reuse.py` -- usunięte 3 testy
  `forward_termination`; dodane: handler tylko rzuca (`os._exit`
  zastubowany na fail), `OrchestratorShutdownSignal` jest BaseException-
  nie-Exception, regresja `raise_signal()`-driven teardown eskaluje przez
  zignorowany SIGTERM do SIGKILL i reapuje `FakePopen`. Statyczny guard
  orchestratora zabrania teraz `os._exit(` / `forward_termination` /
  `_server_proc_for_cleanup` i wymaga `install_shutdown_signal_handler()`
  + `except OrchestratorShutdownSignal`; guard AST trzyma sam
  `_live_server.py` wolny od `os._exit` / `os.kill`.

## Otwarte pytania

- brak

## Sugestie (nie polecenia)

- brak
