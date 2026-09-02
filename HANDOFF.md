# HANDOFF

## Stan na: 2026-09-03 / branch chore/repo-cleanup-pre-v1 / commit fab353a

## Ostatnio zrobione

Porządek w repo przed tagiem v1.0.0. Gałąź `chore/repo-cleanup-pre-v1`
odcięta od `origin/master` (12f9b7e, po mergu PR #7 z katalogiem `docs/`).
Otwarta jako PR #8 na `master` (niezmergowana).

- Commit 1 (9c132f6): usunięty przedimplementacyjny dokument planistyczny
  Cache Managera z korzenia repo (zastąpiony przez `docs/`; kopia zapasowa
  poza repo). Cztery odwołania do niego poprawione tak, by nie zostało
  martwe wskazanie:
  * `web/main.js` (dwa komentarze przy "Copy prompt") -- usunięte zdania
    odsyłające do sekcji planu; merytoryczne uzasadnienie (findNodesByType
    nie schodzi w subgraphy, prompt-jako-input nie ma widgetu) zostaje
    inline bez zmian.
  * `minimaxh3_clipcache/thumbnails.py` (docstring) -- usunięte zdanie
    odsyłające do sekcji planu; reszta uzasadnienia bez zmian.
  * `CLAUDE.md` (sekcja "Cache Manager") -- odwołanie przekierowane na
    `docs/CACHE_MANAGER.md`.
- Commit 2 (7663cec): przenośność skryptów diagnostycznych. Pięć plików
  w `scripts/` miało zaszytą na sztywno absolutną ścieżkę tej maszyny w
  `COMFYUI_ROOT` - teraz env override + fallback wyliczany z układu
  instalacji, ten sam wzorzec co `tests/conftest.py`. Dodatkowo z
  `CLAUDE.md` usunięte 5 wystąpień absolutnej ścieżki domowej tej maszyny,
  zastąpionych generycznym odwołaniem do katalogu instalacji ComfyUI.
- Commit 3 (fab353a): wyliczanie `COMFYUI_ROOT` odporne na symlinki.
  `Path(__file__).resolve()` rozwija symlink PRZED wejściem w górę po
  katalogach, więc przy popularnej instalacji "repo poza custom_nodes/,
  podlinkowane do środka" wyliczony korzeń wskazywał o katalog za wysoko.
  Cztery orkiestratory `_live_server` (`test_ref2video_memory_trend.py`,
  `test_ref2video_server_e2e.py`, `test_ref2video_server_hit.py`,
  `test_server_memory_trend_phase17.py`) przełączone na
  `os.path.abspath(__file__)` + leksykalne wejście w górę, 1:1 jak
  `tests/conftest.py`; komentarz WHY w każdym wprost mówi, że `abspath`
  (nie `resolve`) jest celowe. `scripts/benchmark_conditioning.py:81-82`
  miał ten sam wzorzec (`REPO_ROOT` przez `.resolve()`,
  `DEFAULT_COMFYUI_ROOT` dwa poziomy wyżej) - poprawiony tak samo. Env
  override `COMFYUI_ROOT` NIE został tam dodany, bo skrypt bierze katalog
  ComfyUI przez argument `--comfyui-root` i druga ścieżka nadpisania
  tylko zaciemniłaby kolejność pierwszeństwa.
- Commit HANDOFF: ten plik (osobno).

### Weryfikacja (BEZ ComfyUI, BEZ serwera, BEZ GPU)

- `git grep` na nazwie usuniętego dokumentu planistycznego -- brak wyników.
- `git grep` na absolutnej ścieżce domowej tej maszyny -- brak wyników.
- `git grep -n "resolve().parents\[3\]"` w `scripts/` -- brak wyników.
- `python -m py_compile` na wszystkich zmienionych skryptach + `thumbnails.py` -- OK.
- `node --check` na kopii `web/main.js` (jako `.mjs`) -- składnia OK.
- Odtworzony symlinkowany install (repo poza `custom_nodes/`, symlink do
  środka, `main.py` jako marker): wszystkie 4 skrypty serwerowe oraz
  domyślny `--comfyui-root` benchmarku wyliczają REALNY korzeń ComfyUI
  (`<tmp>/ComfyUI`), podczas gdy stary `Path(__file__).resolve().parents[3]`
  dawał katalog wyżej (`<tmp>`). Env override `COMFYUI_ROOT` nadal
  respektowany. Lokalny install na tej maszynie (zwykły katalog, nie
  symlink) bez zmian - fallback dalej wskazuje tę samą, właściwą
  instalację ComfyUI co wcześniej.
- Pełny pytest w comfyenv: **399 passed / 0 failed / 0 skipped**
  (w tym `tests/test_server_script_safety.py` i
  `tests/test_live_server_stop_pid_reuse.py`, które pilnują skryptów
  serwerowych).

## Ustalenia istotne dla Chat

- Przedimplementacyjny dokument planistyczny Cache Managera już nie
  istnieje w repo. Aktualny opis Cache Managera: `docs/CACHE_MANAGER.md`.
  Reszta dokumentacji w `docs/` (NODE_GUIDE, PERFORMANCE,
  TECHNICAL_DETAILS, TESTING_AND_LIMITATIONS).
- Wzorzec rozwiązywania korzenia ComfyUI (jedno źródło, powielane):
  `COMFYUI_ROOT = os.environ.get("COMFYUI_ROOT", <fallback>)`, gdzie
  fallback to cztery katalogi w górę od pliku
  (`<ComfyUI>/custom_nodes/<repo>/scripts/<plik>`), liczone przez
  `os.path.abspath(__file__)` + `os.path.dirname` (NIE `Path.resolve()`
  ani `os.path.realpath` - te rozwijają symlink instalacji custom node'a
  i wychodzą o katalog za wysoko). Użyte w `tests/conftest.py:38`,
  `tests/test_clip_name_node.py`, `tests/test_node_fl2va_dual.py`,
  `tests/test_node_ref2va_dual.py` oraz w `scripts/test_proxy_gate.py`,
  `scripts/test_ref2video_memory_trend.py`,
  `scripts/test_ref2video_server_e2e.py`,
  `scripts/test_ref2video_server_hit.py`,
  `scripts/test_server_memory_trend_phase17.py`.
- `scripts/benchmark_conditioning.py:81-90`: `REPO_ROOT` /
  `DEFAULT_COMFYUI_ROOT` też liczone bez `.resolve()` (odporne na
  symlink); nadpisanie idzie WYŁĄCZNIE przez argument CLI
  `--comfyui-root`, nie przez zmienną środowiskową.

## Otwarte pytania

- brak (w zakresie tego zlecenia).

## Sugestie (nie polecenia)

- `scripts/test_proxy_gate.py` już liczy `COMFYUI_ROOT` poprawnie
  (`abspath` + 4x `dirname`), ale jego komentarz WHY nie wspomina wprost
  o powodzie symlinkowym - warto dopisać to samo zdanie dla spójności
  (celowo nietknięte w tym zleceniu: "test_proxy_gate.py jest już
  poprawny - nie ruszać").
- `scripts/test_ref2video_server_hit.py:148` liczy `cache_dir` przez
  `Path(__file__).resolve().parent.parent / "cache"` - to inna ścieżka
  (katalog cache repo, nie korzeń ComfyUI) i przy symlinku wskazuje ten
  sam katalog cache przez realną lokalizację; I/O jest transparentne, więc
  nie jest to błąd, ale gdyby ktoś chciał pełnej spójności - też do
  ujednolicenia. Poza zakresem tego zlecenia.
- `pyproject.toml` i rozdzielenie `test_proxy_gate.py` na dwie role to
  osobne pozycje (odpowiednio: osobne zlecenie i pozycja w `TODO.md`),
  celowo nietknięte tutaj.
- Skrypty serwerowe (`test_ref2video_server_*`, `test_server_memory_trend_phase17`,
  `test_ref2video_memory_trend`) nie były uruchamiane end-to-end w tym
  zleceniu (wymagają GPU + ~27 GB encodera); zweryfikowano tylko, że
  poprawnie wyliczają `COMFYUI_ROOT` i importują się bez błędu.
