# HANDOFF

## Stan na: 2026-09-03 / branch chore/repo-cleanup-pre-v1 / commit 7663cec

## Ostatnio zrobione

Porządek w repo przed tagiem v1.0.0. Gałąź `chore/repo-cleanup-pre-v1`
odcięta od `origin/master` (12f9b7e, po mergu PR #7 z katalogiem `docs/`).

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
- Commit 3: ten plik.

### Weryfikacja (BEZ ComfyUI, BEZ serwera, BEZ GPU)

- `git grep` na nazwie usuniętego dokumentu planistycznego -- brak wyników.
- `git grep` na absolutnej ścieżce domowej tej maszyny -- brak wyników.
- `python -m py_compile` na 5 zmienionych skryptach + `thumbnails.py` -- OK.
- `node --check` na kopii `web/main.js` (jako `.mjs`) -- składnia OK.
- Pełny pytest w comfyenv: **399 passed / 0 failed / 0 skipped**
  (w tym `tests/test_server_script_safety.py` i
  `tests/test_live_server_stop_pid_reuse.py`, które pilnują skryptów
  serwerowych).
- Runtime check rozwiązywania `COMFYUI_ROOT` we wszystkich 5 skryptach:
  fallback z układu katalogów wskazuje właściwą lokalną instalację,
  a env override `COMFYUI_ROOT` jest respektowany.

## Ustalenia istotne dla Chat

- Przedimplementacyjny dokument planistyczny Cache Managera już nie
  istnieje w repo. Aktualny opis Cache Managera: `docs/CACHE_MANAGER.md`.
  Reszta dokumentacji w `docs/` (NODE_GUIDE, PERFORMANCE,
  TECHNICAL_DETAILS, TESTING_AND_LIMITATIONS).
- Wzorzec rozwiązywania korzenia ComfyUI (jedno źródło, powielane):
  `COMFYUI_ROOT = os.environ.get("COMFYUI_ROOT", <fallback>)`, gdzie
  fallback to cztery katalogi w górę od pliku
  (`<ComfyUI>/custom_nodes/<repo>/scripts/<plik>`). Użyte w
  `tests/conftest.py:39`, `tests/test_clip_name_node.py:25`,
  `tests/test_node_fl2va_dual.py:33`, `tests/test_node_ref2va_dual.py:32`
  oraz teraz w `scripts/test_proxy_gate.py`,
  `scripts/test_ref2video_memory_trend.py`,
  `scripts/test_ref2video_server_e2e.py`,
  `scripts/test_ref2video_server_hit.py`,
  `scripts/test_server_memory_trend_phase17.py`.
- `scripts/benchmark_conditioning.py` już wcześniej był przenośny
  (`DEFAULT_COMFYUI_ROOT = REPO_ROOT.parent.parent`, `:82`) -- nietknięty.

## Otwarte pytania

- brak (w zakresie tego zlecenia).

## Sugestie (nie polecenia)

- `pyproject.toml` i rozdzielenie `test_proxy_gate.py` na dwie role to
  osobne pozycje (odpowiednio: osobne zlecenie i pozycja w `TODO.md`),
  celowo nietknięte tutaj.
- Skrypty serwerowe (`test_ref2video_server_*`, `test_server_memory_trend_phase17`,
  `test_ref2video_memory_trend`) nie były uruchamiane end-to-end w tym
  zleceniu (wymagają GPU + ~27 GB encodera); zweryfikowano tylko, że
  poprawnie wyliczają `COMFYUI_ROOT` i importują się bez błędu.
