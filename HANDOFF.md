# HANDOFF

## Stan na: 2026-09-03 / branch chore/registry-metadata / commit bb72f14

## Ostatnio zrobione

Metadane publikacyjne dla ComfyUI Registry (krok 3 z planu przed tagiem
v1.0.0). Gałąź `chore/registry-metadata` odcięta od `origin/master`
(e3191fb, po mergu PR #8). Trzy nowe pliki, każdy osobny commit; PR na
`master` niezmergowany.

- Commit 1 (4af23af): `pyproject.toml` w korzeniu repo.
  * `[project].name = "minimaxh3-clipcached"` (id node'a w Registry,
    nieodwracalne po publikacji), `version = "1.0.0"`,
    `license = { file = "LICENSE" }` (LICENSE to MIT),
    `dependencies = ["safetensors"]` (jedyna realna zależność spoza
    ComfyUI; torch/aiohttp/PIL przychodzą z ComfyUI - nie dopisane),
    `classifiers = ["Operating System :: OS Independent"]`.
  * `requires-python = ">=3.9"` - patrz "Ustalenia" niżej.
  * `[project.urls].Repository` = URL repo na GitHubie.
  * `[tool.comfy]`: `PublisherId = "mu5hr00moo"` (małe litery, dokładnie
    jak po @ na profilu publishera), `DisplayName = "MiniMax H3 CLIP-Cached"`,
    `requires-comfyui = ">=0.30.0"` (natywne MiniMax H3 od ComfyUI 0.30.0;
    ta sama wartość co sekcja Requirements w README).
- Commit 2 (6ba3a2b): `.comfyignore` w korzeniu repo. Składnia .gitignore
  (wymóg dokumentacji publishing). Wyklucza z paczki: `tests/`, `.github/`,
  `cache/`, `benchmark_results/`, `CLAUDE.md`, `HANDOFF.md`, `TODO.md`.
  Zostają: `docs/`, `scripts/`, `web/`, `minimaxh3_clipcache/`, `nodes.py`,
  `__init__.py`, `README*.md`, `README_*.png`, `LICENSE`, `SECURITY.md`,
  `CHANGELOG.md`.
- Commit 3 (bb72f14): `.github/workflows/publish.yml`. Oficjalny wzór z
  docs.comfy.org/registry/publishing (uses `Comfy-Org/publish-node-action@main`,
  `personal_access_token: ${{ secrets.REGISTRY_ACCESS_TOKEN }}`,
  `actions/checkout@v7`, trigger: push do gałęzi domyślnej zmieniający
  `pyproject.toml` + `workflow_dispatch`). Wzór z dokumentacji ma zaszyte
  `branches: - main` - zmienione na `master`, inaczej akcja nigdy by się
  nie odpaliła. Istniejące `.github/workflows/tests.yml` nietknięte.
- Commit HANDOFF: ten plik (osobno, w tym samym PR - nie push na master).

### Weryfikacja (BEZ ComfyUI, BEZ serwera, BEZ GPU)

- `python -c "import tomllib; tomllib.load(open('pyproject.toml','rb'))"` -
  parsuje się; asercje na name / version / license / dependencies /
  PublisherId / DisplayName / requires-comfyui / Repository - wszystkie
  przechodzą, PublisherId to małe `mu5hr00moo`.
- `publish.yml` parsuje się przez `yaml.safe_load` w comfyenv; trigger
  `push.branches == ["master"]`, brak `"main"` w pliku; `uses` i token
  zgodne z wzorcem. `tests.yml` bez zmian (`git status`).
- Pełny pytest w comfyenv: **399 passed / 0 failed / 0 skipped**
  (nowe pliki to TOML/YAML/ignore - nic nie ruszają, potwierdzone).
- Symulacja listy plików w publikowanej paczce (git ls-files minus wzorce
  `.comfyignore`): 51 plików w paczce, 37 wykluczonych. Sprawdzone, czy
  `.comfyignore` nie wycina czegoś linkowanego z README.md / `docs/*.md` -
  patrz "Otwarte pytania" (jeden martwy link znaleziony).

## Ustalenia istotne dla Chat

- `requires-python = ">=3.9"`. Podstawa: kod subskryptuje generyki
  wbudowane (`list[...]`, `dict[...]`) w pozycjach ewaluowanych w
  runtime, BEZ `from __future__ import annotations` (nigdzie w repo go
  nie ma):
  * `minimaxh3_clipcache/store.py:372` - adnotacja zwrotu funkcji
    `def gc_orphaned_cache_files(cache_dir: Path) -> list[str]:`
    (adnotacje funkcji są ewaluowane przy definicji).
  * `minimaxh3_clipcache/locking.py:24` - adnotacja zmiennej na poziomie
    modułu `_fingerprint_locks: dict[str, threading.Lock] = {}`
    (adnotacje na poziomie modułu SĄ ewaluowane w runtime).
  Oba rzucają `TypeError` na Pythonie 3.8 (PEP 585 dopiero od 3.9).
  Skan AST całego `nodes.py` + `minimaxh3_clipcache/*.py`: brak `match`,
  brak unii `X | Y` w pozycjach ewaluowanych, brak PEP 695, brak metod
  stdlib z 3.9+ poza samymi generykami - nic nie wymusza >=3.10.
- Dla informacji: sam ComfyUI v0.34.2 ma `requires-python = ">=3.10"`
  (`~/ComfyUI/pyproject.toml:6`), więc w praktyce node i tak działa na
  3.10+. Wartość `>=3.9` odzwierciedla NASZĄ składnię, zgodnie ze
  zleceniem ("najniższą wersję faktycznie wynikającą ze składni").
- Nazwa pola to `requires-comfyui` (z myślnikiem), należy do `[tool.comfy]`
  - potwierdzone w verbatim przykładzie na
    docs.comfy.org/registry/specifications (pierwsze streszczenie WebFetch
    błędnie sugerowało `requires-comfy` - to była halucynacja modelu
    streszczającego, sprostowana drugim, dosłownym pobraniem).
- `.comfyignore` używa składni `.gitignore` i działa warstwowo na nią
  (pliki nieśledzone przez git są już wykluczone). `cache/` i
  `benchmark_results/` są w `.gitignore` jako nieśledzone - w
  `.comfyignore` dodane redundantnie, świadomie (defense-in-depth).
- Pliki w paczce, których zlecenie nie wymieniło wprost, a które tam
  trafią: `.gitignore` (drobny), `pytest.ini` (`testpaths = tests` -
  po wykluczeniu `tests/` to martwa konfiguracja, ale nieszkodliwa).
  Świadomie NIE dodane do `.comfyignore` - zlecenie dało zamkniętą listę.
- Sekret `REGISTRY_ACCESS_TOKEN` Kamil dodaje ręcznie w ustawieniach repo;
  nic związanego z kluczem nie ma w commitach.

## Otwarte pytania

- `docs/TESTING_AND_LIMITATIONS.md:352` linkuje
  `[Bug report template](../.github/ISSUE_TEMPLATE/bug_report.yml)`.
  `.comfyignore` wyklucza całe `.github/`, więc w opublikowanej paczce
  ten względny link jest martwy. Do decyzji: (a) zawęzić wykluczenie do
  `.github/workflows/` (zostawić `ISSUE_TEMPLATE/` w paczce), czy
  (b) zmienić link w docs na bezwzględny URL do GitHuba. Zmiana docs jest
  poza zakresem tego zlecenia - zostawione jako decyzja dla Chat/Kamila.
  To jedyny martwy link znaleziony w symulacji paczki.

## Sugestie (nie polecenia)

- Rekomendacja do otwartego pytania: wariant (b) - poprawić link w
  `docs/TESTING_AND_LIMITATIONS.md` na pełny URL GitHuba. Wykluczenie
  całego `.github/` z paczki jest czystsze niż selektywne wpuszczanie
  jednego podkatalogu, a template zgłoszeń błędu i tak jest użyteczny
  tylko z poziomu GitHuba (nie z lokalnej kopii node'a).
- Po mergu tego PR-a workflow `publish.yml` odpali się automatycznie
  (PR dodaje `pyproject.toml`, co pasuje do filtra `paths`). Upewnić się,
  że sekret `REGISTRY_ACCESS_TOKEN` jest w repo PRZED mergem, inaczej
  pierwszy przebieg padnie na braku tokenu.
- `gh` w tej sesji: PR otwarty / do otwarcia ręcznie - patrz raport.
