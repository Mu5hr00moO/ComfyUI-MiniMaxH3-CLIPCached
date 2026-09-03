# HANDOFF

## Stan na: 2026-09-03 / branch chore/registry-metadata / PR #9 (niezmergowany)

## Ostatnio zrobione

Metadane publikacyjne dla ComfyUI Registry (krok 3 planu przed wydaniem
v1.0.0) plus poprawki po recenzji Chat i bota Greptile. Gałąź
`chore/registry-metadata` odcięta od `origin/master` (e3191fb). PR #9 na
`master`, otwarty, MERGEABLE.

### Pierwsza tura (commity 4af23af..36f52ff)

- `pyproject.toml` (4af23af) — manifest Registry: `name =
  "minimaxh3-clipcached"` (id node'a, nieodwracalne po publikacji),
  `version = "1.0.0"`, `license = { file = "LICENSE" }` (MIT),
  `dependencies = ["safetensors"]`, `[project.urls].Repository`,
  `[tool.comfy]` z `PublisherId = "mu5hr00moo"` (małe litery),
  `DisplayName = "MiniMax H3 CLIP-Cached"`, `requires-comfyui = ">=0.30.0"`.
- `.comfyignore` (6ba3a2b) — składnia .gitignore, warstwa na .gitignore.
- `.github/workflows/publish.yml` (bb72f14) — oficjalny wzór
  `Comfy-Org/publish-node-action@main`; wzór z docs ma zaszyte
  `branches: - main`, zmienione na `master`. `tests.yml` nietknięte.

### Druga tura — poprawki po recenzji (commity 5b99566..0099dd1)

- `requires-python` (5b99566): `">=3.9"` → `">=3.10"`. Nad polem komentarz
  WHY: floor pochodzi z ComfyUI (aktualne ComfyUI, v0.34.2, ma
  `requires-python = ">=3.10"`), nie z samej składni tego repo. Kod repo
  parsowałby się na 3.9 (jedyny konstrukt 3.9+ to subskrypcja generyków
  PEP 585), ale bez ComfyUI node i tak nie działa. Komentarz ma zapobiec
  cofnięciu wartości na 3.9 na podstawie analizy składni.
- Martwy link + pytest.ini (9bfdd3e):
  * `docs/TESTING_AND_LIMITATIONS.md:352` — względny link
    `../.github/ISSUE_TEMPLATE/bug_report.yml` (martwy w publikowanej
    paczce, bo `.comfyignore` wycina `.github/`) zamieniony na pełny URL
    `https://github.com/Mu5hr00moO/ComfyUI-MiniMaxH3-CLIPCached/issues/new?template=bug_report.yml`.
  * `.comfyignore` — dopisany `pytest.ini` (`testpaths = tests`, a
    `tests/` jest wykluczone → martwa konfiguracja w kopii u użytkownika).
    `.gitignore` zostawiony w paczce (drobny, nieszkodliwy).
- `CHANGELOG.md` (0099dd1) — format Keep a Changelog, nagłówek z linkami
  do keepachangelog.com i semver.org. `## [Unreleased]` → `### Planned`
  z dwoma odłożonymi pozycjami z TODO.md (`cache_mode="cache_only"`,
  dynamiczne sloty referencji Ref2VA). `## [1.0.0] - 2026-09-03` opisuje
  CO repo zawiera (pięć węzłów, cache conditioning z uwolnieniem VRAM po
  enkodowaniu, Cache Manager, docs/), nie historię commitów. Na końcu
  sekcji 1.0.0 dwie informacje przedinstalacyjne: ComfyUI >= 0.30.0,
  on-disk cache schema v2.
- Commit HANDOFF: ten plik (osobno, w tym samym PR).

### Trzecia tura — recenzja Greptile na PR #9 (commit b827cce)

- `.github/workflows/publish.yml` (b827cce) — odpowiedź na uwagę P1
  (security) bota Greptile: krok Publish przekazuje `REGISTRY_ACCESS_TOKEN`
  do zewnętrznej akcji, a mutowalny ref (`@main` / `@v7`) pozwoliłby
  wykonać nieprzejrzany kod z dostępem do sekretu. Oba `uses:` przypięte
  do pełnych 40-znakowych SHA z komentarzem wersji obok:
  * `actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1 # v7.0.1`
    (ten SHA to jednocześnie `refs/tags/v7` i `refs/tags/v7.0.1`)
  * `Comfy-Org/publish-node-action@d2366e7abb6ab16f3bb03e3520ae25c8cf749bc9 # main @ 2026-09-03`
    (HEAD gałęzi `main`; tagi `1.0.0`/`1.0.1` wskazują starszy `7578cdb`,
    a tagi i tak są mutowalne — dlatego pinujemy SHA gałęzi `main`)
  Nad krokiem Publish komentarz WHY: dlaczego SHA zamiast `@main`, i że to
  świadome odejście od wzoru z oficjalnej dokumentacji Comfy. `tests.yml`
  poza zakresem tego PR-a, nietknięte. SHA potwierdzone `git ls-remote`
  w momencie wykonania.

## Weryfikacja (BEZ ComfyUI serwera, BEZ GPU)

- `pyproject.toml` parsuje się przez `tomllib`; `requires-python == ">=3.10"`,
  `version == "1.0.0"`, `PublisherId == "mu5hr00moo"`,
  `requires-comfyui == ">=0.30.0"`.
- `git grep -n "\.github/ISSUE_TEMPLATE" -- docs/` → brak trafień. Brak
  jakiegokolwiek względnego linku `](../.github` / `](.github` w docs/ i
  README.md.
- URL nowego linku: `.github/ISSUE_TEMPLATE/bug_report.yml` istnieje na
  `origin/master` (gałąź domyślna), nazwa pliku zgadza się z parametrem
  `?template=bug_report.yml`, repo jest PUBLIC, `curl -L` → HTTP 200
  (anonimowo przekierowuje na login z zachowanym `?template=...` w
  `return_to` — zalogowany użytkownik trafia prosto na formularz).
- `publish.yml` parsuje się przez `yaml.safe_load`; `push.branches ==
  ["master"]`, brak `"main"`. Oba `uses:` mają pełny 40-znakowy SHA jako
  ref (żadnego `@main`/`@v7` jako ref — te stringi zostają tylko w
  komentarzu WHY).
- Symulacja paczki (git ls-files minus wzorce `.comfyignore`): **51 plików
  w paczce, 39 wykluczonych**. `pytest.ini` już wykluczony (wcześniej był
  w paczce). Wykluczone: `tests/` (32), `.github/` (3), `pytest.ini`,
  `CLAUDE.md`, `HANDOFF.md`, `TODO.md`. `cache/` i `benchmark_results/`
  już nieśledzone przez git.
- `CHANGELOG.md`: `## [1.0.0]` zgadza się co do znaku z `version` w
  `pyproject.toml`.
- Pełny pytest w comfyenv: **399 passed / 0 failed / 0 skipped**
  (4 ostrzeżenia DeprecationWarning z transformers, niezwiązane) —
  potwierdzone ponownie po commicie b827cce.

## Ustalenia istotne dla Chat

- `requires-python = ">=3.10"` — na życzenie recenzji podniesione z `>=3.9`.
  Uzasadnienie w komentarzu w pliku i w commicie: node bez ComfyUI nie
  działa, a aktualne ComfyUI (v0.34.2, `~/ComfyUI/pyproject.toml:6`)
  wymaga `>=3.10`. Analiza składni samego repo (floor = 3.9, jedyny
  konstrukt 3.9+ to PEP 585 w `minimaxh3_clipcache/store.py:372` i
  `minimaxh3_clipcache/locking.py:24`) jest odnotowana, ale świadomie
  NIE jest podstawą wartości pola.
- Data w nagłówku `## [1.0.0]` to 2026-09-03 (dzień przygotowania PR-a).
  Jeśli merge nastąpi innego dnia — do ręcznego bumpa przed/przy mergu.
- `CHANGELOG.md` nie ma stopki z link-referencjami `[1.0.0]: .../releases/tag/...`
  — w repo nie ma jeszcze żadnego taga, więc nie zgadywano konwencji
  nazwy (`v1.0.0` vs `1.0.0`). Do dodania przy pierwszym tagu, jeśli
  potrzebne.
- W paczce zostają też `.gitignore` (drobny) — świadomie, zlecenie kazało
  zostawić.

## Otwarte pytania

- brak

## Sugestie (nie polecenia)

- Przed mergem PR #9: upewnić się, że sekret `REGISTRY_ACCESS_TOKEN` jest
  w ustawieniach repo. Merge doda `pyproject.toml` do `master`, co pasuje
  do filtra `paths` w `publish.yml` i od razu odpali publikację do
  Registry; bez tokenu pierwszy przebieg padnie.
- Rozważyć tag `v1.0.0` na commicie merge'a (spójnie z hipotetyczną
  stopką link-referencji w CHANGELOG i z `publish-node-action`, które
  wiąże wydanie Registry z wersją z `pyproject.toml`).
- `.github/workflows/tests.yml` też używa nieprzypiętych
  `actions/checkout@v7` i `actions/setup-python@v7`. Ten workflow NIE
  dostaje żadnego sekretu, więc ryzyko jest niższe niż w `publish.yml`
  i było poza zakresem PR #9. Do przypięcia osobnym PR-em, jeśli chcemy
  spójności.
