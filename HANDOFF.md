# HANDOFF

## Stan na: 2026-09-03 / branch docs/changelog-links / PR do otwarcia

## Ostatnio zrobione

Krok 4 planu przed tagiem v1.0.0: stopka link-referencji w `CHANGELOG.md`.
Gałąź `docs/changelog-links` odcięta od `origin/master` (33293b6, czyli
stan po merge PR #9).

### Commit 1 — CHANGELOG.md (86c3f7b)

- Na końcu pliku dopisana stopka link-referencji w konwencji Keep a
  Changelog:
  * `[Unreleased]: https://github.com/Mu5hr00moO/ComfyUI-MiniMaxH3-CLIPCached/compare/v1.0.0...HEAD`
  * `[1.0.0]: https://github.com/Mu5hr00moO/ComfyUI-MiniMaxH3-CLIPCached/releases/tag/v1.0.0`
- Konwencja tagów ustalona jako prefiks `v` (czyli `v1.0.0`) — stąd
  wartości w stopce.
- Nagłówki sekcji `## [Unreleased]` i `## [1.0.0] - 2026-09-03`
  nietknięte — w Keep a Changelog nawiasy kwadratowe w nagłówku wiążą
  się ze stopką automatycznie.
- `pyproject.toml` NIE ruszany świadomie: push na `master` z tym plikiem
  odpaliłby ponowną publikację do Registry (filtr `paths` w
  `publish.yml`).

### Commit 2 — HANDOFF.md (osobno, w tym samym PR)

## Weryfikacja (BEZ ComfyUI serwera, BEZ GPU)

- Data w nagłówku `## [1.0.0] - 2026-09-03` potwierdzona: commit merge'a
  PR #9 (`33293b6`) ma datę `2026-09-03 08:37:36 +0200` — zgadza się,
  brak korekty.
- Wersje w stopce zgadzają się co do znaku z nagłówkami sekcji:
  `[Unreleased]` ↔ `## [Unreleased]`, `[1.0.0]` ↔ `## [1.0.0] - ...`.
- `git diff origin/master..HEAD --stat` dotyka wyłącznie `CHANGELOG.md`
  (+3) i `HANDOFF.md`.
- `git diff --check` czysty.
- Pełny pytest w comfyenv: **399 passed / 0 failed / 0 skipped**
  (4 ostrzeżenia DeprecationWarning z transformers, niezwiązane).

## Ustalenia istotne dla Chat

- `origin/master` = `33293b6` (po merge PR #9). Paczka jest już
  opublikowana w ComfyUI Registry jako `mu5hr00moo/minimaxh3-clipcached`
  `1.0.0`.
- Konwencja tagów w tym repo: prefiks `v` — `v1.0.0`. Stopka
  `CHANGELOG.md` używa tej formy.
- Stopka link-referencji nie wymagała żadnej zmiany w nagłówkach sekcji
  — Keep a Changelog wiąże `[1.0.0]` w nagłówku z `[1.0.0]:` w stopce po
  samej nazwie w nawiasach kwadratowych.
- Link `[1.0.0]` wskazuje `releases/tag/v1.0.0`, który zacznie działać
  dopiero po utworzeniu tagu i GitHub Release przez Kamila (poza
  zakresem tego PR-a).

## Otwarte pytania

- brak

## Sugestie (nie polecenia)

- Po merge tego PR-a: utworzyć tag `v1.0.0` na commicie merge'a i GitHub
  Release — dopiero wtedy link `[1.0.0]` w stopce CHANGELOG przestanie
  być 404, a link `[Unreleased]` (`compare/v1.0.0...HEAD`) zacznie
  pokazywać sensowny diff.
- `.github/workflows/tests.yml` wciąż używa nieprzypiętych
  `actions/checkout@v7` i `actions/setup-python@v7` i nie ma bloku
  `permissions:`. Nie dostaje żadnego sekretu, więc niższe ryzyko niż
  `publish.yml` — do przypięcia osobnym PR-em, jeśli chcemy spójności.
