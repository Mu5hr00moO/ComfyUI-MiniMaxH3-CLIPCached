# HANDOFF

## Stan na: 2026-09-03 / branch feat/example-workflow / PR do otwarcia

## Ostatnio zrobione

Dodanie przykładowego workflow jako szablonu ComfyUI i wydanie `1.1.0`.
Gałąź `feat/example-workflow` odcięta od `origin/master` (`e47468a`,
czyli stan po merge PR #10, tag `v1.0.0`).

### Commit 1 — szablon (`4c95fec`)

- Nowy katalog `example_workflows/` z dwoma plikami o identycznej
  nazwie bazowej `MiniMax H3 T2V (CLIP-Cached)` (spacje i nawiasy
  włącznie):
  * `MiniMax H3 T2V (CLIP-Cached).json` — graf workflow
  * `MiniMax H3 T2V (CLIP-Cached).jpg` — miniatura kafelka
- Pliki wstawione bez zmiany nazw i bez zmiany zawartości. Windowsowe
  `*:Zone.Identifier` (i tak ignorowane przez `.gitignore`) usunięte z
  drzewa roboczego, nie trafiły do commita.
- Nazwa katalogu `example_workflows/` zweryfikowana w lokalnym źródle
  ComfyUI: `app/custom_node_manager.py:94,127` — to kanoniczna nazwa
  (pozostałe warianty logują "consider renaming"). Trasa
  `/api/workflow_templates/<module>` serwuje ten katalog statycznie
  (`app/custom_node_manager.py:132-138`), więc `.jpg` o tej samej
  nazwie bazowej działa jako miniatura.

### Commit 2 — wydanie 1.1.0 (`0b3ebda`)

- `pyproject.toml`: `version` `1.0.0` -> `1.1.0` (jedyna zmiana w pliku;
  semver: nowa funkcjonalność, bez breaking change).
- `CHANGELOG.md`: nowa sekcja `## [1.1.0] - 2026-09-03` z `### Added`
  (przykładowy workflow w Browse Templates + `example_workflows/`).
  Sekcja `## [Unreleased]` i jej podsekcja `### Planned` bez zmian.
  Stopka: dodany `[1.1.0]: .../releases/tag/v1.1.0`, `[Unreleased]`
  przestawiony na `compare/v1.1.0...HEAD`.
- `README.md`: nowa podsekcja `### Example Workflow` pod `## Installation`
  (2-3 zdania: gdzie szukać po instalacji i w repo). Bez innych zmian.

### Commit 3 — HANDOFF.md (osobno, w tym samym PR)

## Weryfikacja (BEZ ComfyUI serwera, BEZ GPU)

- `example_workflows/MiniMax H3 T2V (CLIP-Cached).json` parsuje się przez
  `json.load` — 6 węzłów top-level; właściwe FL2VA jest w subgraph
  `definitions.subgraphs[0]` ("Image to Video (MiniMax H3)"), którego
  węzły mają `cnr_id="comfy-core"` poza jednym `MiniMaxH3CLIPCachedFL2VA`
  (`aux_id="Mu5hr00moO/ComfyUI-MiniMaxH3-CLIPCached"`). Brak zależności
  od obcych paczek.
- Nazwy bazowe obu plików identyczne co do znaku: `MiniMax H3 T2V
  (CLIP-Cached)`.
- `pyproject.toml` parsuje się przez `tomllib`, `project.version ==
  "1.1.0"`.
- Nagłówek `## [1.1.0]` w `CHANGELOG.md` zgadza się co do znaku z
  wersją w `pyproject.toml`.
- Symulacja publikowanej paczki (pliki śledzone przez git minus
  `.comfyignore`/`.gitignore`): `51 -> 53` plików. Oba pliki z
  `example_workflows/` są w zbiorze publikowanym; `git check-ignore`
  potwierdza, że żadna reguła ignorująca ich nie łapie.
- `git diff --check` czysty.
- Pełny pytest w comfyenv: **399 passed / 0 failed / 0 skipped**
  (4 ostrzeżenia `DeprecationWarning` z `transformers`, niezwiązane).

## Ustalenia istotne dla Chat

- `origin/master` = `e47468a` (po merge PR #10), tag `v1.0.0`. Paczka w
  ComfyUI Registry: `mu5hr00moo/minimaxh3-clipcached` `1.0.0`.
- `example_workflows/` NIE jest wykluczony ani przez `.comfyignore`, ani
  przez `.gitignore` — katalog trafia do publikowanej paczki (symulacja:
  53 pliki zamiast 51).
- Filtr `paths: ["pyproject.toml"]` w `.github/workflows/publish.yml`
  oznacza, że merge tego PR-a na `master` (zmiana `pyproject.toml`)
  automatycznie odpali publikację `1.1.0` do Registry.
- Konwencja tagów: prefiks `v` (`v1.1.0`). Link `[1.1.0]` w stopce
  `CHANGELOG.md` wskazuje `releases/tag/v1.1.0` — zacznie działać
  dopiero po utworzeniu tagu i GitHub Release przez Kamila.
- Zawartość workflow (graf, prompt, ustawienia) nie była modyfikowana —
  poza zakresem zlecenia.

## Otwarte pytania

- brak

## Sugestie (nie polecenia)

- Kolejność po stronie Kamila: najpierw merge PR-a (to odpali publikację
  `1.1.0` do Registry), a tag `v1.1.0` + GitHub Release dopiero po
  merge — analogicznie jak przy `v1.0.0`.
- `.github/workflows/tests.yml` wciąż używa nieprzypiętych
  `actions/checkout@v7` i `actions/setup-python@v7` bez bloku
  `permissions:` — do przypięcia osobnym PR-em, jeśli zależy nam na
  spójności z `publish.yml`.
