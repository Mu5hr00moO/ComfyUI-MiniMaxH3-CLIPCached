# HANDOFF

## Stan na: 2026-09-04 / branch feat/cache-manager-ref-provenance-ui / PR #16 (open)

## Ostatnio zrobione

Cache Manager UI (frontend only, `web/main.js` + `web/styles.css`; zero
zmian w Pythonie). PR #16 ma teraz trzy commity:

1. `fac4020` — panel szczegółów pokazuje nazwy plików źródłowych każdej
   referencji Ref2VA (złączone z `system.ref_sources` po nazwie slotu) +
   skrócony fingerprint wpisu (12 znaków hex) w pasie akcji.
2. `f359b0d` — HANDOFF.md.
3. `f0c4c51` (nowy) — kosmetyka nazw plików + naprawa timerów feedbacku
   kopiowania (opis niżej).
4. ten commit — HANDOFF.md (osobno, ten sam PR).

### Commit `f0c4c51` — kosmetyka nazw plików + timery

`web/styles.css`:

- `.h3cm-detail-ref-sources`: `max-width: 108px` → `width: 120px` (stała),
  `align-items: center` → `stretch`. Kontener jest teraz przewidywalnej
  szerokości; komórka `.h3cm-detail-ref-cell` przez to zawsze 120px szeroka,
  miniatura 64px i etykieta pozycyjna dalej wyśrodkowane.
- `.h3cm-detail-ref-file`: `overflow-wrap: anywhere` (zawijanie) → `display:
  block; overflow: hidden; text-overflow: ellipsis; white-space: nowrap`.
  Każda nazwa to teraz dokładnie jedna linia przycięta wielokropkiem;
  `text-align: center` wciąż centruje krótkie nazwy i "no file source".
  Wysokość kafelka nie zależy już od długości nazwy ani liczby tropów.

`web/main.js` — `buildRefSourceLines()`:

- `title` liczony raz jako `fullTitle`: gdy jest `path` →
  `` `${annotated}\n${path}` `` (nazwa nad ścieżką, w jednym title); gdy
  brak `path` → sam `annotated`. Pozycja bez `path` dostaje więc teraz
  tooltip (wcześniej nie miała żadnego), z samą nazwą.
- Klik dalej kopiuje SAMĄ ścieżkę (`copyToClipboardWithFeedback(line, path,
  fullTitle)` — 2. arg = kopiowana wartość). 3. arg (revert title) to teraz
  `fullTitle`, nie goła ścieżka — inaczej po pierwszym skopiowaniu tooltip
  gubiłby nazwę pliku.
- Pozycja bez `path` dalej niekopiowalna (brak `is-copyable`, `role`,
  listenerów).

`web/main.js` — `copyToClipboardWithFeedback(el, text, revertTitle)`:

- Na wejściu funkcji anuluje poprzedni timer revertu trzymany na
  `el._h3cmCopyRevertTimer` (i zeruje uchwyt) — obejmuje obie ścieżki
  (sukces i „copy failed”). Nowy `setTimeout` zapisuje uchwyt z powrotem
  na element; callback zeruje go po odpaleniu. Drugie kliknięcie w ciągu
  1,5 s dostaje pełne 1,5 s potwierdzenia zamiast ucięcia przez timer
  pierwszego kliknięcia. Element i tak jest wyrzucany przy przebudowie
  panelu, więc uchwyt na elemencie jest OK.

## Weryfikacja (commit `f0c4c51`)

- `node --check` (kopia `.mjs`): czysto.
- `git diff --check`: czysto.
- Bilans klamer `web/styles.css`: 0 (zbalansowane).
- Pełny pytest: **438 passed** (bez zmian w Pythonie, uruchomione dla
  pewności).
- Scratchpadowy harness ESM (loader podstawia `/scripts/app.js` +
  `/scripts/api.js`, minimalne `document`/`navigator`, kontrolowane
  `setTimeout`/`clearTimeout` i schowek; harness i loader NIE commitowane;
  loader dopisuje `export` do dwóch prywatnych funkcji, więc testowany jest
  REALNY bieżący kod, nie kopia) — **26 asercji PASS**, w tym:
  - moduł importuje się bez wyjątku ze stubami ComfyUI;
  - `buildRefSourceLines`: linia z `path` → `title` = `annotated \n path`,
    `is-copyable` + `role=button`; linia bez `path` → `title` = sam
    `annotated`, brak `is-copyable`/`role`/listenera kliknięcia; `[]` →
    jedna linia "no file source" `is-empty` (bez zmian);
  - klik → do schowka trafia SAMA ścieżka (nie title); `title` = "Copied!"
    w oknie; po timerze `title` wraca do `annotated \n path` (NIE do gołej
    ścieżki); uchwyt timera wyzerowany po odpaleniu;
  - drugie kliknięcie w oknie 1,5 s: timer pierwszego kliknięcia
    wyczyszczony, nowy timer zaplanowany, `title` dalej "Copied!" (brak
    przedwczesnego revertu), dopiero drugi timer robi revert;
  - nieudane kopiowanie po udanym też czyści zawisły timer, pokazuje
    „Copy failed…”, nie planuje nowego timera;
  - `detailRefCells` bez regresji (etykiety pozycyjne per typ mimo luki w
    slotach, złączenie tropów po slocie).

## NIE zweryfikowane (do sprawdzenia przez Kamila w żywym ComfyUI)

- Realny render: nazwy plików jako jedna przycięta linia w kafelku Ref2VA,
  stabilna wysokość kafelków przy długich nazwach i przy fan-inie N>1.
- Tooltip przy najechaniu: dwie linie (nazwa nad ścieżką) dla pozycji z
  `path`; jedna linia (sama nazwa) dla pozycji bez `path`.
- Klik → schowek zawiera ścieżkę, "Copied!" przez 1,5 s, po dwóch szybkich
  kliknięciach drugie dostaje pełne 1,5 s.
- Siatka kafelków (120px) nie rozjeżdża panelu w poziomie; miniatura i
  etykieta pozycyjna dalej wyśrodkowane.
- Brak błędów w konsoli przeglądarki.

## Ustalenia istotne dla Chat

- Nazwa pliku referencji w panelu szczegółów to teraz jedna linia
  przycięta wielokropkiem w stałym boksie 120px (`web/styles.css`
  `.h3cm-detail-ref-sources` / `.h3cm-detail-ref-file`). Wysokość kafelka
  nie zależy już od długości nazwy ani liczby tropów.
- `title` linii tropu niesie pełne dane: `annotated` + `\n` + `path` gdy
  ścieżka jest, sam `annotated` gdy jej nie ma. Pozycja bez `path` ma
  teraz tooltip (wcześniej brak).
- Klik kopiuje wyłącznie `path` (bez zmian). Trzeci argument
  `copyToClipboardWithFeedback` (revert title) to teraz pełny `title`, nie
  goła ścieżka — `buildRefSourceLines()` w `web/main.js`.
- `copyToClipboardWithFeedback` anuluje zawisły timer revertu (uchwyt
  `el._h3cmCopyRevertTimer`) na wejściu — poprawka jest w samej
  współdzielonej funkcji, dotyczy każdego jej wywołania.

## Otwarte pytania

- brak

## Sugestie (nie polecenia)

- `copyPromptText()` (ikona kopiuj na boksie promptu) ma własny, osobny
  `setTimeout` bez anulowania — ten sam drobny efekt nakładających się
  timerów, poza zakresem tego zlecenia. Można przy okazji ujednolicić z
  `copyToClipboardWithFeedback`.
