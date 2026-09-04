# HANDOFF

## Stan na: 2026-09-04 / branch `docs/provenance-scope-correction`, trzeci commit (na `5fcad31`)

Ten commit domyka poprawkę dokumentacji o prowenancji referencji: naprawia
te same, dotychczas nietknięte, nieaktualne twierdzenia w `docs/`, które
poprzedni handoff zostawił jako otwarte pytania. Nadal wyłącznie
dokumentacja — `git diff --stat` względem `origin/master` pokazuje
`CLAUDE.md`, `TODO.md`, `docs/CACHE_MANAGER.md`,
`docs/TESTING_AND_LIMITATIONS.md`, `HANDOFF.md` — żadnego `.py`, `.js` ani
`.css`. Testów nie ruszano i nie uruchamiano (zero zmian wykonywalnych).

PR #20 nadal otwarty, teraz z trzema commitami na tej gałęzi
(`1f595a5`, `5fcad31`, ten).

## Ostatnio zrobione

- `TODO.md`, wpis o FL2VA: dopisany akapit o efekcie ubocznym dodania
  bloku `hidden` — zadeklarowanie `UNIQUE_ID` wciąga id węzła do
  in-memory execution-cache ComfyUI, więc przebudowany/przenumerowany
  graf przestaje reużywać RAM-owego wyniku dla tego węzła. Dla Ref2VA to
  zaakceptowany, nieszkodliwy koszt (cache dyskowy kluczowany
  fingerprintem, nie id węzła); dla FL2VA to samo rozumowanie jest
  zapisane jako coś do potwierdzenia przy implementacji, nie założenia
  z góry.
- `docs/TESTING_AND_LIMITATIONS.md`: sekcja o nieśledzonych nazwach
  plików referencji przepisana. Stary tytuł i treść były nieprawdziwe dla
  Ref2VA w całości (łącznie ze zdaniem o świadomym unikaniu introspekcji
  grafu). Nowa treść: Ref2VA śledzi źródło (gdy da się je prześledzić do
  loadera), FL2VA nie.
- `docs/CACHE_MANAGER.md:73-74`: zdanie o nieprzechowywaniu nazw plików
  rozdzielone na dwie części — Ref2VA (dziś częściowo pokazuje źródło) i
  FL2VA (nadal nie), z zachowanym zdaniem o braku automatycznego
  przywracania plików do inputów (prawdziwe dla obu).
- `CLAUDE.md`: do historycznego akapitu (niezmienionego poza tym)
  dopisana jedna adnotacja, że `MANAGER_TODO_ref2video.md` był
  dokumentem roboczym i nigdy nie trafił do tego repozytorium
  (`git log --all` po tej ścieżce jest pusty) — nie był skasowanym
  plikiem.
- Sprawdzone `README.md` i cały `docs/` pod kątem tych samych twierdzeń:
  poza `docs/CACHE_MANAGER.md` i `docs/TESTING_AND_LIMITATIONS.md` nic
  więcej nie znaleziono (trafienia w `docs/TECHNICAL_DETAILS.md` dotyczą
  identyfikacji pliku encodera, nie referencji, i są niezwiązane).

## Ustalenia istotne dla Chat

- Wszystkie trzy otwarte pytania z poprzedniego handoffu zostały
  rozstrzygnięte poprawką w tym samym miejscu (ta sama gałąź, ten sam
  PR), zgodnie z decyzją, że to ta sama nieprawda w trzech miejscach —
  nie osobna gałąź.
- Techniczne fakty o `provenance.py` / `_sync_ref_sources()` /
  `_ref2va_hidden_input_spec()` (nodes.py:936, 283, 1121, 1255) —
  niezmienione od poprzedniego handoffu, patrz historia PR #20.

## Otwarte pytania

Brak nowych. Zakres z poprzedniego handoffu — trzy nieaktualne
twierdzenia w `docs/` i dangling odwołanie do
`MANAGER_TODO_ref2video.md` — jest tym commitem zamknięty.
