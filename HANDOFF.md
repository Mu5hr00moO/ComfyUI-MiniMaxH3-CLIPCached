# HANDOFF

## Stan na: 2026-09-01 / branch master / commit 3526626

## Ostatnio zrobione (2 poprawki z audytu w web/main.js -- pairing + detail panel)

Czysto frontend (guideline #16). Zero zmian w Pythonie/backendzie. Pełny
pytest bez zmian: 360 passed w 12.2s (przed i po -- nie mógł się ruszyć,
bo nic w Pythonie nie tknięte). `node --check` na kopii .mjs OK po każdym
commicie; `git diff --check` czysto.

Trzy commity, jeden na punkt:

### 1. resolvePairing() -- nie składaj pary, gdy któraś strona nie jest "normal" (commit 2b6aafd)

- `resolvePairing()` (`web/main.js:240`) walidowało wzajemne wskaźniki
  `paired_fingerprint` i przeciwne role `is_upscale_target`, ale ignorowało
  `entry.classification`. Wpis "normal" mógł zostać złożony w jeden wiersz
  z partnerem "inconsistent" (uszkodzone/niespójne pliki core), którego
  sidecar `verbose.json` nadal niósł poprawny wskaźnik zwrotny -- w obu
  kierunkach:
  - normal BASE + inconsistent UPSCALE: wiersz base pokazywał badge
    "+ rescaled to WxH" dla partnera, którego `store.load_conditioning()`
    by odrzucił.
  - inconsistent BASE + normal UPSCALE: dobry wpis normal UPSCALE znikał
    z listy w całości (`renderList()` robi `continue` na upscale-stronie
    "valid" pary), a inconsistent base renderował się przez
    `buildInconsistentRow()`, który nic nie wie o parowaniu.
- Nowy status `"inconsistent-pair"` zwracany, chyba że OBIE strony są
  `classification === "normal"` z czytelnym sidecarem (`!!entry.verbose &&
  !!partner.verbose`). Ten status niczego nie składa -- każda strona
  renderuje się jako własny jawny wiersz. Sprawdzenie wstawione PO
  weryfikacji wzajemnego wskaźnika, PRZED odczytem ról.
- Konsumenci `resolvePairing()` (wszyscy w `renderList()` / `buildNormalRow()`)
  bramkują się na `status === "valid"` albo `"orphaned"`, więc
  `"inconsistent-pair"` po prostu przechodzi w zwykły wiersz.

### 2. reattachOpenDetailAfterRender() -- zamknij detal, gdy wpis nie ma wyrenderowanego wiersza (commit ad76b41)

- Funkcja decydowała re-attach vs. close na podstawie przynależności
  otwartego wpisu do `filtered` (podzbiór po search/tag/favorite). To nie
  to samo, co faktyczne wyrenderowanie wiersza: `renderList()` nie tworzy
  wiersza dla upscale-strony poprawnej pary dual-res. Jeśli wpis z otwartym
  panelem detali stał się tą ukrytą stroną (późniejszy dual-res run go
  sparował), zostawał w `filtered`, `populateDetail()` wypełniał panel, a
  `attachDetailAfterRow()` po cichu nie znajdował kotwicy -- panel zostawał
  odłączony od DOM, oznaczony jako otwarty, z nieaktualną treścią.
- `attachDetailAfterRow()` zwraca teraz `rowEl !== null` (czy znalazł
  wiersz); `reattachOpenDetailAfterRender()` woła `closeDetail()`, gdy nie
  znalazł, i nie bierze już nieużywanego argumentu `filtered` (oba
  wywołania w `renderList()` zaktualizowane).

### 3. TODO.md -- usunięty domknięty edge case (commit 3526626)

- Sekcja "Cache Manager UI -- known edge cases in dual-resolution pairing
  display" miała 2 punkty. Punkt 2 (inconsistent base chowa normal upscale)
  jest naprawiony przez commit 2b6aafd -- usunięty. Punkt 1
  (search chowa BASE stronę, UPSCALE nadal pasuje -> nic się nie renderuje)
  NIE jest ruszany tą zmianą (obie strony "normal", `resolvePairing()`
  dalej zwraca "valid") -- zostaje jako jedyny otwarty przypadek. Drugi
  kierunek (normal base + inconsistent upscale) NIE dopisany jako nowy
  TODO, bo też naprawiony.

## Ustalenia istotne dla Chat

- `resolvePairing()` `web/main.js:240` -- pełny doc-comment nad funkcją
  wylicza teraz statusy: `none` / `valid` / `orphaned` / `role-unknown` /
  `inconsistent-pair` (nowy).
- Nowy status jest realnie osiągalny tylko wtedy, gdy partner ma
  `classification === "inconsistent"` z czytelnym sidecarem niosącym
  wskaźnik zwrotny. Partner "legacy" / z uszkodzonym `verbose.json` daje
  `partnerSystem = {}` -> kontrola wskaźnika zwrotnego zawodzi wcześniej
  -> `"orphaned"`, nie `"inconsistent-pair"`.
- `renderList()` gwarantuje, że do `resolvePairing()` trafia tylko wpis
  `classification === "normal"` z `verbose` (inconsistent i legacy są
  przechwytywane wcześniej w pętli), więc kontrole na `entry` w `bothNormal`
  są defensywne -- funkcja jest eksportowana i wołana wprost z testów.
- Weryfikacja: scratchpadowy harness Node ESM + jsdom (16 asercji: S1
  inconsistent BASE + normal UPSCALE, S2 normal BASE + inconsistent
  UPSCALE, S3 detal otwarty na wpisie, który staje się ukrytą upscale-stroną
  pary -> panel się zamyka; plus kontrole: valid pair dalej się składa,
  detal na BASE stronie przeżywa re-check). Wszystkie 3 scenariusze
  regresji potwierdzone jako FAIL na kodzie sprzed poprawek, PASS po.
  Harness NIE jest commitowany (stała konwencja repo -- brak commitowanego
  JS test suite).

## Otwarte pytania

- OTWARTE_PYTANIA_DO_CLAUDE w raporcie: ZLECENIE punkt 4 / KRYTERIUM_DONE
  mówi o "testach" w zakresie "web/main.js + jego testów". Repo nigdy nie
  miało commitowanego JS test suite (brak package.json, brak .mjs w
  historii git, plan o tym nie wspomina) -- ustalona konwencja to
  niecommitowany scratchpad harness + zapis w HANDOFF/CLAUDE.md. Zrobione
  zgodnie z tą konwencją. Jeśli Chat chce jednak commitowany plik testu
  JS (nowa decyzja projektowa, wykracza poza "czysto frontend, mały
  zakres"), trzeba to potwierdzić osobno.

## Sugestie (nie polecenia)

- Nie było potrzeby dodawać nowego badge dla `"inconsistent-pair"` --
  strona normal renderuje się jako zwykły wiersz, strona inconsistent
  ma już własny wiersz z badge "inconsistent". Gdyby w praktyce mylące
  było "dlaczego ten prompt jest na liście dwa razy", drobnym dodatkiem
  byłby tekstowy hint na wierszu normal ("paired entry is inconsistent")
  -- analogicznie do istniejącego badge "⚠ pairing partner missing" dla
  statusu "orphaned". Świadomie pominięte (minimalny zakres).
