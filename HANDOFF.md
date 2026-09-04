# HANDOFF

## Stan na: 2026-09-04 / branch feat/entry-size-in-meta-line / PR #17 (open)

## Ostatnio zrobione

PR #17 (rozmiar wpisu w Cache Managerze) plus domknięcie dwóch uwag review
bota. PR zostaje OTWARTY. Gałąź ma teraz sześć commitów:

1. `2de30b5` — backend (`size_bytes` per wpis) + frontend (formatowanie,
   sumowanie przy dual-res) + testy.
2. `f4896f0` — HANDOFF.md.
3. `bf3b855` — oznaczenie sumy pary na ekranie + zdanie w tooltipie.
4. `d50f05a` — rozmiar w wierszach legacy / inconsistent.
5. `5589af4` — dokumentacja (`docs/CACHE_MANAGER.md` + `CHANGELOG.md`).
6. ten commit — HANDOFF.md (osobno, ten sam PR).

### Uwaga 1: suma pary vs zakres Delete (`bf3b855`)

Problem był realny: `deleteEntry()` bierze JEDEN fingerprint (panel
szczegółów woła go z własnym — `web/main.js:1582`, pasek partnera ze swoim
— `web/main.js:811`), a wiersz pokazywał sumę obu stron. Suma ZOSTAJE (o to
prosił Kamil); naprawą jest oznaczenie liczby.

Na ekranie, przy złożonej parze:

    04.09.2026, 01:53 · 768×1376 (1.06 MP) - 1.5 GB (pair total)

Oznaczenie to dosłownie ` (pair total)` doklejone po rozmiarze, w treści
linii — widoczne bez najeżdżania myszą. Pojawia się dokładnie tam, gdzie
pojawia się suma: wiersz listy i panel szczegółów. Pasek partnera i wpis
bez ważnego parowania oznaczenia NIE dostają.

`entryDisplaySizeBytes()` → `entryDisplaySize()`, zwraca teraz
`{ bytes, isPairTotal }`. Flaga jedzie RAZEM z liczbą, zamiast być
wyprowadzana ponownie w każdym miejscu wywołania — dzięki temu liczba i
oznaczenie nie mogą się rozjechać. `formatEntryMetaLine(system, sizeBytes,
{ isPairTotal })` dalej tylko formatuje i nigdy nie sięga po parowanie.

### Uwaga 2: tooltip (`bf3b855`)

`entryMetaTooltip(system, size)` składa się teraz z niezależnych zdań, po
jednym na pole, które wymaga wyjaśnienia. Pełna treść przy złożonej parze
z rozdzielczością (dwa zdania sklejone spacją):

> Creation date and generation resolution come from two different moments:
> the date is fixed when the entry is first written, while the resolution
> is that of the most recent run that used it. One cached encode serves
> every resolution when no keyframes are connected -- the encode itself
> does not depend on width/height. The size shown is the pair total: this
> entry plus its rescaled partner. Delete removes only this entry -- the
> partner is listed under the "+ rescaled to" badge and has its own Delete
> button.

Zwykły wpis dostaje samo pierwsze zdanie (bez zmian względem stanu sprzed
PR). Warunek `return ""` był FAKTYCZNIE niepoprawny po dodaniu zdania o
rozmiarze i został naprawiony: stara wersja zwracała `""` zawsze, gdy nie
było rozdzielczości, więc złożona para bez `width`/`height` straciłaby
ostrzeżenie. Teraz zdanie o parze jest bramkowane flagą `size.isPairTotal`,
nie rozdzielczością; `""` wraca tylko wtedy, gdy nie ma czego wyjaśniać.

### Uwaga 3: rozmiar w wierszach legacy / inconsistent (`d50f05a`)

Te wiersze buduje `buildSimpleRow()` (`web/main.js:700`), które nie
przechodzi przez `formatEntryMetaLine()`, więc `size_bytes` z backendu
nigdzie się nie pokazywało. Rozmiar siedzi teraz obok hintu, przed
przyciskiem Delete, tym samym `formatBytes()`; brak rozmiaru albo 0 → nic
nie renderujemy. Nowa klasa `.h3cm-row-size` w `web/styles.css` (ten sam
wygląd co `.h3cm-row-created`).

Sumowanie tych wierszy nie dotyczy i dotyczyć nie może: `renderList()`
kieruje obie klasyfikacje do `buildSimpleRow()` ZANIM policzy parowanie, a
`resolvePairing()` i tak odmówiłoby złożenia pary (wymaga czytelnego wpisu
`normal` po obu stronach). To, co widać w takim wierszu, jest dokładnie
tym, co zwolni jego Delete.

### Dokumentacja (`5589af4`)

- `docs/CACHE_MANAGER.md`: rozmiar dopisany do listy pól wpisu, nowa
  sekcja "Entry size" (co liczy, dlaczego total w nagłówku jest nieco
  większy niż suma wpisów, suma przy złożonej parze), oraz jasne
  stwierdzenie w sekcji o kasowaniu, że Delete zawsze działa na jeden wpis
  i gdzie jest przycisk partnera.
- `CHANGELOG.md`: sekcja `[Unreleased]` nie miała `Added` w ogóle. Dodane:
  rozmiar wpisu + oznaczenie sumy pary ORAZ proweniencja referencji z PR
  #14/#15/#16 (nazwy plików z grafu, nazwa slotu, prezentacja w panelu
  szczegółów) — tego w pliku nie było.

## Weryfikacja

- Pełny pytest: **446 passed**, 0 skipped (bez zmian — ta paczka nie rusza
  Pythona).
- `node --check` na kopii `.mjs`: czysto. Bilans klamer `web/styles.css`:
  0. `git diff --check`: czysto.
- Scratchpadowy harness ESM na REALNYM `web/main.js` (loader podstawia
  `/scripts/app.js` i `/scripts/api.js` i dopisuje `export` do trzech
  prywatnych builderów wierszy, więc testowany jest bieżący kod, nie
  kopia; harness i loader NIE commitowane) — **27 asercji PASS** (było
  16). Nowe pokrycie:
  - oznaczenie jest przy sumie, nie ma go przy własnym rozmiarze, nie ma
    go gdy nie ma bajtów do pokazania, i deskryptor z `entryDisplaySize()`
    da się podać wprost do formattera;
  - `entryDisplaySize()` zwraca `{bytes, isPairTotal}` dla wszystkich
    statusów parowania;
  - tooltip: bez zmian dla zwykłego wpisu, zdanie o parze doklejone przy
    złożonej parze, zdanie o parze obecne TAKŻE gdy wpis nie ma
    rozdzielczości, `""` gdy nie ma czego wyjaśniać;
  - realny wiersz legacy i realny wiersz inconsistent zbudowane przez
    własne buildery modułu: tekst spanu rozmiaru, jego pozycja przed
    przyciskiem Delete, brak spanu przy `size_bytes` brakującym / 0 /
    ujemnym / nieliczbowym.

## NIE zweryfikowane (do sprawdzenia przez Kamila w żywym ComfyUI)

- Jak `(pair total)` wygląda w wierszu listy — czy nie rozpycha wiersza
  przy długiej nazwie/prompcie (wiersz ma `flex-wrap`, więc w najgorszym
  razie powinien się zawinąć, nie rozjechać).
- Pełna treść tooltipa po najechaniu na linię meta złożonej pary.
- Rozmiar w realnym wierszu legacy / inconsistent (wymaga takiego wpisu w
  cache; obecny `cache/` ma 29 wpisów, wszystkie `normal`).
- Brak błędów w konsoli przeglądarki.

## Ustalenia istotne dla Chat

- `deleteEntry()` przyjmuje JEDEN fingerprint i nie kaskaduje — to jest
  źródło rozjazdu z sumą pary. Panel szczegółów: `web/main.js:1582`, pasek
  partnera: `web/main.js:811`. Nie zmienione; oznaczona została liczba.
- `entryDisplaySize()` (`web/main.js:175`) zwraca `{bytes, isPairTotal}`;
  `isPairTotal` jest `true` WYŁĄCZNIE dla `resolvePairing()` o statusie
  `"valid"`.
- `entryMetaTooltip(system, size)` (`web/main.js:126`) skleja niezależne
  zdania; zdanie o parze jest bramkowane flagą, a nie obecnością
  rozdzielczości.
- `buildSimpleRow()` (`web/main.js:700`) pokazuje `entryOwnSizeBytes()` —
  nigdy sumy, bo te klasyfikacje nie mogą wejść w złożoną parę.
- Backend bez zmian w tej paczce: `scan_cache()` dalej podaje `size_bytes`
  dla wszystkich trzech klasyfikacji (`minimaxh3_clipcache/scanner.py`).

## Otwarte pytania

- brak

## Sugestie (nie polecenia)

- Wiersz listy zwykłego wpisu nie ma dziś własnego przycisku Delete
  (kasowanie idzie przez panel szczegółów), więc rozjazd „suma vs jeden
  fingerprint” jest tam tylko koncepcyjny. Gdyby kiedyś dochodził Delete
  bezpośrednio w wierszu, warto od razu przemyśleć, czy nie powinien
  proponować skasowania obu stron pary.
- `README.md` ma sekcję o Cache Managerze ze zrzutami ekranu; zrzuty są
  sprzed tej zmiany i nie pokazują rozmiaru. Odświeżenie ich to zadanie
  dla Kamila (wymaga żywego ComfyUI), nie do zautomatyzowania.
