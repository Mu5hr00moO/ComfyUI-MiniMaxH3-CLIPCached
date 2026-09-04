# HANDOFF

## Stan na: 2026-09-04 / branch `feat/cache-size-threshold` / kod w `2deb81c`

Gałąź wyszła z czystego `origin/master` (`d1863fc`, już po zmergowanym
PR #18). Dwa commity: `2deb81c` — kod, ten commit — HANDOFF.md.

Temat: tura 1 progu rozmiaru cache w Cache Managerze — logika,
persystencja i kolorowanie paska statusu. BEZ interfejsu do ustawiania
wartości; w tej turze konfiguracja jest wpisywana ręcznie do
localStorage.

## Ostatnio zrobione

### Zakres i granice funkcji

Wyłącznie warstwa wyświetlania. Limit żyje w przeglądarce: backend nigdy
się o nim nie dowiaduje, nie ma eviction, nie ma blokowania zapisu, Python
jest nietknięty.

Ocenianą liczbą jest `total_size_bytes` z `/check` — czyli CAŁY katalog
`cache/`, oba warianty węzła, miniatury i pliki-śmieci włącznie. To
dokładnie ta liczba, którą pasek statusu już drukuje. Nie istnieje jej
odpowiednik per wariant: `scan_cache()` mierzy katalog jako całość
(`minimaxh3_clipcache/scanner.py:206`, helper `_dir_size_bytes` w
`minimaxh3_clipcache/scanner.py:70-78`), a per-wpisowe `size_bytes` z
założenia nie sumują się do tej wartości
(`minimaxh3_clipcache/scanner.py:37-41`).

### Co doszło w `web/main.js`

- `parseFiniteNumber(raw, {min, max})` (`web/main.js:71`) → liczba albo
  `null`, dla wartości przychodzących z zewnątrz: pola tekstowego,
  localStorage, odpowiedzi API. Przyjmuje string i liczbę; pusty string i
  same spacje odrzuca, żeby `Number("")` nie zamieniło pustego pola w
  zero. Nowy helper, a nie rozszerzenie `positiveBytes()`
  (`web/main.js:170`) — tamten odpowiada na inne pytanie (podstawia `0`
  przy porażce) i leży na ścieżce renderowania rozmiaru wpisu, więc
  zmiana jego kontraktu dotknęłaby niepowiązanego kodu. `positiveBytes()`
  został bez zmian.
- `readCacheSizeOptions()` / `writeCacheSizeOptions()`
  (`web/main.js:1050` i `web/main.js:1068`), klucz
  `h3cm-cache-size-options`, kształt `{ limitBytes, warningPercent }`.
  Zabezpieczone tym samym wzorcem co `readLauncherPosition()`
  (`web/main.js:1762`): `null` przy braku klucza, uszkodzonym JSON-ie,
  JSON-ie który nie jest obiektem, brakującym lub nieużywalnym polu oraz
  przy niedostępnym localStorage; zapis w `try/catch` z cichą porażką.
- `classifyCacheSize(totalBytes, options)` (`web/main.js:1081`) →
  `"off" | "ok" | "warning" | "alert"`. Funkcja czysta: bez DOM, bez
  storage. Oba progi porównują przez `>=`, więc dokładne trafienie w próg
  liczy się jako jego przekroczenie; przy `warningPercent = 100` oba progi
  pokrywają się i wygrywa `alert`.
- `setCacheStatus(text, level)` (`web/main.js:1114`) jest teraz JEDYNYM
  miejscem zapisującym pasek statusu i nakłada tekst razem z poziomem.
  Powód jest konkretny, nie kosmetyczny: rozdzielone, poziom z
  poprzedniego `/check` przeżywał tekst, do którego należał, i malował na
  czerwono niezwiązany komunikat. Wszystkie pięć dotychczasowych
  przypisań `panel.statusEl.textContent` przeszło na tę funkcję i podaje
  poziom `"off"` wszędzie poza udanym `/check`. Prefiksy: `"⚠ "` dla
  `warning`, `"ALERT: "` dla `alert`.

### Co doszło w `web/styles.css`

Dwie klasy tuż za `.h3cm-status` (`web/styles.css:200-206`):
`.h3cm-status-warning` (`#f0bd69`) i `.h3cm-status-alert` (`#ef9a9a` +
`font-weight: 700`). Te same odcienie, które w tym panelu noszą już plakietki
wpisów `legacy` i `inconsistent` (`web/styles.css:379-386`), więc jeden
kolor zachowuje jedno znaczenie. Bez zmiennych CSS motywu ComfyUI — plik
ich nie używa nigdzie (0 wystąpień `var(--`, 92 zahardkodowane wartości
hex), a wprowadzanie ich pod jedną funkcję rozjechałoby się z resztą.

Reguły stoją PO `.h3cm-status` i mają tę samą specyficzność, i to
kolejność w pliku decyduje o nadpisaniu koloru.

## Weryfikacja (stan bieżący)

- `node --check` na kopii `.mjs`: czysto. `git diff --check`: czysto.
  Bilans nawiasów `web/styles.css`: 106/106.
- Pełny pytest: **446 passed** (bez zmian — ta gałąź nie rusza Pythona).
- Harness ESM w scratchpadzie, na REALNYM `web/main.js` (loader podstawia
  `/scripts/app.js` i `/scripts/api.js`, wymusza format `module` i dopisuje
  `export` dla prywatnych `setCacheStatus`, `runCheck`, `toggleFavorite`,
  `deleteEntry` oraz settera na module-level `panel` — testowany jest więc
  bieżący kod, nie kopia; harness i loader NIE commitowane) — **65
  asercji PASS**. Pokryte: wszystkie gałęzie klasyfikatora, dokładne
  trafienie w każdy próg (i o bajt poniżej), `limitBytes` zero i ujemny,
  `totalBytes` zero i nieużywalny, `warningPercent` 1 i 100, brak klucza w
  localStorage, pusta wartość, uszkodzony JSON, JSON nie-obiekt, brakujące
  i zakresowo złe pola, localStorage rzucający wyjątkiem przy odczycie i
  przy zapisie, round-trip zapis/odczyt, prefiksy i klasy dla każdego
  poziomu, oraz `runCheck()` przejechane end-to-end na podstawionym
  `/check` (mały cache → bez klasy, powyżej progu → `warning`, powyżej
  limitu → `alert`).
- Asercja regresyjna została sprawdzona przez mutację: trzy przypisania
  statusu przywrócone do postaci sprzed zmiany
  (`panel.statusEl.textContent = …`) i harness wtedy FAILuje — klasa
  `h3cm-status-alert` zostaje na komunikatach „Update failed” i „Delete
  failed”, a linia „Cache: checking…” startuje z odziedziczonym alarmem.
  Po przywróceniu kodu 65/65 PASS.
- Uwaga na przyszłość: ścieżka NIEUDANEGO `/check` jest dodatkowo
  osłonięta resetem „Cache: checking…”, którym otwiera się `runCheck()`,
  więc sama w sobie czyściłaby klasę nawet bez poziomu. Realnie
  niezabezpieczone bez `setCacheStatus()` są dwa inne miejsca — „Update
  failed” w `toggleFavorite()` i „Delete failed” w `deleteEntry()` — i to
  one są testowane w sekcji 5b harnessu.

## NIE zweryfikowane (do sprawdzenia przez Kamila w żywym ComfyUI)

- Realny wygląd obu kolorów na pasku statusu i czytelność pogrubionego
  `ALERT:` w ciemnym motywie ComfyUI.
- Że wpisanie konfiguracji ręcznie w konsoli przeglądarki działa
  end-to-end, np.:
  `localStorage.setItem("h3cm-cache-size-options", JSON.stringify({limitBytes: 10*1024**3, warningPercent: 80}))`
  a następnie kliknięcie „Check”.
- Przetrwanie ustawienia przez odświeżenie strony.
- Brak błędów w konsoli przeglądarki.

## Ustalenia istotne dla Chat

- `total_size_bytes` i `total_count` z `/check` są ślepe na wariant i na
  `favorite`: `scan_cache()` zwraca je bez filtrów
  (`minimaxh3_clipcache/scanner.py:203-207`), a `check()` oddaje wynik
  verbatim, doklejając wyłącznie `last_used`
  (`minimaxh3_clipcache/routes.py:126-140`).
- `setCacheStatus()` jest funkcją prywatną modułu, bo operuje na
  module-level `panel`. Harness sięga po nią przez loader hook, nie przez
  eksport — `web/main.js` nie ma kodu istniejącego wyłącznie dla testów.
- Czyste helpery (`parseFiniteNumber`, `classifyCacheSize`,
  `readCacheSizeOptions`, `writeCacheSizeOptions`) są eksportowane zgodnie
  z konwencją tego pliku, w którym helpery dotykające localStorage
  (`readLauncherPosition`, `writeLauncherPosition`) też są eksportowane.
- Tura 1 nie dodaje żadnej kontrolki UI. Wartości ustawia się ręcznie w
  localStorage; interfejs do ich wpisywania jest tematem tury 2.

## Otwarte pytania

- brak

## Sugestie (nie polecenia)

- (sugestia) W turze 2, przy polu na limit, warto zdecydować w jakiej
  jednostce użytkownik wpisuje wartość. `limitBytes` jest w bajtach, więc
  pole „GB” wymaga przeliczenia przy zapisie i przy odczycie do pola —
  najlepiej w jednym miejscu, obok `readCacheSizeOptions()`, żeby format
  na dysku pozostał jednoznaczny.
- (sugestia) `writeCacheSizeOptions()` nie waliduje tego, co dostaje —
  waliduje dopiero odczyt. Przy dokładaniu UI warto walidować na wejściu
  (przez `parseFiniteNumber`) i nie zapisywać wartości, których odczyt i
  tak odrzuci, bo inaczej pole może wyglądać na zapisane, a po odświeżeniu
  wrócić do „brak limitu”.
