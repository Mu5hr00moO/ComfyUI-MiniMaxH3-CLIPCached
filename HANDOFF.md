# HANDOFF

## Stan na: 2026-09-04 / branch fix/copy-prompt-feedback-timer (od origin/master 23f3f72)

PR #17 (rozmiar wpisu) został w międzyczasie zmergowany do `master`
(`23f3f72`), więc ta gałąź startuje z czystego `origin/master` i nie ma z
nim kolizji — `copyPromptText()` nie był w PR #17 dotykany.

## Ostatnio zrobione

Ujednolicenie feedbacku kopiowania w Cache Managerze. Gałąź ma dwa
commity: `89e543c` (kod) i ten (HANDOFF.md).

### Problem

`copyPromptText()` (ikonka kopiowania przy prompcie w panelu szczegółów)
trzymała własną kopię mechaniki „Copied!”: dodanie klasy `is-copied`,
podmiana `title`, `setTimeout` na revert po 1,5 s. Bez anulowania timera z
poprzedniego kliknięcia — czyli dokładnie ten sam defekt, który dla
`copyToClipboardWithFeedback()` naprawił PR #16. Drugie kliknięcie w ciągu
okna dostawało skrócone potwierdzenie, bo timer z pierwszego kliknięcia
odpalał w jego środku.

### Czego NIE dało się zrobić: pełne reużycie `copyToClipboardWithFeedback()`

`copyToClipboardWithFeedback(el, text, revertTitle)` sama wykonuje zapis do
schowka. `copyPromptText()` nie ma tekstu promptu — deleguje do
`copyPrompt()` (`web/main.js:1540`), która robi zapis do schowka RAZEM z
efektami ubocznymi, których ten przycisk nie jest właścicielem:
`renderCopyResult()` (panel referencji pod promptem) i ustawienie
`[data-h3cm-refs-hint]` w pasku narzędzi. Żeby oddać zapis wspólnej
funkcji, trzeba by albo zduplikować w `copyPromptText()` wyszukiwanie
wpisu i te dwa efekty uboczne (duplikacja większa niż ta usuwana), albo
dorobić `copyToClipboardWithFeedback()` parametr-callback z operacją
kopiowania — czyli naginanie wspólnej funkcji pod jednego wołającego,
czego zlecenie zakazywało.

Dochodzi druga, mniejsza różnica: na ścieżce błędu
`copyToClipboardWithFeedback()` NIE zdejmuje klasy `is-copied` (wychodzi
przez `return false` zaraz po ustawieniu tytułu), a `copyPromptText()`
zdejmowała ją przez `toggle("is-copied", ok)`. Sklejenie obu w jedno
wywołanie wymusiłoby zmianę zachowania jednej ze stron, a zlecenie
wymagało zachowania widocznego dla użytkownika bez zmian.

### Co zostało zrobione zamiast tego

Sama afordancja została wyciągnięta do dwóch małych helperów, z których
korzystają OBIE funkcje — jedna implementacja mechaniki zamiast dwóch:

- `cancelCopyRevert(el)` — anuluje wiszący `_h3cmCopyRevertTimer`;
- `markCopied(el, revertTitle)` — dodaje klasę, ustawia „Copied!”,
  planuje revert po `COPY_FEEDBACK_MS`.

Podział na dwie połówki jest celowy: anulowanie MUSI się wykonać PRZED
próbą kopiowania (jest asynchroniczna — zapis do schowka może czekać na
zgodę użytkownika, a wtedy stary timer odpaliłby w trakcie `await` i dał
mignięcie), a potwierdzenie ma sens dopiero po niej. Wspólne stałe:
`COPY_FEEDBACK_MS = 1500`, `COPY_FAILED_TITLE`.

Zachowanie widoczne dla użytkownika: bez zmian poza samą naprawą — ta sama
klasa, te same teksty, te same 1,5 s. Efekt uboczny naprawy na ścieżce
błędu `copyPromptText()`: stary timer nie nadpisze już tytułu „Copy failed
- select the text manually” po 1,5 s (wcześniej nadpisywał, wracając do
„Copy prompt”).

### Inne miejsca z tym wzorcem (punkt 3 zlecenia)

W `web/main.js` są dokładnie DWA `setTimeout`. Drugi (`web/main.js:1730`,
w `endDrag()` pływającego launchera) to inny wzorzec i NIE ma tego błędu:
zeruje flagę `suppressClick` z opóźnieniem 0 ms, nie revertuje żadnego
tytułu ani klasy, a flaga jest niezależnie zerowana przy każdym
`pointerdown` (`web/main.js:1698`) i w samym handlerze `click`. Nakładające
się timery nie mogą tam zostawić złego stanu. Zostawione bez zmian.
Poza tym w pliku nie ma `setInterval` ani `requestAnimationFrame`.

## Weryfikacja

- `node --check` na kopii `.mjs`: czysto. `git diff --check`: czysto.
- Pełny pytest: **446 passed** (bez zmian — ta paczka nie rusza Pythona).
- Scratchpadowy harness ESM na REALNYM `web/main.js` (loader podstawia
  `/scripts/app.js` i `/scripts/api.js`, dopisuje `export` dla prywatnej
  `copyPromptText`, wirtualny zegar zamiast prawdziwych timerów; harness
  NIE commitowany) — **18 asercji PASS**:
  - moduł importuje się bez wyjątku, oba helpery są eksportowane;
  - jedno kliknięcie: potwierdzenie widoczne, revert dokładnie na 1500 ms,
    uchwyt timera wyzerowany;
  - **regresja**: dwa kliknięcia w odstępie 1000 ms — stan potwierdzenia
    trwa do końca DRUGIEGO okna (żywy w 1500 ms i w 2499 ms, revert w
    2500 ms), a nie do końca pierwszego;
  - kontrola: ta sama sekwencja BEZ anulowania urywa się w 1500 ms (czyli
    test faktycznie łapie stary błąd);
  - `cancelCopyRevert()` bez wiszącego timera nie rzuca;
  - realna `copyPromptText()` na ścieżce błędu (brak otwartego wpisu →
    `copyPrompt()` zwraca `false`): zdejmuje klasę, ustawia tytuł błędu, a
    stary timer po naprawie już go nie nadpisuje.
- Test regresyjny uruchomiony NAJPIERW na kodzie sprzed naprawy (przez
  `git stash` na `web/main.js`) — asercja „stale timer never reverts the
  failure title” FAIL, po naprawie PASS.

## NIE zweryfikowane (do sprawdzenia przez Kamila w żywym ComfyUI)

- Ścieżka SUKCESU realnej `copyPromptText()` end-to-end. Harness jej nie
  dosięga: `copyPrompt()` musiałoby znaleźć otwarty wpis i wyrenderować
  `renderCopyResult()`, co wymaga pełnego DOM panelu (`createPanel()`
  buduje go przez `innerHTML` + selektory atrybutowe, czego minimalny fake
  document nie odtworzy). Pokryta jest przez helpery, na których ta
  ścieżka teraz stoi, nie przez samą funkcję.
- Wizualnie: że dwa szybkie kliknięcia w ikonkę kopiowania przy prompcie
  faktycznie utrzymują podświetlenie i tooltip „Copied!” przez pełne 1,5 s
  od drugiego kliknięcia.
- Brak błędów w konsoli przeglądarki.

## Ustalenia istotne dla Chat

- `copyPrompt()` (`web/main.js:1540`) łączy zapis do schowka z dwoma
  efektami ubocznymi w panelu (`renderCopyResult()` i ustawienie
  `[data-h3cm-refs-hint]`) i zwraca `bool`. To ona, a nie
  `copyPromptText()`, jest podpięta pod duży przycisk „Copy prompt”
  (`web/main.js:553`); ikonka przy prompcie woła `copyPromptText()`
  (`web/main.js:557`).
- `cancelCopyRevert()` / `markCopied()` (`web/main.js:1486` i
  `web/main.js:1494`) są eksportowane — zgodnie z konwencją tego pliku,
  gdzie helpery dotykające DOM/localStorage (`clampLauncherPosition`,
  `writeLauncherPosition`) też są eksportowane pod harness.
- Uchwyt timera nadal jedzie na elemencie (`el._h3cmCopyRevertTimer`), a
  przebudowa panelu szczegółów wyrzuca element w całości — bez zmian
  względem PR #16.

## Otwarte pytania

- brak

## Sugestie (nie polecenia)

- Ścieżka błędu `copyToClipboardWithFeedback()` nadal zostawia klasę
  `is-copied` na elemencie, jeśli poprzednie kliknięcie się udało, a
  bieżące zawiodło (wychodzi przez `return false` bez zdjęcia klasy, a
  timer jest już anulowany, więc nikt jej nie zdejmie). Element wygląda
  wtedy na „skopiowany”, mając tytuł o błędzie. `copyPromptText()` tego
  problemu nie ma. Nie ruszone — to kod z PR #16, poza zakresem tego
  zlecenia; osobna, jednolinijkowa poprawka, jeśli uznasz za wartą.
