# HANDOFF

## Stan na: 2026-09-04 / branch fix/copy-prompt-feedback-timer / PR #18 (open)

Gałąź wyszła z czystego `origin/master` (`23f3f72`, już po zmergowanym
PR #17). Cztery commity, PR zostaje OTWARTY:

1. `89e543c` — tura 1: kod.
2. `1c702bb` — HANDOFF.md.
3. `4554c0f` — tura 2: kod.
4. ten commit — HANDOFF.md.

Temat całości: ujednolicenie afordancji „Copied!” w Cache Managerze —
jedna implementacja mechaniki dla wszystkich kontrolek kopiujących,
odporna na nakładające się kliknięcia.

## Ostatnio zrobione

### Tura 1 (`89e543c`): `copyPromptText()` bez anulowania timera

`copyPromptText()` (ikonka kopiowania przy prompcie w panelu szczegółów)
trzymała własną kopię mechaniki „Copied!”: klasa `is-copied`, podmiana
`title`, `setTimeout` na revert po 1,5 s — bez anulowania timera z
poprzedniego kliknięcia. Ten sam defekt, który dla
`copyToClipboardWithFeedback()` naprawił PR #16.

PEŁNE reużycie `copyToClipboardWithFeedback()` nie było możliwe bez
naginania jej: ta funkcja sama wykonuje zapis do schowka, a
`copyPromptText()` nie ma tekstu promptu — deleguje do `copyPrompt()`
(`web/main.js:1563`), która robi zapis RAZEM z dwoma efektami ubocznymi,
których ten przycisk nie jest właścicielem (`renderCopyResult()` i
ustawienie `[data-h3cm-refs-hint]`). Wyjściem byłoby albo zduplikowanie w
`copyPromptText()` wyszukiwania wpisu i obu efektów ubocznych (duplikacja
większa niż usuwana), albo dorobienie wspólnej funkcji parametru-callbacka
pod jednego wołającego. Zamiast tego wyciągnięta została sama afordancja,
do helperów używanych przez obie funkcje.

### Tura 2 (`4554c0f`), poprawka A: `is-copied` na ścieżce błędu

`copyToClipboardWithFeedback()` nie zdejmowała klasy `is-copied` przy
nieudanym kopiowaniu. Gdy poprzednie kliknięcie się udało,
`cancelCopyRevert()` zabijał timer, który by ją posprzątał — element
zostawał podświetlony jako „skopiowany”, mając tytuł o błędzie.
`copyPromptText()` tego nie miała, bo robiła `remove` jawnie. To była
sugestia z raportu po turze 1, teraz zrealizowana.

### Tura 2, poprawka B: osierocone timery przy nakładających się wywołaniach

Zgłoszenie review bota — zweryfikowane, realne. Anulowanie PRZED `await`
nie wystarcza: dwa kliknięcia mogą oba wyczyścić jeszcze pusty uchwyt,
zanim którekolwiek cokolwiek zaplanuje (próba kopiowania jest
asynchroniczna, więc pierwsze kliknięcie może nie dojść do swojego timera,
zanim drugie go poszuka). Oba planują potem revert, element pamięta tylko
późniejszy uchwyt, a osierocony wcześniejszy timer odpala w środku okna
tego późniejszego.

`markCopied()` (`web/main.js:1508`) i nowa `markCopyFailed()`
(`web/main.js:1522`) anulują PONOWNIE w momencie nakładania wyniku — to
ostatnia chwila, w której osierocony timer da się jeszcze złapać.
Anulowanie przed `await` ZOSTAJE: chroni przed czym innym, timerem z
wcześniejszego, już rozstrzygniętego kliknięcia, odpalającym w trakcie
długiego oczekiwania na zgodę użytkownika. Anulowanie jest idempotentne,
więc oba mogą stać obok siebie.

Ścieżka błędu została wyciągnięta do `markCopyFailed()`, żeby obaj
wołający nie mogli się znowu rozjechać tak, jak rozjechali się przy
poprawce A — jedna implementacja zamiast dwóch kopii trzech linijek.

### Świadomie POZA zakresem (odnotowane komentarzem, nie zaimplementowane)

Gdy nałożone wywołania rozstrzygną się w odwrotnej kolejności, a
wcześniejsze zawiedzie, element pokaże „Copy failed” mimo udanego
późniejszego kopiowania. Numerowanie generacji pod ten przypadek byłoby
przerostem formy nad treścią przy tooltipie żyjącym 1,5 s. Zapisane w
komentarzu nad stałymi (`web/main.js:1474`).

### Inne miejsca z tym wzorcem (punkt 3 zlecenia z tury 1)

W `web/main.js` są dokładnie DWA `setTimeout`. Drugi (`web/main.js:1754`,
w `endDrag()` pływającego launchera) to inny wzorzec i NIE ma tego błędu:
zeruje flagę `suppressClick` z opóźnieniem 0 ms, nie revertuje żadnego
tytułu ani klasy, a flaga jest niezależnie zerowana przy każdym
`pointerdown` (`web/main.js:1721`) i w samym handlerze `click`. Nakładające
się timery nie mogą tam zostawić złego stanu. Zostawiony bez zmian. Poza
tym w pliku nie ma `setInterval` ani `requestAnimationFrame`.

## Weryfikacja (stan bieżący)

- `node --check` na kopii `.mjs`: czysto. `git diff --check`: czysto.
- Pełny pytest: **446 passed** (bez zmian — ta gałąź nie rusza Pythona).
- Harness ESM w scratchpadzie, na REALNYM `web/main.js` (loader podstawia
  `/scripts/app.js` i `/scripts/api.js` i dopisuje `export` dla prywatnych
  `copyPromptText` i `copyToClipboardWithFeedback`, więc testowany jest
  bieżący kod, nie kopia; wirtualny zegar zamiast prawdziwych timerów;
  sterowalny stub `navigator.clipboard.writeText` zwracający obietnice
  rozstrzygane ręcznie, dzięki czemu dwa wywołania mogą być „w locie”
  naraz; harness i loader NIE commitowane) — **33 asercje PASS** (po
  turze 1 było 18).
- Zgodnie ze stałą zasadą projektu asercje regresyjne były w OBU turach
  uruchamiane NAJPIERW na kodzie sprzed naprawy (przez `git stash` na
  `web/main.js`) i tam FAILowały:
  - tura 1 — `5c` „stale timer never reverts the failure title”;
  - tura 2 — **6 asercji**: `3a` (przepisany dawny test kontrolny:
    `markCopied()` bez jawnego anulowania przez wołającego ma utrzymać
    drugie okno), `6b`/`6c` (dwa nałożone wywołania, `writeText`
    rozstrzygane ręcznie w t=0 i t=500 — potwierdzenie żyje w 1500 ms i
    1999 ms, revert dopiero w 2000 ms, czyli osierocony timer został
    złapany), `7a` (`is-copied` zdjęte na ścieżce błędu po udanym
    poprzednim kliknięciu), `8a`/`8c` (nałożone wywołania, gdzie
    późniejsze zawodzi po tym, jak wcześniejsze zaplanowało revert —
    klasa zdjęta, tytuł błędu nie jest wycierany o 1500 ms; dokładnie ten
    przypadek wymaga anulowania PO `await`).

## NIE zweryfikowane (do sprawdzenia przez Kamila w żywym ComfyUI)

- Ścieżka SUKCESU realnej `copyPromptText()` end-to-end. Harness jej nie
  dosięga: `copyPrompt()` musiałoby znaleźć otwarty wpis i wyrenderować
  `renderCopyResult()`, co wymaga pełnego DOM panelu (`createPanel()`
  buduje go przez `innerHTML` + selektory atrybutowe, czego minimalny fake
  document nie odtworzy). Pokryta jest przez helpery, na których ta
  ścieżka stoi. `copyToClipboardWithFeedback()` jest natomiast pokryta
  bezpośrednio, na obu ścieżkach.
- Wizualnie: że dwa szybkie kliknięcia w ikonkę kopiowania przy prompcie
  faktycznie utrzymują podświetlenie i tooltip „Copied!” przez pełne 1,5 s
  od drugiego kliknięcia.
- Realna odmowa dostępu do schowka w przeglądarce (w harnessie ścieżka
  błędu jest symulowana odrzuconą obietnicą).
- Brak błędów w konsoli przeglądarki.

## Ustalenia istotne dla Chat

- Afordancja kopiowania stoi teraz na trzech eksportowanych helperach:
  `cancelCopyRevert()` (`web/main.js:1501`), `markCopied()`
  (`web/main.js:1508`), `markCopyFailed()` (`web/main.js:1522`). Oba
  helpery nakładające wynik anulują na wejściu — to NIE jest redundancja
  wobec anulowania przed `await`; każde chroni przed innym przypadkiem
  (opis w komentarzu nad stałymi, `web/main.js:1474`).
- `copyPrompt()` (`web/main.js:1563`) łączy zapis do schowka z dwoma
  efektami ubocznymi w panelu (`renderCopyResult()` i ustawienie
  `[data-h3cm-refs-hint]`) i zwraca `bool`. To ona, a nie
  `copyPromptText()`, jest podpięta pod duży przycisk „Copy prompt”
  (`web/main.js:553`); ikonka przy prompcie woła `copyPromptText()`
  (`web/main.js:557`). To jest powód, dla którego `copyPromptText()` nie
  może po prostu wołać `copyToClipboardWithFeedback()`.
- Helpery są eksportowane zgodnie z konwencją tego pliku, gdzie helpery
  dotykające DOM/localStorage (`clampLauncherPosition`,
  `writeLauncherPosition`) też są eksportowane pod harness.
- Uchwyt timera nadal jedzie na elemencie (`el._h3cmCopyRevertTimer`), a
  przebudowa panelu szczegółów wyrzuca element w całości — bez zmian
  względem PR #16.

## Otwarte pytania

- brak

## Sugestie (nie polecenia)

- brak
