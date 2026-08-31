# HANDOFF

## Stan na: 2026-08-31 / branch master / commit 576b0c4

## Ostatnio zrobione
- **Faza 2 (UI) dual-resolution pairing w `web/main.js` + `web/styles.css`**
  (commit 576b0c4). Backend (5286ce9 + cf731ba) był już gotowy; ta zmiana
  uczy Cache Managera czytać `paired_fingerprint` / `paired_width` /
  `paired_height` / `is_upscale_target` z `entry.verbose.system`.
- Nowa czysta, eksportowana funkcja
  `resolvePairing(entry, entriesByFingerprint)` (`web/main.js:203`).
  Zwraca **obiekt statusu** (nie `null`), bo UI musi rozróżnić trzy różne
  "nie-parowania":
  - `{ status: "none" }` - brak `paired_fingerprint`.
  - `{ status: "valid", partner, entryIsUpscale, partnerIsUpscale }` -
    wskazanie wzajemne **oraz** obie strony mają jawne, przeciwne
    `is_upscale_target`.
  - `{ status: "orphaned" }` - jest `paired_fingerprint`, ale partnera
    nie ma w `entriesByFingerprint` **albo** partner nie wskazuje z
    powrotem.
  - `{ status: "role-unknown" }` - wskazanie wzajemne, ale
    `is_upscale_target` brak po którejś stronie (para sprzed cf731ba)
    albo obie role równe. Rola **nigdy** nie jest zgadywana z
    `paired_width * paired_height`.
- `entriesByFingerprint` budowany w `renderList()` z **pełnej** listy
  `entries` z ostatniego `/check`, nie z `filtered` - inaczej aktywne
  wyszukiwanie mogłoby fałszywie pokazać kogoś jako osieroconego.
- `renderList()`: strona upscale ważnej pary (`status === "valid" &&
  entryIsUpscale`) **nie dostaje własnego wiersza** (`continue`).
- `buildNormalRow(entry, generation, lastUsedFingerprint, pairing)` -
  nowy 4. parametr:
  - Strona bazowa ważnej pary: przycisk-plakietka
    `+ rescaled to {paired_width}×{paired_height} (X.XX MP)` (klasy
    `h3cm-badge h3cm-badge-paired h3cm-pair-toggle`). Klik toggluje
    wcięty blok `.h3cm-pair-strip` pokazujący TYLKO: skrócony fingerprint
    partnera (`.slice(0,12)…`, ten sam wzorzec co reszta pliku),
    `formatEntryMetaLine(partner.verbose.system)` (data + wymiary), i
    osobny przycisk **Delete** wołający `deleteEntry(partner.fingerprint,
    null)` - ten sam mechanizm co każdy inny Delete, tylko inny cel, bez
    kaskady. Prompt nie jest powtarzany.
  - `status === "orphaned"`: plakietka `⚠ pairing partner missing`
    (klasy `h3cm-badge h3cm-badge-orphaned`, z `title=` wyjaśniającym) +
    normalny, w pełni widoczny wiersz. `role-unknown` i `none`
    plakietki **nie** dostają.
  - Podświetlenie "ostatnio używany": zapala się teraz też na wierszu
    bazowym, gdy `lastUsedFingerprint === pairing.partner.fingerprint`
    (po dual-res przebiegu ostatni zapisany fingerprint to ukryta strona
    upscale, bo `_execute_*_once` woła się drugi raz dla upscale).
- `web/styles.css`: `.h3cm-badge-paired`, `.h3cm-badge-orphaned`,
  `.h3cm-pair-toggle`, `.h3cm-pair-toggle.is-open`, `.h3cm-pair-strip`,
  `.h3cm-pair-strip[hidden]`, `.h3cm-pair-strip-meta` - kolorystyka z
  istniejącej palety `.h3cm-badge-*` / `.h3cm-row.is-last-used`, bez
  nowego systemu.

## Ustalenia istotne dla Chat
- Odczyt pól z `entry.verbose.system` (przechodzą przez scanner/routes bez
  zmian): `paired_fingerprint` (str), `paired_width` / `paired_height`
  (int - **wymiary partnera**, nie własne), `is_upscale_target` (bool -
  **własna rola**: `false` = baza, `true` = upscale).
- `entry` z `/check` ma kształt `{ fingerprint, classification:
  "normal"|"legacy"|"inconsistent", reason?, verbose: {user, system} |
  null }`. Brak per-wpis pola rozmiaru w bajtach - dlatego blok
  rozwinięcia pokazuje wymiary pikselowe (`formatEntryMetaLine`), nie
  rozmiar plików.
- `data.last_used` z `/check` = `{ fl2va: fp|null, ref2va: fp|null }`
  (`minimaxh3_clipcache/last_used.py`).
- Osierocone parowanie po **ponownym** uruchomieniu dual-res node'a z tą
  samą bazą, inną drugą rozdzielczością: `_pair_verbose_entries`
  przepina wskaźnik bazy na nowy upscale, stary upscale zostaje z
  jednostronnym wskazaniem - `resolvePairing` klasyfikuje go jako
  `orphaned`, renderowany normalnie z plakietką.

## Otwarte pytania
- brak (KRYTERIUM_DONE ZLECENIA spełnione headless - patrz niżej).
- **Do sprawdzenia przez użytkownika w żywym ComfyUI** (CC nie może):
  realny render przycisku-plakietki i wciętego bloku, faktyczny klik
  toggle / Delete na partnerze, podświetlenie "ostatnio używany" po
  realnym dual-res przebiegu, brak błędów w konsoli, wygląd kolorów
  plakietek w ciemnej palecie.

## Weryfikacja (headless, scratchpad sesji)
- `node --check` na kopii `.mjs` `web/main.js` - OK.
- Brace-balance `web/styles.css` - OK.
- Harness Node z fake DOM (stuby `/scripts/app.js`, `/scripts/api.js`;
  napędza `openPanel → runCheck → renderList`, serializuje drzewo
  `panel.listEl`; plus unit-testy `resolvePairing`) - **ALL PASS** dla
  przypadków a-g z KRYTERIUM_DONE:
  a) para ważna, wzajemna, oba `is_upscale_target` → 1 wiersz (bazowy) z
     plakietką, upscale bez wiersza.
  b) klik plakietki → blok pokazuje dane partnera, bez promptu.
  c) `paired_fingerprint` → nieistniejący fp → wiersz normalny +
     plakietka "osierocony".
  d) stary upscale z jednostronnym wskazaniem po re-parowaniu → baza
     paruje z nowym upscale, stary upscale wiersz normalny + plakietka
     "osierocony".
  e) `is_upscale_target` brak po którejś stronie → oba wiersze normalne,
     BEZ plakietki (role-unknown, nie osierocony).
  f) wpis bez `paired_fingerprint` → identyczny kształt wiersza jak przed
     zmianą (star + label + created, nic więcej).
  g) `lastUsedFingerprint` = fp strony upscale ważnej pary → podświetlenie
     na wierszu bazowym.
- Pełny pytest (guard regresji, żaden plik .py nie ruszony):
  **303 passed** (`conda run -n comfyenv python -m pytest -q`).
- Pliki weryfikacji: scratchpad sesji `verify_dualres_ui.txt`,
  `test.mjs`, `setup-dom.mjs`, `hook.mjs`, `stub-*.mjs` (nietrackowane).

## Sugestie (nie polecenia)
- Skrajny, nieobsłużony przypadek (świadomie, zgodnie z ZLECENIEM
  "continue" bez wyjątków): jeśli strona bazowa ważnej pary wypadnie z
  `filtered` przez filtr tekstowy/tag, a strona upscale przejdzie (bo
  metadane `user` są per-fingerprint i mogą się różnić), to upscale
  dostaje `continue` i nic się nie renderuje mimo `filtered.length > 0`
  (brak komunikatu "brak wyników"). Rzadkie; do rozważenia osobno.
- `inconsistent` wpis będący stroną bazową pary: renderuje się jako
  wiersz inconsistent (bez plakietki rescale), upscale nadal ukryty.
  Nietknięte - poza zakresem.
