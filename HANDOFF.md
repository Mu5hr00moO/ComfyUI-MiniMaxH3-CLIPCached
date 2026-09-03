# HANDOFF

## Stan na: 2026-09-04 / branch feat/cache-manager-ref-provenance-ui / PR (do otwarcia)

## Ostatnio zrobione

Cache Manager UI (frontend only, `web/main.js` + `web/styles.css`; zero
zmian w Pythonie): panel szczegółów pokazuje teraz (1) nazwy plików
źródłowych każdej referencji Ref2VA, złączone z `system.ref_sources`
(PR #14) po nazwie slotu z `system.references[i].slot` (PR #15), oraz
(2) skrócony fingerprint wpisu (12 znaków hex — jak w logu ComfyUI) w
pasie akcji.

### Commit 1 — kod (`web/main.js`, `web/styles.css`)

Nowe czyste, eksportowane funkcje (testowalne bez DOM):

- `refSourcesForReference(ref, refSources)` — zwraca listę tropów
  `{annotated, path?}` dla `ref.slot`. Złączenie WYŁĄCZNIE po nazwie
  slotu, nigdy po `index` (rozjeżdżają się przy luce w numeracji slotów).
  Brak `ref.slot` (sidecar sprzed PR #15), brak `refSources`, wartość
  nie-lista, wpisy bez `annotated` → `[]`.
- `detailRefCells(references, refSources)` — model renderu siatki Ref2VA:
  jedna komórka na referencję, `posLabel` liczony per typ w JS
  ("Picture 2"), `sources` złączone per komórka. Wydzielone z
  `renderDetailRefs()`, które jest teraz cienkim malowaniem DOM.
- `detailFingerprintLines(entry, pairing)` — 1 linia dla zwykłego wpisu
  (12 znaków, bez etykiety), 2 linie dla ważnej pary dual-res
  (`resolvePairing().status === "valid"`): własna + partnera, każda z
  etykietą `base` / `rescaled`. Bezpieczne przy braku fingerprintu.

Nowe funkcje DOM (nie eksportowane, malowanie):

- `buildRefSourceLines(sources)` — blok pod `posLabel`. Zawsze widoczny:
  0 tropów → wyciszony kursywą "no file source" (odróżnia referencję
  wygenerowaną VAEDecode/EmptyImage od awarii trace'u). 1 trop → nazwa
  (`annotated`). N>1 → wszystkie, jeden pod drugim. Gdy trop ma `path`:
  `title` = pełna ścieżka + klik/Enter kopiuje ją do schowka przez
  `copyToClipboardWithFeedback()`. Gdy brak `path` → sam `annotated`,
  bez tooltipa i bez kopiowania. Surowa nazwa slotu (`ref_image_5`)
  nigdy nie jest pokazywana.
- `renderDetailFingerprint(container, lines)` — `<code class="h3cm-fp">`
  per linia, opcjonalna etykieta roli.
- `copyToClipboardWithFeedback(el, text, revertTitle)` — generyczny
  helper klik-kopiuj, ta sama mechanika co `copyPromptText`
  (klasa `is-copied` + swap `title`, revert po 1.5 s).
- `entriesByFingerprintFromLastCheck()` — mapa fp→entry z ostatniego
  `/check` (to samo co buduje `renderList()`), żeby panel szczegółów
  mógł rozwiązać parowanie bez zależności od `renderList()`.

Zmiany w istniejącym kodzie:

- `renderDetailRefs()` — nowy 5. parametr `refSources`; pętla po
  `detailRefCells(...)`; każda komórka dostaje `buildRefSourceLines()`.
- `populateDetail()` — przekazuje `system.ref_sources` do
  `renderDetailRefs()`; woła `renderDetailFingerprint(...)`.
- Szablon panelu: `<div class="h3cm-detail-fingerprint"
  data-h3cm-detail-fingerprint>` w `.h3cm-detail-actions`, PO
  `[data-h3cm-detail-status]`, dosunięty w prawo przez `margin-left:auto`
  (nie przykrywa statusu, zawija się pod spód na wąskim panelu).
- `web/styles.css` — nowe reguły `.h3cm-detail-ref-sources`,
  `.h3cm-detail-ref-file{,.is-empty,.is-copyable,.is-copied}`,
  `.h3cm-detail-fingerprint`, `.h3cm-detail-fp-line`,
  `.h3cm-detail-fp-role`, dopisane obok istniejących reguł
  `.h3cm-detail-ref-*` / `.h3cm-detail-status`. Ciemna paleta i konwencja
  `h3cm-*` zachowane; monospace przez współdzieloną klasę `.h3cm-fp`.

### Commit 2 — HANDOFF.md (osobno, w tym samym PR)

## Weryfikacja

- `node --check` (kopia `.mjs`): czysto.
- `git diff --check`: czysto.
- Bilans klamer `web/styles.css`: 103/103.
- Scratchpadowy harness ESM (loader podstawia `/scripts/app.js` +
  `/scripts/api.js`, minimalne `document`/`navigator`/`URL`; harness i
  `package.json` NIE commitowane) — **21 asercji PASS**, w tym:
  - moduł importuje się bez wyjątku ze stubami ComfyUI;
  - `refSourcesForReference`: złączenie po slocie przy luce w numeracji
    (`index 1` → `ref_image_2`, nie `ref_image_1`), N>1 fan-in w całości
    z zachowanym brakiem `path`, slot obecny ale nietraced'owany → `[]`,
    referencja bez pola `slot` → `[]`, brak `refSources` → `[]`, wartość
    nie-lista → `[]`, wpisy bez `annotated` odrzucone;
  - `detailRefCells`: `posLabel` "Picture 1..3" mimo luki w slotach,
    liczniki per typ niezależne (image/audio/video), pusty `refSources`
    → każda komórka `sources: []`, `index` zachowany do fetcha
    miniatury;
  - `detailFingerprintLines`: zwykły wpis → 1 linia 12-znakowa bez roli;
    dual-res base/rescaled → 2 linie z właściwymi rolami po obu stronach;
    brak fingerprintu/parowania → bezpieczna pusta linia; `orphaned` →
    1 linia.

## NIE zweryfikowane (do sprawdzenia przez Kamila w żywym ComfyUI)

- Realny render: nazwy plików pod miniaturą w panelu Ref2VA, zawijanie
  długich nazw w `max-width:108px`, tooltip pełnej ścieżki, klik →
  schowek + "Copied!".
- Wygląd pasa akcji: pojedynczy fingerprint po prawej vs dwa (dual-res)
  jeden pod drugim; brak kolizji z `[data-h3cm-detail-status]`
  ("Saving…"/"Saved.") i zachowanie przy wąskim panelu.
- Referencja bez tropu: renderuje wyciszony kursywą tekst "no file
  source" (nie puste miejsce).
- Brak błędów w konsoli przeglądarki.

## Ustalenia istotne dla Chat

- Panel szczegółów Cache Managera łączy `system.references[i]` z
  `system.ref_sources` WYŁĄCZNIE po `ref.slot`. `index` (pozycja po
  kompakcji) celowo nie jest używany — `refSourcesForReference()` w
  `web/main.js`.
- `system.ref_sources[slot]` jest zawsze listą; N>1 to normalny przypadek
  (fan-in), renderowany w całości, każdy trop w osobnej linii.
- Trop bez `path` (odrzucony przez `get_annotated_filepath`) →
  pokazywany sam `annotated`, bez tooltipa i bez kopiowania.
- Skrócony fingerprint = `fingerprint.slice(0, 12)`, zgodny z
  `proxy.py` (`fingerprint[:12]` w `[CACHE HIT]`/`[CACHE MISS]`).
- Dla wpisu dual-res (`resolvePairing().status === "valid"`) pas akcji
  pokazuje DWA fingerprinty (własny + partnera) z etykietami
  `base`/`rescaled` z `entryIsUpscale`/`partnerIsUpscale`.
- Fingerprint pokazuje się dla OBU wariantów (FL2VA i Ref2VA) — dual-res
  istnieje dla obu.

## Otwarte pytania

- brak

## Sugestie (nie polecenia)

- Rozważyć pokazanie tej samej pary fingerprintów w wierszu listy przy
  rozwiniętym pasku "+ rescaled to WxH" (dziś strip pokazuje tylko
  `partner.fingerprint`, bez własnego i bez etykiet roli).
