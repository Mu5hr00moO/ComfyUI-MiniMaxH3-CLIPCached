# HANDOFF

## Stan na: 2026-09-04 / branch `docs/provenance-scope-correction` / kod w `1f595a5`

Gałąź wyszła z czystego `origin/master` (`d1863fc`, już po zmergowanym
PR #18). Dwa commity: `1f595a5` — dokumentacja, ten commit — HANDOFF.md.

Temat: usunięcie rozjazdu między dokumentacją a kodem w sprawie śledzenia
nazw plików źródłowych referencji. Wyłącznie dokumentacja — `git diff
--stat` względem `origin/master` pokazuje `CLAUDE.md` i `TODO.md`, żadnego
`.py`, `.js` ani `.css`. Testów nie ruszano i nie uruchamiano (nie ma
czego regresować: zero zmian wykonywalnych).

## Ostatnio zrobione

### Co było niezgodne

Dwa miejsca opisywały całą sprawę nazw plików jako odłożoną:

- `TODO.md`, wpis „Reference source filenames -- dedicated loader
  wrappers”,
- `CLAUDE.md`, sekcja „Rozważone i ODŁOŻONE (nie odrzucone na stałe):
  nazwy plików referencji”.

Oba twierdziły, że introspekcja grafu byłaby krucha i wychodziłaby poza
kontrakt węzła, a jedynym poprawnym rozwiązaniem jest osobna rodzina
wrapperów `LoadImage`/`LoadVideo`/`LoadAudio`. Tymczasem
`minimaxh3_clipcache/provenance.py` istnieje i robi dokładnie tę
introspekcję dla slotów `ref_*` w Ref2VA, a Cache Manager pokazuje wynik.

### `TODO.md`

Wpis zawężony do tego, co realnie zostało otwarte: `first_frame` /
`last_frame` w FL2VA. Nowy nagłówek: „Reference source filenames --
FL2VA's `first_frame` / `last_frame`”. Treść mówi, czego brakuje
technicznie (węzły FL2VA nie deklarują bloku `hidden`, więc nie dostają
`PROMPT`/`UNIQUE_ID`; sam przejazd po grafie jest generyczny poza filtrem
`_REF_INPUT_PREFIXES`) i że pierwotne zastrzeżenie o kontrakcie węzła
przestało być przeszkodą. Wpis zachowany, nie skasowany — po zawężeniu
nadal opisuje konkretną, wykonalną pozycję backlogu.

### `CLAUDE.md`

Historyczne akapity zostawione nietknięte (zapis tego, co wtedy
rozważano). Dopisane cztery akapity: że wariant Ref2VA został
zrealizowany inną drogą, na czym polegało zawężenie zakresu, które
zdjęło wagę pierwotnemu zastrzeżeniu, wskazanie docstringu
`provenance.py` jako autorytatywnego opisu granic (bez duplikowania go)
oraz zdanie o tym, co pozostaje odłożone. Nagłówek sekcji zmieniony z
„Rozważone i ODŁOŻONE (nie odrzucone na stałe): nazwy plików referencji”
na „Nazwy plików referencji: Ref2VA zrealizowane, FL2VA nadal odłożone”,
bo stary nagłówek sam w sobie był tym rozjazdem. Języki plików
zachowane: `TODO.md` po angielsku, `CLAUDE.md` po polsku.

## Ustalenia istotne dla Chat

- `collect_ref_sources(prompt, unique_id)` chodzi BFS po grafie wstecz od
  każdego slotu `ref_*`; literał liczy się tylko na węźle-liściu (żadne
  wejście nie jest linkiem) — minimaxh3_clipcache/provenance.py:167-210,
  246-296.
- Wywoływane wyłącznie ze ścieżki Ref2VA, przez `_sync_ref_sources()` —
  nodes.py:283; blok `hidden` mają tylko węzły Ref2VA —
  nodes.py:1121 i nodes.py:1255 (`_ref2va_hidden_input_spec()`,
  nodes.py:936).
- `MiniMaxH3CLIPCachedFL2VA.INPUT_TYPES()` nie ma bloku `hidden` —
  nodes.py:706-728. To jest jedyny realny brak po stronie FL2VA.
- UI konsumuje `system.ref_sources` i pokazuje nazwy plików pod
  referencjami w panelu szczegółów — web/main.js:1258 (przekazanie do
  `renderDetailRefs`, web/main.js:1173), web/main.js:1107
  (`detailRefCells`).
- Nieaktualne twierdzenia tej samej rodziny zostały w `docs/` i NIE są
  ruszane w tej gałęzi (patrz „Otwarte pytania”).

## Otwarte pytania

- `docs/TESTING_AND_LIMITATIONS.md:105-117`, sekcja „Original reference
  filenames are not tracked”, twierdzi wprost, że węzły nie znają nazwy
  pliku źródłowego i że odzyskanie jej wymagałoby dedykowanych wrapperów,
  a introspekcji grafu świadomie unikano. Dla Ref2VA to jest dziś
  nieprawda. Poprawiać w osobnej gałęzi?
- `docs/CACHE_MANAGER.md:73-74` mówi, że panel „does not preserve the
  original source filenames”. Dla Ref2VA nieaktualne (dla FL2VA nadal
  prawdziwe). Ta sama decyzja co wyżej.
- `CLAUDE.md` w historycznym akapicie odsyła do `MANAGER_TODO_ref2video.md
  punkt 10`; tego pliku nie ma już w repo. Zostawione bez zmian jako
  zapis historyczny — czy warto to odesłanie usunąć/zastąpić?

## Sugestie (nie polecenia)

- Jeżeli obie sekcje w `docs/` mają być poprawione, warto to zrobić jedną
  gałęzią razem, bo mówią o tym samym i różnią się tylko szczegółowością.
