# HANDOFF

## Stan na: 2026-09-02 / branch feature/dualres-drop-latent-upscale / commit 673cc7e

## Ostatnio zrobione

Odpowiedź na automatyczny review greptile-apps na PR #3: wzmocnienie
`test_dual_runs_both_resolutions_with_shared_inputs` w obu plikach testów
dual. Wcześniej test sprawdzał tylko `cond2 is not None`, a wspólna
`FakeRealClip.encode_from_tokens_scheduled()` zwraca jedną stałą wartość
dla każdego wejścia, więc zamiana miejscami dwóch wyjść CONDITIONING w
`return` któregokolwiek node'a DualRes przeszłaby niezauważona. Teraz ten
jeden test w każdym pliku dostaje lokalną podklasę `FakeRealClip` z
resolution-aware encode (zwracany conditioning niesie `(width, height)`
odczytane z kształtu tensora obrazu / referencji w tokens); asercje
wymagają, żeby `cond` i `cond2` były rozróżnialne i każdy odpowiadał
swojej rozdzielczości. Wspólna klasa `FakeRealClip` (używana przez
pozostałe testy) nietknięta.

Weryfikacja wzorcem regresyjnym: po tymczasowej zamianie
`cond`/`cond_upscale` w `return` obu klas DualRes w `nodes.py`
(`nodes.py:715`, `nodes.py:1095`) oba testy FAILują; po przywróceniu --
PASS. Pełny pytest: 398 passed / 0 skip / 0 fail. `git diff` ograniczony
do dwóch plików testowych, wyłącznie w obrębie tego jednego testu w
każdym. `nodes.py` bez zmian.

- Commit 3 (673cc7e): `tests/test_node_fl2va_dual.py`,
  `tests/test_node_ref2va_dual.py` -- wzmocnienie testu return-slot.
- Commit 4: ten plik.

---

Usunięcie wyjścia `latent_upscale` z obu node'ów Dual Resolution
(`MiniMaxH3CLIPCachedFL2VADualRes`, `MiniMaxH3CLIPCachedRef2VADualRes`).
Oba zwracają teraz trzy wyjścia -- `(positive, latent, positive_upscale)`
-- zamiast czterech. Przebieg upscale nadal wykonuje pełny cache'owany
encode, ale jego AV latent był zawsze świeżym pustym tensorem w rozmiarze
upscale, bez niczego przeniesionego z pierwszego przebiegu, więc realny
workflow upscale i tak go nie używał (upscaluje odszumiony latent z
pierwszego przebiegu zewnętrznym node'em). Zostaje tylko conditioning
rozdzielczości upscale.

Zmiana zbudowana na nowej gałęzi `feature/dualres-drop-latent-upscale`
odbitej od `origin/master` (aa82610, po merge PR #2). Od teraz wszystkie
zmiany idą przez PR, nie bezpośrednio na master.

- Commit 1 (0389036): `nodes.py`, `tests/test_node_fl2va_dual.py`,
  `tests/test_node_ref2va_dual.py`, `README.md` -- RETURN_TYPES /
  RETURN_NAMES, docstringi klas, tooltipy `generate_upscale_cond`, linia
  logu `[UPSCALE COND SKIPPED]`, sekcja README, testy dual.
- Commit 2 (efe8ed6): ten plik.

Walidacja na gałęzi: `python -m py_compile nodes.py` OK; pełny pytest
398 passed / 4 warnings (świeży przebieg); `grep -rn "latent_upscale"`
po `nodes.py` + obu plikach testów dual + `README.md` -- pusty wynik.

## Ustalenia istotne dla Chat

- Oba node'y Dual Resolution:
  `RETURN_TYPES = ("CONDITIONING", "LATENT", "CONDITIONING")`,
  `RETURN_NAMES = ("positive", "latent", "positive_upscale")`
  (`nodes.py:673-674`, `nodes.py:1038-1039` na tej gałęzi).
- `generate_upscale_cond` BOOLEAN default `True` bez zmian
  (`nodes.py:656`, `nodes.py:999`). Gdy `False`: `positive_upscale`
  zwraca `None`, `_pair_verbose_entries()` pominięte, encoder ładowany
  co najwyżej raz. Linia logu: `[UPSCALE COND SKIPPED] <fp>: ...
  positive_upscale not computed`.
- Gdy `generate_upscale_cond=True`: przebieg upscale nadal wywołuje
  `_execute_fl2va_once()` / `_execute_ref2va_once()` w całości; tylko
  zwracany AV latent jest odrzucany (`cond_upscale, _, fp2 = ...`).
  Fingerprint / HIT-MISS / verbose pairing bez zmian.
- Pełny pytest: 398 passed (`testpaths = tests`).
- `test_dual_runs_both_resolutions_with_shared_inputs` (oba pliki dual)
  po wzmocnieniu realnie chroni kolejność krotki `return` obu node'ów
  DualRes: `cond` musi nieść `(1344, 768)`, `cond2` musi nieść
  `(1920, 1088)`, i muszą być rozróżnialne. Marker rozdzielczości jest
  wstrzykiwany tylko lokalnie w tym teście (podklasa `FakeRealClip`),
  reszta testów dalej używa stałej wartości ze wspólnej klasy.
- `.gitignore` ma niescommitowaną, niezwiązaną z tą sesją zmianę
  (dopisany `README_WORKING.md`) -- czyjaś inna, niedokończona robota,
  celowo nietknięta, NIE wchodzi w diff tej gałęzi.

## Otwarte pytania

- brak

## Sugestie (nie polecenia)

- brak
