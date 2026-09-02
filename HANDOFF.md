# HANDOFF

## Stan na: 2026-09-02 / branch feature/dualres-drop-latent-upscale / commit 0389036

## Ostatnio zrobione

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
- Commit 2: ten plik.

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
- `.gitignore` ma niescommitowaną, niezwiązaną z tą sesją zmianę
  (dopisany `README_WORKING.md`) -- czyjaś inna, niedokończona robota,
  celowo nietknięta, NIE wchodzi w diff tej gałęzi.

## Otwarte pytania

- brak

## Sugestie (nie polecenia)

- brak
