# HANDOFF

## Stan na: 2026-08-30 / branch master (po poprawce kolejności zwalniania proxy)

## Ostatnio zrobione
- Poprawka kolejności zwalniania referencji w
  `_release_real_clip_safety_net` (`nodes.py`). Regresja pochodziła z
  commita 500d668: po wydzieleniu bloku `finally` do wspólnej funkcji
  `del proxy` w środku helpera kasował już tylko *parametr* funkcji, a
  `execute()` wciąż trzymał tę samą referencję na swoim stosie (czeka w
  `finally` na powrót helpera). W efekcie `gc.collect()` i
  `comfy.model_management.soft_empty_cache()` odpalały się, gdy proxy — i
  osiągalny przez nie realny ~27 GB encoder — wciąż był żywy. Dotyczyło
  KAŻDEGO MISS/refresh, nie tylko ścieżki awaryjnej.
- `_release_real_clip_safety_net(proxy)` robi teraz tylko targeted unload
  + `logger.warning` (jak dawniej) i zwraca `bool`: `True` gdy
  `proxy.did_load_real_clip` (realny load — caller ma posprzątać),
  `False` przy HIT.
- Oba `execute()` (FL2VA i Ref2VA) w `finally`:
  ```python
  if _release_real_clip_safety_net(proxy):
      del proxy
      gc.collect()
      comfy.model_management.soft_empty_cache()
  ```
  `del proxy` wykonuje się teraz we właściwej ramce (`execute()`), gdzie
  jest ostatnią referencją — obiekt ginie natychmiast przez refcounting,
  PRZED `gc.collect()`/`soft_empty_cache()`, dokładnie jak przed 500d668.
  Świadoma, minimalna duplikacja 3 linii w obu miejscach — nie cofka
  całego refaktoru.
- Docstring `_release_real_clip_safety_net` rozszerzony o wyjaśnienie
  WHY `del`/`gc`/`soft_empty_cache` nie mogą żyć w tej funkcji (parametr
  = dodatkowa żywa referencja u wywołującego do jego powrotu).

## Ustalenia istotne dla Chat
- `git diff nodes.py` ogranicza się do `_release_real_clip_safety_net`
  (`nodes.py:276`) i dwóch bloków `finally` w `execute()`
  (`nodes.py:392`, `nodes.py:575`) — 46 wstawień / 14 usunięć.
- Nowy test regresyjny w OBU plikach:
  `tests/test_node.py::test_k2_proxy_unreachable_before_soft_empty_cache_on_miss`
  i `tests/test_node_ref2va.py::test_j2_proxy_unreachable_before_soft_empty_cache_on_miss`.
  Monkeypatch `comfy.model_management.soft_empty_cache`; wewnątrz sprawdza
  przez `weakref.ref` na obiekcie proxy, że jest już nieosiągalny w
  momencie wywołania `soft_empty_cache()`. Weakref (nie strong ref), żeby
  sam test nie trzymał obiektu.
- Potwierdzone lokalnie: oba nowe testy FAILują na kodzie sprzed
  poprawki (`git stash push nodes.py`, `assert True is False` — proxy
  wciąż żywy) i PASSują po niej.
- `pytest tests/test_node.py tests/test_node_ref2va.py -v` — 67 passed
  (38 FL2VA + 29 Ref2VA), 0 SKIP, 0 FAIL. Pełny pakiet: 262 passed.
- Pełny output: scratchpad `release_order_fix_report.txt`.
- Poprzedni HANDOFF (dla 500d668) klasyfikował tę różnicę jako
  „teoretyczną, bez konsekwencji dla VRAM/RAM" — to była błędna ocena:
  realny encoder jest osiągalny przez proxy przez cały czas trwania
  `gc.collect()`/`soft_empty_cache()`, więc reclaim ich nie obejmuje.

## Otwarte pytania
- brak
