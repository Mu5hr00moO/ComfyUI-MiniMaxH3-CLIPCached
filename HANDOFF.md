# HANDOFF

## Stan na: 2026-08-31 / branch master / commit b22774f

## Ostatnio zrobione
- Dodany trzeci node w repo: **MiniMaxH3CLIPName** ("MiniMax H3 CLIP Name").
  Jeden widget `clip_name` (ten sam dropdown co FL2VA/Ref2VA) -> jeden
  output typu COMBO, do podpięcia (po "Convert widget to Input") pod
  `clip_name` dowolnej liczby MiniMaxH3CLIPCachedFL2VA /
  MiniMaxH3CLIPCachedRef2VA naraz. Jedno miejsce zmiany enkodera zamiast N.
- `nodes.py`: nowy `_ComboType(str)` (output type dopasowujący się do
  dowolnej listy), nowy `_clip_name_input_spec(tooltip=None)` (wspólny
  wpis INPUT_TYPES dla `clip_name`), nowa klasa `MiniMaxH3CLIPName`.
  FL2VA i Ref2VA używają teraz `_clip_name_input_spec()` zamiast trzeciej
  kopii wpisu inline - ich tooltip bez zmian (domyślny w helperze jest
  bajt-w-bajt tym samym tekstem co wcześniej inline).
- `__init__.py`: trzy nowe wpisy (import klasy + NODE_CLASS_MAPPINGS +
  NODE_DISPLAY_NAME_MAPPINGS).
- `tests/test_clip_name_node.py`: nowy plik, 10 testów.
- `tests/test_node.py` `test_f`: zbiór oczekiwanych kluczy
  NODE_CLASS_MAPPINGS rozszerzony o trzeci node (wymuszone przez
  KRYTERIUM_DONE "0 FAIL" - asercja `set(...) == {2 klucze}` inaczej pada).
- Weryfikacja: `python -m py_compile nodes.py __init__.py` OK;
  pełny pytest **271 passed, 0 skipped, 0 failed** (było 261, +10 nowych).

## Ustalenia istotne dla Chat

### Wariant COMBO - potwierdzony bezpośrednio przeciw prawdziwej walidacji ComfyUI
- `comfy_execution.validation.validate_node_input` **da się** zaimportować
  w zwykłym środowisku pytest bez żadnego dodatkowego stanu procesu
  (bez `init_extra_nodes`, bez serwera). Jedyna zależność tego modułu to
  `from comfy_api.latest import IO`, a `conftest.py` już wrzuca korzeń
  ComfyUI na `sys.path`. Test `test_comfy_execution_validation_is_importable`
  to formalnie potwierdza.
- Przeciw prawdziwemu `validate_node_input`
  (`comfy_execution/validation.py`):
  - `validate_node_input(_ComboType("COMBO"), <lista opcji clip_name>)` -> `True`
  - `validate_node_input(_ComboType("COMBO"), "COMBO")` -> `True`
  - `validate_node_input(_ComboType("COMBO"), "STRING")` -> `False`
  - `validate_node_input(_ComboType("COMBO"), "CONDITIONING")` -> `False`
  - `validate_node_input("COMBO", <lista opcji clip_name>)` -> `False`
    (goły string "COMBO" NIE przechodzi - potwierdza, że `_ComboType`
    faktycznie coś rozwiązuje, nie jest kosmetyką).
  Ścieżka w źródle: pierwsza linia `if not received_type != input_type`
  honoruje override `__ne__` na typie otrzymanym; `_ComboType.__ne__`
  zwraca `False` dla dowolnej `list`.
- W `validation.py` jest komentarz `# if we ever want to break them on
  purpose, this can be removed` przy specjalnym przypadku list/COMBO -
  zachowanie nieudokumentowane, ale obecnie stabilne; do rewizji przy
  większym bumpie ComfyUI. Odnotowane w docstringu `_ComboType`.

### Kwestia środowiskowa: `import nodes` pod pytest
- Pod prawdziwym ComfyUI `import nodes` wewnątrz naszego `nodes.py` łapie
  globalny moduł ComfyUI (już w `sys.modules`). Pod pytest łapie **nasz
  własny** `nodes.py` (korzeń repo jest przed korzeniem ComfyUI na
  `sys.path`), który nie ma `MAX_RESOLUTION`.
- Skutek: `MiniMaxH3CLIPCachedFL2VA.INPUT_TYPES()` /
  `...Ref2VA.INPUT_TYPES()` rzuca `AttributeError: module 'nodes' has no
  attribute 'MAX_RESOLUTION'` pod pytest. Dotąd nie ujawniało się, bo
  ŻADEN istniejący test nie wołał `INPUT_TYPES()`.
- Rozwiązane w `test_clip_name_node.py` fixture
  `node_module_with_real_comfy_nodes`: ładuje prawdziwy
  `<ComfyUI>/nodes.py` (~0,2 s, bez serwera) do `sys.modules["nodes"]`
  na czas testu i przywraca poprzednie wiązanie po. To realne obejście
  (ładuje prawdziwy moduł), nie mock.
- `MiniMaxH3CLIPName.INPUT_TYPES()` sam NIE wymaga tego fixture - ma tylko
  widget `clip_name`, zero `nodes.MAX_RESOLUTION`.

## Otwarte pytania
- brak (KRYTERIUM_DONE spełnione: py_compile OK, nowy plik testowy zielony
  w tym test przeciw prawdziwej walidacji, pełny pakiet 271/0/0).
- Do sprawdzenia przez użytkownika w żywym UI (CC nie może): czy output
  MiniMaxH3CLIPName faktycznie podłącza się pod `clip_name` po "Convert
  widget to Input" na FL2VA/Ref2VA i przechodzi Queue bez type_mismatch,
  oraz czy jeden output da się rozgałęzić na wiele node'ów naraz.
