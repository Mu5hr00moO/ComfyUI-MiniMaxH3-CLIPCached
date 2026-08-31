# HANDOFF

## Stan na: 2026-08-31 / branch master

## Ostatnio zrobione
- Rozpoznanie mechanizmu typu COMBO w lokalnym ComfyUI 0.34.2 +
  comfyui-frontend-package 1.49.6, pod przyszły samodzielny node
  "MiniMax H3 CLIP Name" (jeden output podpinany do `clip_name` na N
  węzłach FL2VA/Ref2VA po wystawieniu widgetu jako input). NIE napisano
  żadnego kodu node'a — samo rozpoznanie.
- Empirycznie przetestowano `comfy_execution.validation.validate_node_input`
  na zainstalowanym kodzie (skrypt jednorazowy, nie commitowany):
  - `RETURN_TYPES=("COMBO",)` (literalny string) → `validate_node_input`
    zwraca **False** przy targecie będącym listą opcji (stary format
    combo). Link narysuje się w UI, ale Queue padnie z
    `return_type_mismatch`.
  - `RETURN_TYPES=(folder_paths.get_filename_list("text_encoders"),)`
    (lista opcji jako "typ") → **True**, o ile lista jest treściowo równa
    liście `clip_name` targetu (obie strony wołają ten sam
    `folder_paths.get_filename_list("text_encoders")`).
  - `RETURN_TYPES=("*",)` → **True** (ale łączy się z każdym typem).
  - `str`-subclass "COMBO" z nadpisanym `__ne__`/`__eq__` (match dowolnej
    listy) → **True** dla listy, **False** dla `STRING`, **True** dla
    `COMBO`; do frontendu to zwykły string "COMBO".

## Ustalenia istotne dla Chat
- Wersje: ComfyUI 0.34.2 (`git describe` = v0.34.2, HEAD 169fcf35);
  frontend przypięty w `requirements.txt:1` =
  `comfyui-frontend-package==1.49.6` (zgodne z zainstalowanym pakietem).
- `clip_name` w obu węzłach cached jest starym formatem combo:
  `(folder_paths.get_filename_list("text_encoders"), {...})` —
  `nodes.py:333` (FL2VA) i `nodes.py:512` (Ref2VA). Żaden z węzłów nie ma
  `VALIDATE_INPUTS` (tylko `IS_CHANGED`, `nodes.py:363` / `:534`).
- Backend NIE konwertuje starego formatu combo na string "COMBO":
  `comfy_execution/graph.py:99-104` — konwersja jawnie zakomentowana.
  `get_input_info` zwraca surową listę opcji jako `input_type`.
- `validate_inputs` w `execution.py:934-951`: dla linku czyta
  `RETURN_TYPES[slot]` źródła i woła `validate_node_input(received_type,
  input_type, strict=False)`. Brak `VALIDATE_INPUTS` z parametrem
  `input_types` na targecie → sprawdzenie typu FAKTYCZNIE się wykonuje.
- `comfy_execution/validation.py`: `"COMBO"` (str) vs lista → linie 38-43
  → **False**. Lista vs identyczna lista → linia 24 (`not (a != b)`) →
  **True**. `"*"` → linia 28 → **True**. Komentarz w liniach 36-39
  ("custom nodes that output lists of options as the type ... if we ever
  want to break them on purpose, this can be removed") — ten wzorzec jest
  tolerowany, nie jest kontraktem.
- `server.py:759`: `info['output'] = obj_class.RETURN_TYPES` przekazywane
  1:1 do `/object_info` (element-lista → JSON `[[...]]`).
- Frontend (odczyt z source-map 1.49.6, NIE test w żywym UI):
  - `transformNodeDefV1ToV2` (`src/schemas/nodeDef/migration.ts:56,62-64`):
    output typu `Array` → `type:'COMBO'` + `options`. Input combo →
    `type:'COMBO'` (`migration.ts:116-123`).
  - `litegraphService.addOutputs` (`:341-357`) i `addInputSocket`
    (`:218-231`) używają `spec.type` z V2 → gniazdo dostaje string
    `"COMBO"`.
  - `LiteGraph.isValidConnection` (`LiteGraphGlobal.ts:688`): `'COMBO'`
    vs `'COMBO'` → true; `'*'` traktowane jak `0` → true z każdym.
  - "Convert widget to input" jest **deprecated / no-op** w 1.49.6
    (`src/extensions/core/widgetInputs.ts:430-438,518-523` — tylko
    `console.warn`). Widget i gniazdo współistnieją; gniazdo combo jest
    zawsze obecne, typu `"COMBO"`.
  - Promocja przez subgraf (`src/core/graph/subgraph/promotionUtils.ts:
    287-290`): `subgraph.addInput(name, String(sourceSlot.type ?? ...))`
    → też `"COMBO"`. Ten sam string co "convert widget to input".
  - Stockowy frontendowy `PrimitiveNode`
    (`src/extensions/core/widgetInputs.ts:31-403`) jest `isVirtualNode`
    — output startuje jako `'*'`, zwężany do `'COMBO'` przy pierwszym
    połączeniu; wartość wpisywana wprost do widgetu targetu przed
    wysłaniem promptu → backend go nigdy nie waliduje. Realny node
    backendowy nie ma tej furtki.

## Rekomendacja (jako sugestia dla Chat, nie polecenie)
- Docelowy `RETURN_TYPES` nowego node'a:
  `RETURN_TYPES = (folder_paths.get_filename_list("text_encoders"),)`,
  `RETURN_NAMES = ("clip_name",)`. Bez dodatkowych flag/atrybutów, bez
  zmian w `nodes.py` FL2VA/Ref2VA.
- Wariant odporniejszy na rozjazd treści listy: `str`-subclass "COMBO"
  z `__ne__`/`__eq__` matchującym dowolną listę (udokumentowany trik
  AnyType, zawężony do combo). Do frontendu wygląda jak string "COMBO".
- `("COMBO",)` literalnie — NIE (przechodzi w UI, pada na Queue).

## Otwarte pytania
- Świadoma kruchość wariantu z gołą listą: `RETURN_TYPES` jest liczone
  RAZ przy imporcie modułu; dodanie/usunięcie pliku w `models/text_encoders`
  w trakcie sesji rozjeżdża listę z `INPUT_TYPES()` targetu (liczone
  świeżo) do czasu restartu ComfyUI. Wariant `str`-subclass to omija.
- Frontendowa część rozpoznania to ODCZYT source-map, nie test w żywym
  UI. Do domknięcia: jeden fizyczny test (nowy node → drop output na
  `clip_name` 2× FL2VA → Queue).
