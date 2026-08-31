# HANDOFF

## Stan na: 2026-08-31 / branch master / commit 4c5fbf7

## Ostatnio zrobione
- Dodane dwa nowe, samodzielne node'y "dual resolution":
  **MiniMaxH3CLIPCachedFL2VADualRes** ("MiniMax H3 CLIP-Cached FL2VA
  (Dual Resolution)") i **MiniMaxH3CLIPCachedRef2VADualRes** ("MiniMax H3
  CLIP-Cached Ref2VA (Dual Resolution)"). Każdy liczy CONDITIONING+LATENT
  dla DWÓCH rozdzielczości (width/height oraz nowe width2/height2) z
  jednego zestawu współdzielonych inputów - eliminuje ryzyko rozjazdu tych
  wartości między dwiema osobnymi instancjami node'a w jednym workflow.
  RETURN_TYPES = (CONDITIONING, LATENT, CONDITIONING, LATENT),
  RETURN_NAMES = (positive, latent, positive_2, latent_2).
- `nodes.py`: ciało `MiniMaxH3CLIPCachedFL2VA.execute()` (od budowy proxy
  w dół) wydzielone do modułowej funkcji `_execute_fl2va_once(clip_name,
  vae, prompt, width, height, length, first_frame, last_frame,
  cache_mode) -> (cond, latent)`. Analogicznie
  `MiniMaxH3CLIPCachedRef2VA.execute()` -> `_execute_ref2va_once(clip_name,
  vae, audio_vae, prompt, width, height, length, ref_image_size,
  ref_images, ref_videos, ref_video_audios, ref_audios, cache_mode) ->
  (cond, latent)` (cztery argumenty `ref_*` to gotowe dicty
  `{nazwa: wartość}` - wrapper woła `_build_ref_slot_dicts` na płaskich
  slotach i przekazuje wynik dalej). Czysta relokacja: stare klasy to
  teraz cienkie wrappery, ich sygnatury / INPUT_TYPES / IS_CHANGED bez
  zmian. Dual node woła `_execute_*_once` DWA razy (raz per rozdzielczość),
  wszystkie pozostałe argumenty współdzielone.
- Zero logiki warunkowej po naszej stronie: dual node zawsze woła pełną,
  niezmienioną ścieżkę encode dwa razy, istniejący fingerprint/proxy sam
  decyduje HIT/MISS dla drugiej rozdzielczości.
- `__init__.py`: cztery nowe linie (2 importy klas + wpisy w
  NODE_CLASS_MAPPINGS / NODE_DISPLAY_NAME_MAPPINGS).
- `tests/test_node_fl2va_dual.py` i `tests/test_node_ref2va_dual.py`: nowe
  pliki (9 testów każdy), reużyty istniejący harness (FakeRealClip,
  _patch_common, monkeypatch na stockowym execute, prawdziwy CachedClipProxy
  na tmp_path).
- `tests/test_node.py` `test_f`: zbiór oczekiwanych kluczy
  NODE_CLASS_MAPPINGS rozszerzony o dwa nowe node'y (jedyna dozwolona
  zmiana w istniejących testach - "mapowanie kluczy jak przy CLIP Name").
- Weryfikacja: `python -m py_compile nodes.py __init__.py` OK; pełny pytest
  **289 passed, 0 skipped, 0 failed** (było 271, +18 nowych). Output w
  scratchpadzie sesji: `dualres_test_output.txt`.

## Ustalenia istotne dla Chat

### Dowód empiryczny: cache sam obsługuje oba przypadki (test 6c ZLECENIA)
Dwa warianty `fake_execute` w KAŻDYM z nowych plików testowych, przeciw
prawdziwemu `CachedClipProxy` + prawdziwemu `compute_fingerprint` (tylko
stockowy `MiniMaxH3ImageToVideo/ReferenceToVideo.execute` jest zamockowany):
- "resolution-independent": `fake_execute` NIE wplata width/height w to, co
  idzie do `clip.tokenize(...)` -> dual node z RÓŻNYM (width,height) vs
  (width2,height2) daje `real_clip.encode_calls == 1` (druga rozdzielczość
  = HIT przez niezmieniony fingerprint) - `test_dual_resolution_independent_input_encodes_once`.
- "resolution-dependent": `fake_execute` wplata width/height w kształt
  tensora idącego do `clip.tokenize(...)` (symuluje keyframe/ref_image po
  `_resize`) -> `real_clip.encode_calls == 2` -
  `test_dual_resolution_dependent_input_encodes_twice`.
- Bonus: `test_dual_same_resolution_twice_still_encodes_once` - nawet przy
  resolution-dependent input, width2==width/height2==height => 1 encode.

### Fakty potwierdzone w kodzie
- Stockowe `MiniMaxH3ImageToVideo.execute` / `MiniMaxH3ReferenceToVideo.execute`
  (`comfy_extras/nodes_minimax_h3.py`) NIE mutują wejściowych tensorów ani
  dictów: `_resize()` tworzy nowe tensory, ref dicty są tylko czytane przez
  `.values()` / `.items()`. Dlatego dual Ref2VA bezpiecznie przekazuje te
  SAME obiekty dictów do obu wywołań `_execute_ref2va_once`
  (`nodes.py` - metoda `execute` klasy `MiniMaxH3CLIPCachedRef2VADualRes`).
- Dual node NIE trzyma nigdy dwóch encoderów naraz: `_execute_*_once`
  robi `del proxy; gc.collect(); soft_empty_cache()` we własnej ramce
  (jego finally), więc pierwsze wywołanie w pełni zwalnia encoder zanim
  drugie zbuduje swój proxy. Zgodne z twardą zasadą CLAUDE.md
  ("Nigdy nie trzymać więcej niż jednego dużego modelu rezydentnego").
- Wpisy cache dual node'a lądują pod istniejącymi wariantami "fl2va" /
  "ref2va" w Cache Managerze (`_execute_*_once` przekazuje te same stringi
  do `_sync_verbose_metadata` / `_record_last_used`). Brak zmian w
  `web/*.js`.
- Efekt uboczny (drobny, nieblokujący): dual node woła `_record_last_used`
  dwa razy, więc "aktywny wiersz" w Cache Managerze pokaże fingerprint
  DRUGIEJ rozdzielczości (width2/height2). Poza zakresem ZLECENIA.

## Otwarte pytania
- brak (KRYTERIUM_DONE spełnione: py_compile OK; targeted 85/0/0 w tym oba
  warianty 6c; pełny pakiet 289/0/0; git diff nodes.py = tylko
  _execute_*_once + dwie nowe klasy + cienkie wrappery; git diff
  __init__.py = tylko nowe wpisy).
- Do sprawdzenia przez użytkownika w żywym UI (CC nie może): render obu
  node'ów w edytorze, oba wyjścia positive/latent i positive_2/latent_2
  podłączają się poprawnie, realny podwójny encode przy keyframe'ach
  (dwie linie "Requested to load MiniMaxH3TEModel_" w logu) vs pojedynczy
  przy t2va bez keyframe'ów.

## Sugestie (nie polecenia)
- Konstrukcja bloku `optional` z ref_* slotami jest teraz zduplikowana
  między `MiniMaxH3CLIPCachedRef2VA.INPUT_TYPES` a
  `MiniMaxH3CLIPCachedRef2VADualRes.INPUT_TYPES` (~10 linii pętli). Można
  by kiedyś wyciągnąć wspólny helper - świadomie NIE zrobione teraz, bo
  ZLECENIE wymagało "bez zmiany INPUT_TYPES" istniejącego node'a.
