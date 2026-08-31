# HANDOFF

## Stan na: 2026-08-31 / branch master / commit (po tej sesji)

## Ostatnio zrobione
- **Nowy opcjonalny bool `generate_upscale_cond` (default True) w obu
  klasach DualRes** (`MiniMaxH3CLIPCachedFL2VADualRes`,
  `MiniMaxH3CLIPCachedRef2VADualRes`).
  - Gdy `False`: drugie wywołanie `_execute_*_once` (rozdzielczość
    upscale) NIE wykonuje się w ogóle - zero kosztu encode/VRAM.
    `execute()` zwraca `(cond, latent, None, None)` natychmiast po
    pierwszym (bazowym) wywołaniu. `_pair_verbose_entries()` też się nie
    woła (brak drugiego fingerprintu do sparowania), nie powstaje żaden
    sidecar z `paired_fingerprint`.
  - Gdy `True` / pominięty: zachowanie identyczne jak przed zmianą (drugie
    wywołanie + parowanie).
  - Powód istnienia przełącznika: dual-res node to jedno atomowe wywołanie
    Pythona zwracające 4 wyjścia naraz. ComfyUI nie umie wykonać go
    "częściowo", więc bypass konsumenta `positive_upscale`/`latent_upscale`
    w dół grafu (np. cały łańcuch upscalera z mode=4) NIE oszczędza kosztu
    encode'u - node i tak musi się wykonać w całości dla pozostałych
    wyjść. Ten bool jest jedynym mechanizmem, który faktycznie pomija ten
    encode. Udokumentowane w docstringach obu klas ("Do not fix this by
    making the second encode conditional on something else").
- `git diff nodes.py` ograniczony do: docstringów obu klas DualRes, ich
  `INPUT_TYPES()` (nowy wpis `generate_upscale_cond` w `optional`, przed
  `cache_mode`) i ich `execute()` (nowy parametr + wczesny return). Zero
  zmian w `_execute_fl2va_once` / `_execute_ref2va_once` /
  `_pair_verbose_entries`.
- Tooltip nowego wejścia (dosłownie, oba node'y): "When off, the second
  (upscale-resolution) encode is skipped entirely - positive_upscale/
  latent_upscale come back as None. Turn off for a plain generation where
  nothing downstream uses the upscale outputs; turn on when you actually
  need them. Bypassing the upscale consumer downstream does NOT skip this
  encode by itself - this is the only thing that does, because the node
  runs as one atomic call."

## Testy (workflow test-first: nowe testy napisane i widziane jako FAIL przed implementacją)
- `tests/test_node_fl2va_dual.py` i `tests/test_node_ref2va_dual.py`, po 3
  nowe testy każdy:
  - `test_dual_generate_upscale_cond_false_skips_the_second_encode` - na
    input zależnym od rozdzielczości (normalnie 2 realne encode): przy
    `generate_upscale_cond=False` `real_clip.encode_calls == 1`,
    `unload_calls == 1`, zwrócone `(cond_upscale, latent_upscale) ==
    (None, None)`.
  - `test_dual_generate_upscale_cond_false_does_not_pair` - spy na
    `node_module._pair_verbose_entries`: `call_count == 0`, powstaje
    dokładnie 1 sidecar `.verbose.json`, bez `paired_fingerprint`.
  - `test_dual_generate_upscale_cond_true_is_the_default` - jawne
    `=True` == pominięcie: 2 encode, parowanie wołane raz.
- Zaktualizowane (schema faktycznie się zmieniła - dodane wejście
  opcjonalne): `test_dual_input_types_adds_second_resolution_only` w obu
  plikach - oczekiwany zbiór `optional` powiększony o
  `generate_upscale_cond`, plus asercje typu (`BOOLEAN`), default (`True`)
  i obecności tooltipa. Testy behawioralne dual (encode-count, pairing,
  shared-inputs) NIE ruszane.
- `python -m py_compile nodes.py` - OK.
- Pełny pytest: **309 passed, 0 skipped, 0 failed** (`conda run -n
  comfyenv python -m pytest`). Przed sesją: 303 passed.
- Output zapisany w scratchpadzie sesji:
  `generate_upscale_cond_result.txt`.

## Ustalenia istotne dla Chat
- `MiniMaxH3CLIPCached{FL2VA,Ref2VA}DualRes.execute()` mają teraz sygnaturę
  z `generate_upscale_cond=True` jako ostatnim parametrem
  (nodes.py, po `cache_mode="auto"`).
- Wczesny return `(cond, latent, None, None)` przy `False` - konsumenci
  `positive_upscale`/`latent_upscale` w grafie muszą tolerować `None`
  (ComfyUI standardowo pozwala nie podłączać wyjścia; podłączony
  konsument dostanie `None`).
- `IS_CHANGED` NIE zmieniane: `generate_upscale_cond` to literalne wejście
  node'a, więc własny execution-cache ComfyUI i tak wymusza re-exec przy
  jego zmianie - nie trzeba go składać do sygnatury.
- Nowe wejście jest w `optional`, tuż przed `cache_mode`. Dla Ref2VA
  dodane w `INPUT_TYPES()` po `_ref_slots_input_spec()`, przed
  `optional["cache_mode"]`.

## Otwarte pytania
- brak.
- **Do sprawdzenia przez użytkownika w żywym ComfyUI** (CC nie może):
  realny render checkboxa `generate_upscale_cond` w obu node'ach,
  zachowanie grafu gdy `positive_upscale`/`latent_upscale` = `None` trafia
  do podłączonego (nie zbypassowanego) konsumenta, potwierdzenie w logu
  serwera że przy `False` linia "Requested to load MiniMaxH3TEModel_" nie
  pojawia się drugi raz.

## Sugestie (nie polecenia)
- Ewentualny follow-up: analogiczny log INFO w `execute()` przy
  `generate_upscale_cond=False` ("upscale encode skipped by
  generate_upscale_cond=False") - spójne z zasadą projektu "jasno
  wypisuj którą ścieżką poszło wykonanie". Świadomie pominięte teraz, bo
  ZLECENIE ograniczało diff do INPUT_TYPES/execute/docstringów i nie
  wspominało o logowaniu.
