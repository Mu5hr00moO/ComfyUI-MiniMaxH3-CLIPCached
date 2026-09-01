# HANDOFF

## Stan na: 2026-09-01 / branch master / commit 5c5b2f1

## Ostatnio zrobione (3 poprawki wording/dokumentacja z finalnego audytu Grok+Codex)

Żadna nie zmienia zachowania kodu produkcyjnego — tylko treść README,
jednego loga WARNING, jednego komentarza i skryptu testowego. Trzy commity,
po jednym na punkt. Pełny pytest 349 passed przed i po.

### 1. DualRes: "encoder loads at most once" tylko dla cache_mode="auto" (commit 550b2cb)
- README (tabela `width_upscale`/`height_upscale` + akapit "This is a
  consistency feature") oraz docstringi obu węzłów DualRes
  (`nodes.py` `MiniMaxH3CLIPCachedFL2VADualRes` / `...Ref2VADualRes`) i
  4 tooltipy `width_upscale`/`height_upscale` twierdziły, że druga
  rozdzielczość to zwykły cache HIT i encoder ładuje się co najwyżej raz.
- To prawda WYŁĄCZNIE dla `cache_mode="auto"`. Przy `cache_mode="refresh"`
  proxy jest budowane z `force_refresh=True` (`nodes.py:330`), więc obie
  rozdzielczości idą ścieżką REFRESH (`proxy.py:141-142`) i re-enkodują
  niezależnie od zgodności fingerprintu — encoder ładuje się dwa razy.
- Dopisane jawne zdanie o tym w README (nowy akapit) i w obu docstringach;
  tooltipy przeformułowane na "with cache_mode auto ... ; cache_mode
  refresh always re-encodes".

### 2. Log ENCODER ABI UNAVAILABLE — realny efekt, nie "cache wyłączony" (commit 28d142d)
- `encoder_abi.py:55-63`: było "disk caching is disabled for this session
  (every run will be a real encode, cache_mode is ignored)". Sugerowało
  całkowite wyłączenie cache'a.
- Realnie (potwierdzone w `proxy.py:132-231`): wyłączony jest tylko
  HIT/reuse. Przy `available=False` oba węzły wymuszają `force_refresh=True`
  + `encoder_abi_id="unavailable"`; udany encode NADAL woła
  `save_conditioning()` (`proxy.py:223`) — zapisuje pliki cache pod
  fingerprintem z sentinelową wartością ABI.
- Nowa treść: "cache HIT/reuse is disabled for this session: every run
  encodes for real regardless of cache_mode ... A successful encode may
  still write cache files, under a sentinel fingerprint for this unknown
  ABI". Prefiks `[ENCODER ABI UNAVAILABLE]` i `(%s)` z wyjątkiem
  zachowane — test `test_encoder_abi.py::test_d_...` asercjuje tylko
  prefiks, nie wymagał zmiany.
- Ten sam/równoważny komentarz w `nodes.py:288-292` (`_is_changed_common`)
  też poprawiony ("a cache HIT/reuse is unsafe this session ... a
  successful encode may still write cache files ... only reuse is
  suppressed").

### 3. test_ref2video_server_e2e.py — usunięta fałszywa asercja wewnątrz-sesyjnego HIT-a (commit 5c5b2f1)
- Druga iteracja wysyłała bajt-identyczny graf w tej samej sesji serwera
  i asercjowała `[CACHE HIT]`. Własny execution cache ComfyUI
  ("Prompt executed in 0.00 seconds") przechwytuje to ZANIM nasz węzeł
  wykona się drugi raz — ten "HIT" nie dowodził niczego o CachedClipProxy.
- `test_ref2video_server_hit.py` już poprawnie dowodzi realnego proxy
  HIT-a przez ŚWIEŻY serwer (pusty execution cache) na fingerprincie z
  `/tmp/r7_last_fingerprint.txt`.
- Skrypt teraz: jedna submisja, weryfikacja rejestracji węzłów + realnego
  MISS-a, zapis fingerprintu dla follow-upu. Docstring i verdict opisują
  zawężony zakres i wskazują `test_ref2video_server_hit.py` jako dowód
  HIT-a.

## Ustalenia istotne dla Chat
- `python -m py_compile` na wszystkich zmienionych `.py` — OK po każdym commicie.
- `git diff --check` — czysty po każdym z 3 commitów.
- `pytest tests/test_encoder_abi.py` — 5 passed (punktowo, po commicie 2).
- Pełny pytest (`conda run -n comfyenv python -m pytest -q`): **349 passed**,
  identycznie przed i po sesji (zmiany są doc-only).
- Commity tej sesji: 550b2cb (DualRes auto vs refresh), 28d142d (log ABI),
  5c5b2f1 (e2e script scope).
- `MiniMaxH3ImageToVideo.execute()` w tej wersji ComfyUI ma sygnaturę
  keyword-based (`clip=`, `vae=`, `prompt=`, `width=`, `height=`,
  `length=`, `first_frame=`, `last_frame=`) — `_execute_fl2va_once`
  (`nodes.py:470`) wywołuje ją tak.

## Otwarte pytania
- brak.

## Sugestie (nie polecenia)
- brak nowych w tej sesji.
