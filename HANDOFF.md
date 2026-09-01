# HANDOFF

## Stan na: 2026-09-01 / branch master / commit 855f480

## Ostatnio zrobione (weryfikacja README_WORKING.md względem kodu)

Zadanie kontrolne (ZLECENIE_DLA_CC): potwierdzenie, że treść
`README_WORKING.md` (docelowy monolit README.md + docs/*.md, jeszcze
niepublikowany -- czekamy na screeny) zgadza się z aktualnym stanem kodu.
Bez zmian w kodzie. Pełny pytest: 398 passed (`testpaths = tests`,
398 collected). ComfyUI runtime: `comfyui_version.__version__ == "0.34.2"`
(pyproject "0.34.2", `git describe` = v0.34.2).

Wynik: wszystkie sprawdzane punkty ZGODNE z kodem. Szczegóły niżej.

## Ustalenia istotne dla Chat

- Display name wszystkich 5 zarejestrowanych node'ów -- `__init__.py:71-76`
  (`NODE_DISPLAY_NAME_MAPPINGS`):
  - `MiniMaxH3CLIPCachedFL2VA` -> "MiniMax H3 CLIP-Cached FL2VA"
  - `MiniMaxH3CLIPCachedFL2VADualRes` -> "MiniMax H3 CLIP-Cached FL2VA (Dual Resolution)"
  - `MiniMaxH3CLIPCachedRef2VA` -> "MiniMax H3 CLIP-Cached Ref2VA"
  - `MiniMaxH3CLIPCachedRef2VADualRes` -> "MiniMax H3 CLIP-Cached Ref2VA (Dual Resolution)"
  - `MiniMaxH3CLIPName` -> "MiniMax H3 CLIP Name"
  Stare "MiniMax H3 CLIP-Cached Images to Video" NIE występuje nigdzie w kodzie.
- Ref2VA jest w pełni cache'owane. `nodes.py:820` `_execute_ref2va_once()`
  buduje `CachedClipProxy` przez `_build_cached_proxy()` (`nodes.py:836`)
  i podstawia go do stockowego `MiniMaxH3ReferenceToVideo.execute()`
  (`nodes.py:840`). `CachedClipProxy` (`proxy.py:55`) jest generyczny wobec
  `tokenize(prompt, **kwargs)` -- ta sama ścieżka HIT/MISS/REFRESH co FL2VA.
- Cache Manager jest w kodzie na master (backend + UI, scalone):
  `minimaxh3_clipcache/routes.py` rejestruje 5 endpointów pod
  `/h3_cache_manager` na `PromptServer.instance.routes`; import w
  `__init__.py:58` (best-effort). Frontend: `WEB_DIRECTORY = "./web"`
  (`__init__.py:80`), `web/main.js` (~59 KB), `web/styles.css`.
- Dual Resolution: oba warianty zaimplementowane -- `MiniMaxH3CLIPCachedFL2VADualRes`
  (`nodes.py:587`) i `MiniMaxH3CLIPCachedRef2VADualRes` (`nodes.py:941`).
  Oba: `RETURN_NAMES = ("positive", "latent", "positive_upscale", "latent_upscale")`
  (`nodes.py:669`, `nodes.py:1027`), input `generate_upscale_cond` BOOLEAN
  default `True` (`nodes.py:651`, `nodes.py:987`), plus `width_upscale` /
  `height_upscale`.
- Testy: pytest 398 passed / 398 collected. Draft ("398 passed") aktualny;
  "45 testów" z żywego README nieaktualne. Skrypty w `scripts/` to osobne
  diagnostyki real-model, NIE wchodzą do `pytest` (`pytest.ini`:
  `testpaths = tests`).
- Wersja ComfyUI: 0.34.2 (nie 0.34.0). README_WORKING linia 102 zgodne.
- Ścieżki plików referencjonowane w dokumentacji -- wszystkie istnieją:
  `minimaxh3_clipcache/proxy.py`, `minimaxh3_clipcache/fingerprint.py`,
  `scripts/test_stock_vs_cache.py`, `scripts/test_ref2video_equivalence.py`,
  `scripts/test_clip_unload_isolation.py`, `scripts/test_vae_memory_isolation.py`.
  Także `minimaxh3_clipcache/encoder_abi.py`, `serialize.py`, `store.py`.
- Kategoria UI: wszystkie 5 node'ów mają
  `CATEGORY = "model/conditioning/minimax/cached"` (`nodes.py:565,671,903,1029,1117`).
- `cache_mode`: tylko `["auto", "refresh"]`, default `"auto"` -- w 4 node'ach
  wykonawczych (`nodes.py:553,658,878,994`). `CLIPName` nie ma `cache_mode`.
  Brak trzeciej wartości. Wewnętrzne `force_refresh` wymuszane też gdy ABI
  encodera niedostępne (`nodes.py:359`) -- to jest opisane w README_WORKING
  (sekcja "Encoder ABI"). `cache_only` nieistniejące (README: "planned, not
  implemented").
- Ścieżka cache: `CACHE_DIR = os.path.join(REPO_ROOT, "cache")`,
  `REPO_ROOT = os.path.dirname(os.path.abspath(__file__))` w `nodes.py:38-39`
  (nodes.py w korzeniu repo). `routes.py:52` liczy tę samą ścieżkę
  (dirname o poziom wyżej z `minimaxh3_clipcache/`). Efektywnie:
  `ComfyUI/custom_nodes/ComfyUI-MiniMaxH3-CLIPCached/cache`. Zgodne.
- `CACHE_SCHEMA_VERSION = 2` (`fingerprint.py:20`). Zgodne z opisem "schema v2".
- Pliki `docs/*.md` (NODE_GUIDE, CACHE_MANAGER, PERFORMANCE, TECHNICAL_DETAILS,
  TESTING_AND_LIMITATIONS) jeszcze NIE istnieją jako osobne pliki -- treść jest
  w monolicie `README_WORKING.md`. Oczekiwane przed splitem, nie rozbieżność.

## Otwarte pytania

- brak

## Sugestie (nie polecenia)

- brak
