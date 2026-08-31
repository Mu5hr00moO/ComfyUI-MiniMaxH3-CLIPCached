# HANDOFF

## Stan na: 2026-09-01 / branch master / commit c88e315

## Ostatnio zrobione (sesja porządkowa — 3 niezależne zadania)

### 1. Log INFO przy generate_upscale_cond=False (commit 7bce233)
- `execute()` obu klas DualRes: tuż przed `return (cond, latent, None,
  None)` dodane:
  ```
  logger.info(
      "[UPSCALE COND SKIPPED] %s: generate_upscale_cond=False - "
      "positive_upscale/latent_upscale not computed", fp1[:12],
  )
  ```
  Format zgodny z realną konwencją pliku (bracket-tag + `%s: ` opis +
  `- ` konsekwencja, jak `[CACHE HIT]` / `[VERBOSE PAIRING FAILED]` w
  proxy.py/nodes.py). Tekst loga NIE zmieniony względem propozycji
  ZLECENIA.
- `git diff nodes.py` = tylko te dwa bloki logu (4 linie każdy), nic
  więcej.
- Nowe testy caplog (po 1 w każdym pliku dual):
  `test_dual_generate_upscale_cond_false_logs_one_info_line` — przy
  `False` dokładnie jeden rekord INFO z tagiem `[UPSCALE COND SKIPPED]`
  zawierający pierwsze 12 znaków fingerprintu bazowego (czytane z
  zapisanego sidecara `*.verbose.json`); przy `True` (default) tego
  rekordu NIE ma.

### 2. README.md — tekst, bez screenshotów (commit 7782e31)
- Sekcja Installation: "Two new nodes" -> "Five new nodes" + lista 5
  node'ów z linkami do nowych sekcji.
- Nowa sekcja `## The CLIP Name node` — jeden widget (ten sam dropdown
  `models/text_encoders`), jedno wyjście, do podpięcia w wielu FL2VA/
  Ref2VA/DualRes przez "Convert widget to Input"; wzmianka że output
  dopasowuje się do dowolnej listy plików encodera bez restartu (bez
  wchodzenia w `_ComboType`).
- Nowa sekcja `## Dual Resolution variants` — tabelka wejść
  (`width_upscale`/`height_upscale`, `generate_upscale_cond`),
  wyjaśnienie że to feature spójności a NIE optymalizacja (fingerprint i
  tak dedupuje gdy piksele wychodzą identyczne), podsekcja
  `### generate_upscale_cond` z wyraźnym akapitem że bypass downstream
  konsumenta NIE pomija drugiego encode (atomowe wywołanie, 4 wyjścia
  naraz), wzmianka o logu `[UPSCALE COND SKIPPED]`.
- `## Cache Manager` — nowy akapit "Dual-resolution pairing": składanie
  dwóch wpisów w jeden wiersz (bazowy) z plakietką "+ rescaled to WxH",
  Delete NIE kaskaduje.
- Istniejące sekcje FL2VA/Ref2VA/How the cache works — nietknięte.

### 3. TODO.md — nowy plik w korzeniu repo (commit c88e315)
Backlog świadomie odłożonych wątków (żeby nie zniknęły z pamięci
projektu). 6 pozycji:
- decoupling `scripts/test_proxy_gate.py` (rola gate vs. moduł fixtur —
  4 importerzy: test_stock_vs_cache, test_ref2video_equivalence,
  test_clip_unload_isolation, test_vae_memory_isolation) — z ZLECENIA.
- 2 edge case'y UI pairing (search chowa base a pokazuje upscale ->
  pusty render; wpis `inconsistent` jako base pary -> brak plakietki
  rescale) — z ZLECENIA, zweryfikowane w web/main.js
  (`renderList`/`buildInconsistentRow`, commit 576b0c4).
- opcjonalny explicit paired-delete — z ZLECENIA.
- dynamiczne sloty referencji Ref2VA zamiast stałych 9/3/3/3 — z
  ZLECENIA (zweryfikowane: stock używa `io.Autogrow.Input` +
  `minimax_ref_items=`).
- **[dodane przeze mnie, nie z ZLECENIA]** `cache_mode="cache_only"` —
  wspomniane jako planned w README "Limitations", niezaimplementowane.
- **[dodane przeze mnie, nie z ZLECENIA]** śledzenie nazw plików
  referencji przez dedykowane wrappery loaderów — wskaźnik do pełnego
  uzasadnienia w sekcji "Rozważone i ODŁOŻONE" CLAUDE.md.

## Ustalenia istotne dla Chat
- Finalny tekst loga (obie klasy DualRes, nodes.py, tuż przed early
  return przy `generate_upscale_cond=False`): tag `[UPSCALE COND
  SKIPPED]`, poziom INFO, format `"[UPSCALE COND SKIPPED] %s:
  generate_upscale_cond=False - positive_upscale/latent_upscale not
  computed"` z argumentem `fp1[:12]`. Bez zmian względem propozycji
  ZLECENIA.
- Log NIE wchodzi do fingerprintu ani HIT/MISS — czysto informacyjny,
  spójny z zasadą projektu "jasno wypisuj którą ścieżką poszło
  wykonanie".
- `python -m py_compile nodes.py` — OK.
- Pełny pytest: **311 passed, 0 skipped, 0 failed** (przed sesją
  porządkową: 309). Nowe: 2 testy caplog.
- Output sesji: scratchpad `session_cleanup_result.txt`.
- Commity tej sesji: 72f8c9c (generate_upscale_cond — poprzednie
  ZLECENIE), 7bce233 (log), 7782e31 (README), c88e315 (TODO.md).

## Otwarte pytania
- brak.
- **Do sprawdzenia przez użytkownika w żywym ComfyUI** (CC nie może):
  linia `[UPSCALE COND SKIPPED]` faktycznie w logu serwera przy
  `generate_upscale_cond=False` + brak drugiego "Requested to load
  MiniMaxH3TEModel_"; render checkboxa w obu node'ach DualRes; render
  nowych sekcji README (Markdown na GitHub — kotwice `#the-clip-name-node`,
  `#dual-resolution-variants`).
- Wciąż zaległe (nie z tej sesji): screenshoty RAM/VRAM MISS vs HIT do
  README (CLAUDE.md "R10 prep" / komentarz `<!-- TODO ... -->` w README) —
  task dla użytkownika, wymaga nvitop przy żywej generacji.

## Sugestie (nie polecenia)
- Rozważyć przeniesienie sekcji "No cache-only mode" z README
  "Limitations" do jednego miejsca — teraz jest i w README, i w TODO.md
  (świadoma duplikacja: README = user-facing limitation, TODO.md =
  backlog dev). Jeśli kiedyś `cache_only` powstanie, usunąć obie wzmianki
  razem.
