# ComfyUI-MiniMaxH3-CLIPCached — kontekst projektu

## Cel
Custom node ComfyUI: "MiniMax H3 CLIP-Cached Images to Video". Cache'uje wynik
text/vision-encodingu (Qwen3-VL, przez natywny obiekt CLIP ComfyUI z
clip_type=MINIMAX) na dysku, żeby przy powtórzonym prompt+first_frame+
last_frame nie trzeba było w ogóle ładować ~27 GB encodera.

## Zasada nadrzędna
Nie tworzymy własnej wersji MiniMaxH3ImageToVideo. Tworzymy przezroczysty
cached-CLIP proxy i pozwalamy stockowemu node'owi (comfy_extras/nodes_minimax_h3.py)
wykonać całą właściwą mechanikę H3 (resize, VAE encode keyframes, AV latent,
minimax_keyframes). Nigdy nie kopiujemy ani nie reimplementujemy tej logiki.

## Potwierdzone fakty o środowisku (nie zakładać nic ponad to bez ponownej weryfikacji)
- ComfyUI lokalnie: v0.34.0, w /home/kamil/ComfyUI
- MiniMaxH3ImageToVideo.execute(clip, vae, prompt, width, height, length,
  first_frame=None, last_frame=None) w comfy_extras/nodes_minimax_h3.py
  - na clip wywołuje WYŁĄCZNIE: clip.tokenize(prompt, images=images) i
    clip.encode_from_tokens_scheduled(tokens)
  - images przekazywane do tokenize() są JUŻ po resize dopasowanym do
    width/height (funkcja _resize, dzieje się PRZED tokenize, poza
    obiektem clip) — więc hashowanie tego, co proxy dostanie w tokenize(),
    jest równoważne hashowaniu dokładnego wejścia obrazkowego Qwena, bez
    potrzeby reimplementacji _resize
  - minimax_keyframes jest doklejane do cond PO encode_from_tokens_scheduled,
    przez VAE, niezależnie od clip — cache ma obejmować WYŁĄCZNIE surowy
    output encode_from_tokens_scheduled, nic więcej
- comfy.model_management.unload_model_and_clones(model, unload_additional_models=True,
  all_devices=False) — bezpieczne z domyślnymi argumentami: keep_loaded
  explicite zachowuje wszystko niepowiązane po clone_base_uuid; nie zdejmuje
  niepowiązanych modeli
- clip.patcher jest przypisywane bezwarunkowo w __init__ klasy CLIP w
  comfy/sd.py (self.patcher = ModelPatcher(...)) — dostępne zawsze,
  niezależnie od clip_type
- Encoder H3 lokalnie: models/text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors
  (~27,1 GB)
- Repo tego node'a: własny .git, NIEZALEŻNE od repo głównego ComfyUI
  (które go ignoruje przez regułę /custom_nodes/)
- nodes.CLIPLoader() z clip_type=MINIMAX w trybie auto/lowvram tworzy
  tylko lekki obiekt/patcher - faktyczny transfer wag na GPU (~27GB)
  dzieje się leniwie, przy PIERWSZYM realnym wywołaniu tokenize/encode,
  nie przy samym load_clip(). Potwierdzone empirycznie: pierwszy realny
  encode ~115s, kolejny realny encode na już-rezydentnym modelu ~1.2s,
  cache HIT ~0.0s. Ważne dla interpretacji przyszłych benchmarków w README
  (faza 25) - nie mylić kosztu "zimnego" encode na rozgrzanym modelu z
  kosztem pełnego cold-startu.
- Trzeci zaobserwowany wariant czasu encode (po ~115s cold i ~1.2s warm
  w izolowanych skryptach): pełny serwer ComfyUI, encoder ciepły w OS
  page cache z wcześniejszych sesji, "dynamic VRAM loading" (streaming
  wag zamiast blokującego transferu całości) - MISS przez prawdziwe
  /prompt API zajął 19.6s. Trzy różne liczby dla "MISS" w zależności od
  stanu page cache i trybu ładowania - nie traktować rozbieżności między
  benchmarkami jako niespójności w logice cache'a, tylko jako różnicę w
  warunkach środowiskowych.
- Faza 18 potwierdzona end-to-end w prawdziwym ComfyUI (nie tylko w
  testach jednostkowych): node ładuje się bez błędu, prawdziwy MISS przez
  /prompt API przechodzi (real CLIP proxy -> real Qwen3-VL encode ->
  conditioning+latent -> targeted unload), VRAM wraca do baseline po
  zatrzymaniu serwera.
- __init__.py dodaje katalog repo do sys.path (append, NIE insert(0) -
  potwierdzone że globalny moduł "nodes" ComfyUI jest już w sys.modules
  na tym etapie startu, więc nie ma ryzyka przechwycenia), bo nasz
  własny pakiet "minimaxh3_clipcache" nigdy nie trafia na sys.path automatycznie
  przez load_custom_node(), w przeciwieństwie do "nodes"/"comfy"/
  "folder_paths" które są już zaimportowane globalnie przed dotarciem
  do custom_nodes.
- Idle VRAM baseline (nvidia-smi, zero procesów CUDA) na tej maszynie
  WSL2 waha się w paśmie ~1100-1650 MiB (obserwowane wartości: 1117,
  1162, 1243, 1647 MiB w różnych testach). To narzut sterownika/
  kompozytora hosta Windows przy GPU passthrough do WSL2, nie wyciek
  ani proces-widmo. Traktować dowolną wartość w tym paśmie przy braku
  procesów jako czysty stan wyjściowy - nie zatrzymywać się przy każdym
  drobnym odchyleniu w tym zakresie.
- ComfyUI NIE deduplikuje niezależnych wywołań comfy.sd.load_clip() na
  ten sam plik - dwa osobne wywołania tworzą dwa osobne, w pełni
  rezydentne obiekty modelu w VRAM jednocześnie. Nigdy nie trzymać
  więcej niż jednego dużego (~15GB+ na GPU) modelu tekst-encodera
  rezydentnego naraz na tej karcie (16GB) - zawsze unload przed
  kolejnym load, nie tylko na samym końcu skryptu/testu.
- Potwierdzony, bezpieczniejszy tryb awarii: czyste
  torch.OutOfMemoryError (proces kończy się, sterownik odzyskuje VRAM
  sam, brak wpisów OOM w dmesg) jest jakościowo różny od wcześniejszego
  pełnego crasha hosta - tamten wynikał z RÓWNOLEGŁEJ generacji w
  ComfyUI + testu, nie z samego przekroczenia VRAM przez jeden
  kontrolowany proces.

### Faza 24: dochodzenie w sprawie narastającego RAM/VRAM - ROZWIĄZANE

Początkowa obserwacja (faza 23): powtarzane load/unload CLIP w jednym
procesie skryptu testowego pokazywało narastające, nieodzyskiwane VRAM
(16GB -> 30GB -> 44GB -> 55GB) prowadzące do torch.OutOfMemoryError,
a osobny izolowany test pokazał ~13GB rezydualnego RSS (RAM procesu)
po JEDNYM cyklu load/unload/del/gc.collect() - ekstrapolacja tego tempa
(kilka cykli z rzędu) doprowadziła do realnego przepełnienia RAM systemu
i wymusiła ręczny SIGKILL.

Przyczyna źródłowa: OBA niebezpieczne testy ładowały CLIP przez ręczne
`import nodes; nodes.CLIPLoader()...` z pominięciem main.py, co oznacza
że comfy.memory_management.aimdo_enabled pozostawało False (flaga ta
jest ustawiana WYŁĄCZNIE w main.py:300, nigdy automatycznie). Bez aimdo
(DynamicVRAM), ComfyUI ładuje cały 27GB state_dict w pełni do RAM/VRAM
procesu przez klasyczną ścieżkę safetensors.safe_open() - i coś w tej
ścieżce (nie ustalone dokładnie które ogniwo - potencjalnie mmap Arc
w bibliotece safetensors trzymany żywy dłużej niż oczekiwano, nie
potwierdzone ostatecznie) zostawia rezydualne RSS między cyklami.

Próba odtworzenia aimdo=True ręcznie w izolowanym skrypcie (poza main.py)
NIE POWIODŁA SIĘ - comfy_aimdo.host_buffer.HostBuffer rzuca AttributeError
('NoneType' object has no attribute 'hostbuf_allocate'), bo natywna
biblioteka C (lib) zostaje None gdy aimdo jest inicjalizowane poza
dokładną sekwencją main.py. To ograniczenie/bug w comfy_aimdo (cudzym
kodzie) - świadomie nieścigane dalej, poza zakresem tego projektu.

Rozstrzygający test (krok 3c): trzy różne prompty wysłane sekwencyjnie
przez prawdziwe /prompt API do żywego `python main.py` (gdzie aimdo
faktycznie się poprawnie inicjalizuje) dały PRAKTYCZNIE PŁASKI trend RSS
serwera (2.51GiB -> 2.55GiB -> 2.57GiB, przyrosty rzędu dziesiątek MB,
nie gigabajtów) mimo trzech niezależnych, realnych cache MISS (potwierdzone
trzema różnymi fingerprintami i trzema różnymi czasami wykonania w logu
serwera: 20.03s/18.85s/18.62s).

WNIOSEK: w prawdziwym środowisku produkcyjnym (serwer uruchomiony
normalnie przez `python main.py`, z działającym DynamicVRAM/aimdo) nasz
node NIE POWODUJE narastającego wycieku RAM/VRAM przy wielu kolejnych,
różnych promptach w jednej sesji serwera. Oba niebezpieczne incydenty
z tej fazy diagnozowały wyłącznie SZTUCZNĄ, nieprodukcyjną ścieżkę
(ręczne ładowanie CLIP z pominięciem main.py/aimdo), którą nasze wczesne
skrypty testowe (fazy 4-5, 12, 23) siłą rzeczy używały, bo omijają
main.py z założenia (to jest OK dla testowania POPRAWNOŚCI logiki
cache'a - torch.equal itd. - ale NIE nadaje się do wnioskowania o
zużyciu pamięci w realnych warunkach).

Nie testowano: bardzo długich sesji serwera (setki/tysiące promptów) ani
zachowania przy cache_mode="refresh" w pętli - jeśli w przyszłości pojawią
się realne zgłoszenia narastającej pamięci w produkcji, zacząć od
sprawdzenia czy aimdo faktycznie jest aktywne na danym sprzęcie
użytkownika (nie każdy sprzęt je wspiera - main.py robi auto-detekcję),
zanim szuka się dalej.

Zewnętrzny przegląd (Grok) trafnie wskazał rozjazd między planem fazy 17
(unload + del + gc.collect() + soft_empty_cache()) a faktycznym kodem
w nodes.py (miał tylko unload). Domknięte: dodano brakujące del/gc/
soft_empty_cache, zweryfikowano na 10 sekwencyjnych, realnych cache MISS
przez żywy serwer - RSS +110MB łącznie (~11MB/iteracja, spójne z narzutem
/history serwera, nie z encoderem), MemAvailable płaskie (~52.8GB przez
całość), czasy MISS identyczne co do 0.1s (21.0s) bez degradacji. Zmiana
była zgodnościowa (dopasowanie kodu do planu), nie krytyczną naprawą -
test na 3 iteracjach wcześniej i na 10 teraz dają ten sam, płaski wynik.

### Jak uruchamiać ComfyUI lokalnie
- Standardowy sposób odpalania ComfyUI w tym środowisku (WSL Ubuntu):
    conda activate comfyenv
    cd ~/ComfyUI
    python main.py
  Środowisko conda: "comfyenv". Jeśli jakikolwiek skrypt/test uruchamia
  serwer ComfyUI w NOWEJ, nieinteraktywnej powłoce (np. bash_tool w tle,
  bez odziedziczonego stanu terminala) - nie polegać na tym że
  "conda activate" zadziała automatycznie. Zamiast tego użyć jednego z:
    conda run -n comfyenv python main.py
  albo pełnej ścieżki do interpretera z tego środowiska
  (np. wynik `conda run -n comfyenv which python`).
- Potwierdzone lokalnie (nie zgadywać): domyślna nieinteraktywna powłoka
  bash_tool w tej sesji CC ma CONDA_DEFAULT_ENV=base i `python` wskazujący
  na /home/kamil/miniconda3/bin/python (base env, Python 3.14.6, BEZ
  torch — `import torch` rzuca ModuleNotFoundError). ~/.bashrc ma tylko
  standardowy blok `conda init`, bez żadnego `conda activate comfyenv`.
  Czyli samo "python main.py" bez `conda run -n comfyenv` w bash_tool
  NIE zadziała. Test end-to-end fazy 18 użył jawnie
  `conda run -n comfyenv --no-capture-output python main.py` i to
  zadziałało poprawnie — to jest potwierdzony, działający wariant do
  używania w przyszłych sesjach CC uruchamiających serwer ComfyUI ze
  skryptu/bash_tool.

- Dostępna dwa razy powtarzająca się obserwacja (bramka faza 4-5 i test
  roundtrip faza 12): RAM "available" nie wraca do stanu sprzed load po
  unload_model_and_clones - do zbadania jako PIERWSZY punkt fazy 24, nie
  incydentalnie.
- tests/conftest.py dodaje /home/kamil/ComfyUI do sys.path, co pozwala
  importować comfy.nested_tensor (samodzielny moduł, tylko torch) w
  pytest BEZ uruchamiania ComfyUI czy ładowania modelu - wzorzec do
  ponownego użycia, jeśli inne moduły comfy.* okażą się podobnie lekkie.

## Zasady kodowania (obowiązują przez cały projekt)
- Małe, recenzowalne zmiany: jeden commit = jedna logiczna zmiana. Nie
  mieszać refaktoru z nową funkcją.
- Zawsze najpierw: zrozum → zmień jedną rzecz → przetestuj → obejrzyj diff → commit.
- Kod w języku angielskim (nazwy, komentarze, komunikaty błędów, logi) —
  nawet jeśli rozmawiamy po polsku.
- Testuj od najmniejszej jednostki w górę: proxy → zgodność ze stockiem →
  cache hit/miss → inwalidacja → RAM/VRAM → dopiero potem README.
- Brak cichych fallbacków: błąd przy cache miss (np. OOM przy ładowaniu
  encodera) ma jawnie wybuchnąć z czytelnym komunikatem (co zawiodło, czego
  oczekiwano, co otrzymano) — nigdy nie zwracać pustego/domyślnego
  conditioningu ani nie użyć cichej starej wartości z cache.
- Brak pickle w formacie cache — tensory w safetensors, struktura osobno
  (JSON), zgodnie z tym co robi ComfyUI-H3-Multishot.
- Zapis cache atomowo: plik tymczasowy + os.replace() na koniec, nigdy
  bezpośredni zapis do docelowej ścieżki.
- Nie hashować całych plików modeli — identyfikacja encodera przez
  (clip_name, file_size, mtime_ns) z os.stat().
- Logowanie: jasno wypisuj, którą ścieżką poszło wykonanie (HIT/MISS),
  zajętość RAM/VRAM przed i po unload.
- Żadnych `git add .` — jawnie staged pliki. Nie commitować: cache/,
  __pycache__/, plików tymczasowych.
- Nie zgaduj API ComfyUI z pamięci/GitHuba — sprawdzaj lokalnie w tym repo
  (/home/kamil/ComfyUI), bo lokalna wersja może się różnić.
- Na końcu KAŻDEJ fazy/kroku, przed przejściem do następnego: uruchom
  `git status --short`. Jeśli jest tam cokolwiek niescommitowane, co
  zostało już zweryfikowane (testy przeszły, działanie potwierdzone) -
  scommituj to natychmiast, zanim jakikolwiek kolejny plik zacznie na
  tym polegać. Nie zostawiaj "wiszących" zmian między sesjami/fazami.

## Plan działania (kolejne fazy, nie przeskakiwać)
**STATUS: wszystkie fazy 1-25 zakończone. Ta sekcja to zapis
historyczny procesu, nie lista otwartych zadań.**

1. Osobne repo — GOTOWE
2. Baza = stockowy MiniMaxH3ImageToVideo, wywoływany bezpośrednio, bez
   kopiowania logiki
3. Weryfikacja lokalnego core przed każdą fazą — GOTOWE dla kontraktu clip
4. SpyClipProxy bez cache'a — deleguje 1:1, zapisuje co dostał — W TRAKCIE
5. Bramka go/no-go: MiniMaxH3ImageToVideo.execute() z podstawionym proxy
   musi zadziałać identycznie jak z prawdziwym clip — JEŚLI NIE, ZATRZYMAĆ
   PROJEKT I ZMIENIĆ ARCHITEKTURĘ
6. Test zgodności: stock CLIP vs proxy CLIP → identyczny CONDITIONING/LATENT
7. Proxy: tokenize(self, prompt, **kwargs) (nie sztywne images=None) —
   przyszłościowo pod minimax_ref_items= dla ref2va
8. Cache key liczony z danych PRZECHWYCONYCH przez proxy (post-resize), nie
   z surowych inputów node'a
9. Canonical request do fingerprintu: CACHE_SCHEMA_VERSION, clip identity,
   prompt, kwargs przekazane do tokenize(), wszystkie tensory z kwargs
10. Identyfikacja encodera: clip_name + file_size + mtime_ns
11. Hashowanie tensorów: tensor.detach().cpu().contiguous(), deterministyczna
    serializacja struktury (ustalona kolejność kluczy)
12. Dopiero teraz dokładamy cache do proxy — GOTOWE
13. Cache hit → prawdziwy CLIP/Qwen NIE jest w ogóle ładowany — GOTOWE
14. Cache miss → load real clip → real tokenize/encode → conditioning →
    zapis cache → targeted unload → return — GOTOWE
15. Cache zawiera WYŁĄCZNIE output encode_from_tokens_scheduled — nic
    więcej (nie AV latent, nie VAE keyframe latents, nie minimax_keyframes)
    — GOTOWE
16. Format cache: safetensors + osobna struktura/metadata, bez pickle,
    zapis atomowy — GOTOWE
17. Targeted unload: unload_model_and_clones(clip.patcher), potem del clip;
    gc.collect(); soft_empty_cache() — GOTOWE
18. Publiczny node NIE ma wejścia CLIP — ma clip_name (string) + leniwe
    ładowanie wewnątrz execute(), żeby HIT omijał CLIPLoader w grafie
    — GOTOWE, potwierdzone end-to-end w prawdziwym ComfyUI (nie tylko
    unit testami)
19. Reszta inputów/outputów identyczna ze stockiem poza tą jedną zamianą
20. Tryby v1: auto (hit→load, miss→encode+save), refresh (ignoruje cache,
    nadpisuje). cache_only później.
21. Testy inwalidacji: zmiana promptu/first_frame/last_frame/clip_name/
    podmiana pliku → MISS. Zmiana seed/sampler/steps/scheduler → HIT.
22. Test zgodności prompt+obraz: dokładny prompt i dokładne obrazy, bez
    prób semantycznej interpretacji "zgodności"
23. Test końcowy: stock == cached-MISS == cached-HIT (conditioning +
    finalne przygotowanie H3), tensory przez torch.allclose, nie exact
    equality
24. Test pamięci: RAM przed/podczas/po unload/podczas samplingu —
    kryterium: Qwen nie zostaje jako balast po przygotowaniu conditioningu
25. README + example workflow — DOPIERO na końcu, po przejściu wszystkich testów

Po utworzeniu pliku zrób git add CLAUDE.md i commit z opisem
"Add project context and plan for CC sessions". Nic więcej teraz nie rób.

## Cache Manager (drugi wątek)

Pełny plan w CACHE_MANAGER_PLAN.md. Kluczowy niezmiennik: cache jest
source of truth, manager jest warstwą indeksującą, fingerprint = ID
wpisu, prompt jest read-only w managerze.

### Faza 5 - lokalna konwencja PromptServer routes

Zweryfikowane lokalnie (grep w custom_nodes/ tej instalacji ComfyUI,
`server.py`), nie z pamięci. Dwie niezależne implementacje potwierdzają
ten sam wzorzec:
- `MiniMaxH3-Prompt-Writer/backend/routes.py` (+ `tests/test_routes_stability.py`)
- `ComfyUI-MemoryVisualization/__init__.py`

Ustalenia:
- **Rejestracja**: na poziomie modułu `routes = PromptServer.instance.routes`
  (import `from server import PromptServer`), potem dekoratory
  `@routes.get("/path")` / `@routes.post("/path")` na
  `async def handler(request: web.Request) -> web.Response`.
- **`PromptServer.instance` jest dostępny w momencie ładowania custom
  node'a** - obie referencyjne implementacje sięgają po niego wprost przy
  imporcie modułu, bez czekania. Potwierdzona kolejność w `main.py`:
  `server.PromptServer(asyncio_loop)` (linia ~536, ustawia
  `PromptServer.instance = self` i `self.routes = web.RouteTableDef()`)
  -> `nodes.init_extra_nodes()` (linia ~542, tu ładują się custom nodes,
  więc tu wykonuje się nasz `__init__.py` i dekoratory `@routes.get/post`
  rejestrują handlery na `self.routes`) -> `prompt_server.add_routes()`
  (linia ~556, `self.app.add_routes(self.routes)` - dopiero tu trasy stają
  się aktywne w aiohttp). Czyli rejestracja dekoratorem przy imporcie
  trafia do tablicy tras ZANIM `add_routes()` ją zamontuje.
- **Trigger rejestracji**: `__init__.py` importuje moduł z routes (w
  Prompt-Writer: `from .backend import routes as _routes  # noqa`). Sam
  import = rejestracja endpointów.
- **Kształt odpowiedzi**: `web.json_response(payload, status=...)`; błędy
  jako `web.json_response({"error": ...}, status=4xx)`. Bajty obrazu:
  `web.Response(body=<bytes>, content_type="image/jpeg")` (wzorzec z
  `server.py` `/view`, linia ~576).
- **Testy bez ComfyUI**: stub `sys.modules["server"]` z atrapą
  `PromptServer.instance.routes`, której `.get`/`.post` to pass-through
  dekoratory (`lambda fn: fn`) - handlery zostają zwykłymi funkcjami
  modułu i woła się je wprost z podrobionym `request` (`.query` dict,
  `async def json()`). Ten stub jest w `tests/conftest.py` (§23.4 planu).
  Wzorzec skopiowany z `MiniMaxH3-Prompt-Writer/tests/test_routes_stability.py`.
- Nasz `__init__.py` NIE używa importów relatywnych (patrz jego docstring),
  więc routes importujemy jako `import minimaxh3_clipcache.routes` w
  `try/except` - rejestracja opcjonalnego UI nie może wywalić ładowania
  node'a, więc błąd = tylko `logger.warning`, nie wyjątek (to nie jest
  "cichy fallback" ze ścieżki poprawności cache'a - to opcjonalny UI).

Co ZOSTAJE do ręcznej weryfikacji w Fazie 7 przy żywym ComfyUI (nie da
się tego sensownie sprawdzić unit testem):
- realny `Content-Type`/transfer bajtów miniaturki przez HTTP (endpoint
  `thumbnail` nie był jeszcze wywołany z istniejącym plikiem na żywo),
- brak kolizji z żadnym innym custom node rejestrującym podobny prefix.

### Faza 5/6 - co zweryfikowano na żywym serwerze (headless, curl)

Uruchomiony `python main.py --port 8199 --cpu` (osobny port, żeby nie
kolidować z sesją użytkownika; baza dała warning o locku, ale serwer
wstał). Sprawdzone `curl`-em, BEZ przeglądarki:
- nasz node ładuje się bez błędu, ZERO warningu
  "Cache Manager REST routes not registered" -> rejestracja dekoratorów
  `@routes.get/@routes.post` na `PromptServer.instance.routes` DZIAŁA na
  realnym serwerze (nie tylko na stubie),
- `GET /h3_cache_manager/check` -> HTTP 200, kształt JSON dokładnie
  `{entries:[{fingerprint,classification,verbose}], total_count,
  total_size_bytes}` (scanner znalazł 12 realnych wpisów legacy z
  wcześniejszych faz - `.safetensors`+`.json` bez `.verbose.json`,
  poprawnie sklasyfikowane jako "legacy"),
- `GET /h3_cache_manager/get?fingerprint=nope` -> 400,
  `?fingerprint=<realny 64-hex bez verbose>` -> 404,
- `POST /h3_cache_manager/update` z nieistniejącym fingerprintem -> 404,
- `GET /h3_cache_manager/thumbnail?...&index=notint` -> 400,
- `GET /extensions/ComfyUI-MiniMaxH3-CLIPCached/main.js` -> 200
  `text/javascript`, `.../styles.css` -> 200 `text/css`
  (`WEB_DIRECTORY="./web"` serwuje pliki poprawnie).

### Faza 6 część 1 - czego NIE sprawdzono (do oceny użytkownika w przeglądarce)

`web/main.js` + `web/styles.css` napisane (szkielet: floating launcher
"H3 Cache", modal z `role="dialog"`/`aria-modal`, toolbar z "Check" +
status, surowa lista wierszy `fingerprint[:12]…` + badge
normal/legacy, Escape zamyka, `app.registerExtension` z
commands/menuCommands/setup). Zweryfikowana tylko składnia JS
(`node --check` jako ES module - OK) i że pliki są serwowane po HTTP.
NIE uruchomiono w przeglądarce, więc NIE potwierdzone:
- czy `import { app } from "/scripts/app.js"` / `"/scripts/api.js"`
  faktycznie się rozwiązują w tej wersji frontendu ComfyUI,
- czy `app.registerExtension({commands, menuCommands, setup})` w tej
  wersji ma dokładnie takie API (skopiowane z Prompt-Writer, ale wersje
  frontendu mogą się różnić),
- czy panel się renderuje, czy launcher jest widoczny nad canvasem,
- czy klik "Check" w realnym UI odpala `api.fetchApi` i wyświetla wynik,
- cokolwiek dotyczące wyglądu/UX/z-index/kolizji stylów z ComfyUI.
To jest jawnie zostawione użytkownikowi do obejrzenia na żywo (Faza 6
z założenia wymaga człowieka przy przeglądarce).

### Faza 6 część 2 - co zweryfikowano (bez przeglądarki) i czego nie

`web/main.js` rozbudowany o: klientowy filtr (search / tag select /
favorites-only), miniatury referencji przez `api.fetchApi().blob()` +
`URL.createObjectURL` (z rewokacją przy każdym re-renderze i licznikiem
generacji przeciw wyścigom), inline panel szczegółów pod listą (pełny
prompt w `<pre>`, edycja name/notes/tags/favorite, Save -> POST /update
-> runCheck -> ponowne otwarcie tego samego wpisu). Load/Delete dalej
poza zakresem. Czyste funkcje (`filterEntries`, `formatBytes`,
`parseTags`, `entryLabel`, `allNormalTags`, `shortPrompt`) dostały
`export` żeby dało się je testować z Node bez DOM.

Zweryfikowane samodzielnie:
- **Node harness** (scratchpad, stub loaderem podmieniający
  `/scripts/app.js` i `/scripts/api.js`, minimalny `document`): moduł
  importuje się bez wyjątku, `app.registerExtension` dostaje poprawny
  kształt (name/commands/menuCommands), `setup()` nie rzuca. 24/24
  asercje na czystych funkcjach przechodzą - w tym reguła "legacy widoczny
  tylko przy zerowych filtrach" i wszystkie kombinacje search/tag/favorite.
  Harness NIE jest commitowany (to nie jest test JS w suite, tylko
  jednorazowa weryfikacja).
- **Żywy serwer** (`main.py --port 8199 --cpu`), `curl`, dokładnie te
  endpointy których używa nowy JS:
  - `GET /check` -> 200, poprawny kształt,
  - `POST /update` częściowy `{fingerprint, favorite:true}` -> 200,
    zmienia tylko `favorite`, `system` i pozostałe pola `user` nietknięte
    (to jest ścieżka gwiazdki w wierszu),
  - `POST /update` pełny `{fingerprint, name, notes, tags, favorite}` ->
    200 (ścieżka przycisku Save),
  - `GET /get?fingerprint=` potwierdza trwałość zapisu,
  - `main.js` / `styles.css` serwowane 200 z poprawnym Content-Type.
  Test na tymczasowym `.verbose.json` dla istniejącego fingerprintu,
  usuniętym po teście - realny cache użytkownika nietknięty.

NIE sprawdzone (wymaga przeglądarki, do oceny użytkownika): faktyczne
renderowanie listy/wierszy/chipów, ładowanie i wyświetlanie miniatur
jako `<img>`, otwieranie panelu szczegółów po kliknięciu wiersza,
edycja + Save + odświeżenie w realnym DOM, brak błędów JS w konsoli przy
tych akcjach, wygląd/UX/kolizje stylów.

Uboczne potwierdzenie: w trakcie tej sesji użytkownik w SWOJEJ instancji
ComfyUI zrobił realną generację cached-node'em -> powstał nowy wpis
cache `ad219594...` z `.json` + `.safetensors` + `.verbose.json`
(prawdziwy multi-shot prompt). Czyli `_sync_verbose_metadata` z Fazy 2
zadziałał end-to-end w produkcji, nie tylko w unit testach. `/check`
poprawnie pokazuje ten wpis jako "normal".

### Faza 6 część 3 - lokalna konwencja graph/widget API + weryfikacja

**STATUS: zapis do widgetu w grafie ZOSTAŁ WYCOFANY.** "Load" (teraz
"Copy prompt") kopiuje prompt do schowka, nie do noda - patrz
CACHE_MANAGER_PLAN.md sekcja 14 i "Faza 6 część 5" niżej. Powody
strukturalne: `app.graph.findNodesByType()` nie schodzi do subgrafów
(node w subgrafie niewidoczny), a prompt przekonwertowany z widgetu na
input nie ma widgetu w `node.widgets` do ustawienia. Poniższe ustalenia
o graph/widget API zostają jako zapis dochodzenia (i uzasadnienie
wycofania), ale `applyLoad()`/`loadIntoNode()`/`nodeOptionLabel()`/picker
NIE ISTNIEJĄ już w kodzie.

**Uwaga metodologiczna:** NIE mam tu dostępu do żywej konsoli DevTools w
przeglądarce. To co niżej jest zweryfikowane wobec ŹRÓDŁA frontendu
ComfyUI (pakiet pip `comfyui-frontend-package`, `static/assets/*.js` +
`*.js.map` z `sourcesContent`) oraz działającego przykładu tego samego
autora, NIE w interaktywnej konsoli.

Ustalenia (źródło w nawiasach):
- **`app`** = `window.comfyAPI.app.app`, **`api`** = `window.comfyAPI.api.api`
  (`static/scripts/app.js`, `api.js` - to tylko cienkie shimy re-eksportujące).
- **`app.graph.findNodesByType(type)`** zwraca **tablicę** node'ów, których
  `node.type` (case-insensitive) == `type`; iteruje po `graph._nodes`
  (litegraph, `settingStore-CwkLtSKP.js`: `findNodesByType(e,t){let
  n=e.toLowerCase();...for(let e of r)e.type?.toLowerCase()==n&&t.push(e)...}`).
- **`node.type` naszego node'a** = `"MiniMaxH3CLIPCachedImageToVideo"`
  (klucz `NODE_CLASS_MAPPINGS`; potwierdzone przez `GET
  /object_info/MiniMaxH3CLIPCachedImageToVideo` -> `"name":
  "MiniMaxH3CLIPCachedImageToVideo"`).
- **widget "prompt"** jest wieloliniowym STRING (`/object_info`:
  `['STRING', {'multiline': True, 'dynamicPrompts': True}]`). W tej wersji
  frontendu tworzony jako **DOM widget**: `node.addDOMWidget(name,
  'customtext', inputEl, {getValue, setValue})`
  (`useStringWidget.ts` w `settingStore-CwkLtSKP.js.map`). Jego setter
  `set value(v)` robi `inputEl.value = v` ORAZ aktualizuje reaktywny
  `widgetStore`, a bazowy `BaseDOMWidgetImpl.set value` dodatkowo woła
  `this.callback?.(this.value)` (`domWidget.ts`). Czyli **`widget.value =
  prompt` samo w sobie propaguje do widocznego textarea** i odpala
  callback - nie trzeba nic więcej. `widget.element` = textarea
  (`widget.inputEl` to deprecated alias, dotknięcie loguje ostrzeżenie -
  używamy `widget.element`).
- **`node.widgets`** to tablica; szukanie: `node.widgets.find(w =>
  w.name === "prompt")` (wzorzec w rgthree, kjnodes, MMH3Tools).
- **repaint po zmianie wartości**: `node.graph?.setDirtyCanvas(true, true)`
  (dokładnie ten wzorzec w `ComfyUI-MMH3Tools/web/js/mmh3_dimension_calculator.js`
  tego samego autora - ustawia `w.value` combo-widgetów, potem
  `node.graph?.setDirtyCanvas(true, true)`).

Zweryfikowane samodzielnie (dot. wersji z zapisem do widgetu - historyczne):
- **Żywy serwer** (`main.py --port 8199 --cpu`, `curl`): pełny cykl
  Delete - utworzono syntetyczny wpis (`deadbeef00...`, zmyślony fp,
  `.json` + `.safetensors` + `.verbose.json` + `thumbnails/<fp>_0.jpg`),
  `POST /h3_cache_manager/delete {fingerprint}` -> `200 {"deleted": ...}`,
  **wszystkie 4 pliki usunięte**, `/get` potem 404. Realny cache
  użytkownika nietknięty (zmyślony fp nigdy nie był realnym wpisem).
  `/delete` ze złym fp -> 400, `/check` dalej działa. **To dalej aktualne**
  (endpoint Delete się nie zmienił).

NIE sprawdzone (wymaga przeglądarki, do oceny użytkownika):
- `window.confirm` dla Delete, odświeżenie listy po Delete,
- miniatury referencji w komunikacie po Copy prompt, brak błędów JS w
  konsoli, wygląd/UX.

### Faza 6 część 4 - drobna poprawka Load result

W komunikacie po Load każda miniatura referencji jest teraz: klikalna
(`<a target="_blank" rel="noopener">`, `href` = ten sam blob object URL co
`img.src`, ustawiany dopiero po pobraniu - klik otwiera dokładnie ten plik
≤256px, bez udawania większej rozdzielczości), z podpisem wymiarów w px
(`${img.naturalWidth}×${img.naturalHeight}px`, uzupełniane w `img.onload`,
"—" dopóki się nie załaduje), większa (88px, `object-fit: contain` żeby
proporcje były widoczne). Pod listą referencji dodane zdanie: "This is the
only visual reference this cache entry has — the original image file is
never stored." Świadomie ZERO filename/ścieżki (plan 10.2/14). Zmiana
czysto klientowa (DOM), nic po stronie serwera. Zweryfikowane: `node
--check`, harness 32/32 (nowy check: `renderLoadResult` z niepustymi
`references` nie rzuca pod atrapą DOM). Reszta - do obejrzenia w
przeglądarce (jak wyżej).

### Faza 6 część 5 - Load -> Copy prompt (schowek) + ikonka copy

Zapis do widgetu w grafie wycofany (powody w części 3 wyżej i w
CACHE_MANAGER_PLAN.md sekcji 14). USUNIĘTE z `web/main.js`:
`loadIntoNode()`, `applyLoad()`, `nodeOptionLabel()`, `NODE_TYPE`,
`onLoadClick()`, `showLoadResultText()`, cała logika pickera węzłów
(`data-h3cm-target-node` / `data-h3cm-load-into-selected`) i `<div
data-h3cm-load-picker>` z szablonu; z `web/styles.css` reguły
`.h3cm-load-picker*`. Renaming `data-h3cm-load-*` -> `data-h3cm-copy-*`,
`.h3cm-load-result` -> `.h3cm-copy-result`, `renderLoadResult` ->
`renderCopyResult` (headline jako parametr), `loadResultObjectUrls` ->
`copyResultObjectUrls` itd.

Nowe: `copyPrompt()` - `navigator.clipboard.writeText(prompt)`, na sukces
`renderCopyResult(..., "Copied to clipboard.")`, na wyjątek (np.
niesecure context) `renderCopyResult(..., "Couldn't copy automatically -
select the prompt above and copy it manually.")`. Info o referencjach +
miniatury + zdanie "Load these images manually..." zostają bez zmian
(nadal prawdziwe - prompt w schowku, obrazy nie).

Ikonka copy w lewym-górnym rogu `<pre>` z promptem (`data-h3cm-prompt-copy`,
`.h3cm-prompt-copy`, wrap `.h3cm-prompt-wrap`): `copyPromptText(button)` -
kopiuje SAM tekst promptu, feedback przez klasę `.is-copied` (kolor
`#7edeb3` - istniejąca "pozytywna" zieleń z `.h3cm-badge-normal`, nie
nowa) + `title` "Copied!" na 1.5s, na błąd `title` "Copy failed - select
the text manually". `.h3cm-prompt` dostał `padding-top: 32px` żeby ikonka
nie nachodziła na pierwszą linię.

Zweryfikowane: `node --check`, harness 22/22 (moduł ładuje się czysto,
usunięte symbole `loadIntoNode`/`applyLoad`/`nodeOptionLabel` faktycznie
`undefined`, czyste funkcje + filtry dalej OK, `setup()` nie rzuca). Do
obejrzenia w przeglądarce: faktyczne kopiowanie do schowka (i fallback
gdy `navigator.clipboard` niedostępne), feedback ikonki, wygląd.
