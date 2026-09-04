# ComfyUI-MiniMaxH3-CLIPCached — kontekst projektu

## Cel
Custom node ComfyUI: "MiniMax H3 CLIP-Cached FL2VA". Cache'uje wynik
text/vision-encodingu (Qwen3-VL, przez natywny obiekt CLIP ComfyUI z
clip_type=MINIMAX) na dysku, żeby przy powtórzonym prompt+first_frame+
last_frame nie trzeba było w ogóle ładować ~27 GB encodera.

## Zasada nadrzędna
Nie tworzymy własnej wersji MiniMaxH3ImageToVideo. Tworzymy przezroczysty
cached-CLIP proxy i pozwalamy stockowemu node'owi (comfy_extras/nodes_minimax_h3.py)
wykonać całą właściwą mechanikę H3 (resize, VAE encode keyframes, AV latent,
minimax_keyframes). Nigdy nie kopiujemy ani nie reimplementujemy tej logiki.

## Potwierdzone fakty o środowisku (nie zakładać nic ponad to bez ponownej weryfikacji)
- ComfyUI lokalnie: v0.34.2 (aktualizowane w trakcie prac nad Ref2Video,
  patrz notatka R1 o length), w katalogu instalacji ComfyUI (ten repo to
  jego custom_node). Wersję sprawdzać przez comfyui_version.py /
  pyproject.toml (oba: "0.34.2") oraz git describe.
- Wersja ComfyUI w RUNTIME: `from comfyui_version import __version__`
  (albo `import comfyui_version; comfyui_version.__version__`). To jest ten
  sam sposób, którego używa sam ComfyUI - server.py robi
  `from comfyui_version import __version__` (linia 44) i zwraca to pole jako
  "comfyui_version" w /system_stats (linia ~724), a main.py loguje je przy
  starcie. Plik comfyui_version.py leży w korzeniu repo ComfyUI i jest
  generowany automatycznie z pyproject.toml przez build. Moduł jest lekki
  (sam string), bezpieczny do importu w node bez uruchamiania serwera.
- UWAGA: użytkownik utrzymuje lokalne monkey-patche na czysty ComfyUI
  (git stash "MiniMax H3 local monkey patches before master update" w repo
  ComfyUI, NIE w tym repo). Łatki dotyczą: (a) widgetu
  length w nodes_minimax_h3.py (min=1/max=3600/step=1 zamiast stockowego
  min=5/max=3600/step=17, we wszystkich 3 node'ach: EmptyMiniMaxH3LatentAV,
  MiniMaxH3ImageToVideo, MiniMaxH3ReferenceToVideo), plus zabezpieczenia
  n<=1 w align_frame_count()/video_latent_t() i max(1, length) zamiast
  max(5, length) w temporal_shape(); (b) zakomentowania v = v.clone() w
  comfy/ldm/minimax/model.py (Attention, perf/VRAM). Ten projekt celuje w
  CZYSTY upstream, nie w załatane środowisko - jeśli coś kiedyś znowu
  wygląda niespójne ze stockiem, sprawdź NAJPIERW czy lokalne ComfyUI ma
  nałożone łatki (git status / git stash list w repo ComfyUI, nie w tym
  repo), zanim uznasz to za bug w naszym kodzie.
- Łatka na v.clone() w comfy/ldm/minimax/model.py jest OBECNIE nałożona
  (git status w repo ComfyUI pokazuje "M" na tym pliku - stan po
  częściowym re-aplikowaniu po aktualizacji do v0.34.2; łatka length
  została w stashu i NIE jest nałożona). Dotyczy realnej ścieżki
  obliczeniowej modelu (forward Attention), nie tylko UI węzła. Poza
  zakresem tego projektu, ale warto wiedzieć że tam jest - może wpływać
  na czasy/VRAM w benchmarkach.
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
  na interpreter base env miniconda (Python 3.14.6, BEZ torch —
  `import torch` rzuca ModuleNotFoundError). ~/.bashrc ma tylko
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
- tests/conftest.py dodaje korzeń ComfyUI (COMFYUI_ROOT, domyślnie
  wyliczany z układu katalogów) do sys.path, co pozwala
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
  (clip_name, file_size, mtime_ns, ctime_ns) z os.stat().
- Logowanie: jasno wypisuj, którą ścieżką poszło wykonanie (HIT/MISS),
  zajętość RAM/VRAM przed i po unload.
- Żadnych `git add .` — jawnie staged pliki. Nie commitować: cache/,
  __pycache__/, plików tymczasowych.
- Nie zgaduj API ComfyUI z pamięci/GitHuba — sprawdzaj lokalnie w repo
  ComfyUI (katalog instalacji, ten repo to jego custom_node), bo lokalna
  wersja może się różnić.
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
10. Identyfikacja encodera: clip_name + file_size + mtime_ns + ctime_ns
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

## Ref2Video (R1-R10) - status: WSZYSTKIE ZAKOŃCZONE

Drugi node w tym repo, MiniMaxH3CLIPCachedRef2VA, wrapuje stockowy
comfy_extras.nodes_minimax_h3.MiniMaxH3ReferenceToVideo tą samą zasadą
nadrzędną co FL2VA (proxy nad clip, zero reimplementacji mechaniki H3).
Fazy poniżej to skrócone podsumowanie, analogiczne do stylu faz 1-25 -
pełna historia jest w historii commitów (git log), nie tutaj.

R1: kontrakt MiniMaxH3ReferenceToVideo zweryfikowany lokalnie (ref_items
    jako płaska lista heterogenicznych dictów, kolejność semantycznie
    znacząca, audio-przed-swoim-wideo, tylko tokenize/encode_from_tokens_scheduled
    na clip)
R2: fingerprint stres-testowany na syntetycznym ref_items (heterogeniczna
    lista, markery audio, kolejność, znaczniki czasowe) - PASS
R3: bramka go/no-go z SpyClipProxy podstawionym przed stockowy
    MiniMaxH3ReferenceToVideo - PASS
R4: równoważność stock/proxy na realnym enkoderze (image+video+audio) -
    torch.equal, 12/12 pól PASS
R5-R6: MiniMaxH3CLIPCachedRef2VA dodany - v1, 18 stałych opcjonalnych
    slotów (9 obrazów, 3 wideo, 3 ich ścieżki dźwiękowe, 3 samodzielne
    audio) zamiast stockowego io.Autogrow, builder slot->dict przetestowany
    w izolacji
R7: end-to-end przez żywy serwer (dwa świeże serwery, ten sam graf) -
    real MISS i real HIT potwierdzone w logach (/tmp/r7_server.log,
    /tmp/r7_hit_server.log)
R8: testy inwalidacji (zmiana treści/rozmiaru referencji, zamiana
    kolejności slotów) - w trakcie prac poprawiona błędna hipoteza:
    indeks konkretnego slotu jest nieistotny dla fingerprintu, liczy się
    wyłącznie względna kolejność niepustych referencji
R9: test trendu pamięci na żywym serwerze przy wielu kolejnych realnych
    MISS - trend płaski, spójny z wynikiem dla FL2VA z fazy 24
R10: README rozszerzone o sekcję Ref2VA obok FL2VA (zrzuty ekranu
    porównawcze RAM/VRAM wciąż zaległe - patrz sekcja "R10 prep" poniżej)

## R10 prep: zrzuty ekranu do README (TODO dla użytkownika, nie dla CC)

Przed napisaniem sekcji README dla Ref2VA - zrzuty ekranu porównawcze
zajęcia RAM/VRAM (nvitop/nvidia-smi na żywo):

- BEZ cache'a: pełny real encode przez stockowy node/nasz node przy
  MISS - pokazać moment ładowania ~26GB Qwen3-VL encodera (linia
  "Requested to load MiniMaxH3TEModel_" w logu serwera) + peak VRAM/RAM
  w tym momencie.
- Z cache'em (HIT): identyczny graf, drugi przebieg - pokazać że ta
  linia w logu w ogóle się nie pojawia, VRAM/RAM zostaje płaskie.

Gotowy scenariusz do powtórzenia: dokładnie ten sam setup co R7 (dwa
świeże serwery, ten sam graf Ref2VA) - logi z R7 już to demonstrują
tekstowo (/tmp/r7_server.log dla MISS, /tmp/r7_hit_server.log dla HIT),
brakuje tylko wizualnego zrzutu ekranu z monitora GPU/RAM w trakcie.

To jest task DLA UŻYTKOWNIKA (wymaga ręcznego zrobienia screenshotów przy
odpalonym nvitop), nie coś do zautomatyzowania w kodzie/testach.

## Cache Manager (drugi wątek)

Pełny opis w docs/CACHE_MANAGER.md. Kluczowy niezmiennik: cache jest
source of truth, manager jest warstwą indeksującą, fingerprint = ID
wpisu, prompt jest read-only w managerze.

### Faza M2: przełącznik FL2VA/Ref2VA w web/main.js

Dodane w `web/main.js` + `web/styles.css` (Python bez zmian, pełny pytest
172 passed przed i po):
- moduł-level `currentVariant` ("fl2va"|"ref2va"), dwa przyciski
  `data-h3cm-variant` w toolbarze, `switchVariant()` - klient-side, BEZ
  nowego fetcha do /check, resetuje search/tag/favorites i zamyka detail
  przy przełączeniu (świadoma decyzja: brak osobnego stanu filtrów per
  zakładka)
- `filterEntries()` czyta `currentVariant` (moduł-level, nie argument) i
  odcina wpisy po `verbose.system.node_variant || "fl2va"` PRZED
  pozostałymi warunkami; legacy i wpisy bez verbose = "fl2va"
- nagłówek "Cache: N entries / size" NIEzmieniony - dalej całościowy z
  lastCheckResult, nie przeliczany po wariancie
- wiersz listy Ref2VA: pierwsze 3 referencje (miniatura dla image/video,
  pill "audio" dla audio bez fetcha thumbnaila), "+N more" dla reszty
- panel szczegółów Ref2VA (`renderDetailRefs()`): WSZYSTKIE referencje z
  etykietą pozycyjną liczoną per typ w locie w JS ("Picture 1/2…",
  "Video 1…", "Audio 1…"), NIGDY z zapisanego pola; FL2VA detail bez
  zmian (nowy kontener `data-h3cm-detail-refs` zostaje `hidden`)
- WSZĘDZIE gdzie JS czyta `reference.type` / `node_variant` jest fallback
  `|| "image"` / `|| "fl2va"` - działa też na cache sprzed migracji
  (migrate_verbose_schema_v2.py jeszcze NIE odpalony na realnym cache/)

Zweryfikowane BEZ przeglądarki:
- `node --check` (kopia .mjs): składnia OK dla main.js
- brace-balance styles.css: OK
- harness Node w scratchpadzie (loader hook stubuje /scripts/app.js i
  /scripts/api.js, minimalny document): moduł importuje się bez wyjątku
  na top-level; `filterEntries` z domyślnym `currentVariant="fl2va"`
  przepuszcza tylko fl2va + legacy + wpisy bez verbose, odcina ref2va
- osobny mikro-test algorytmu liczników pozycyjnych: kolejność
  image/video/audio -> "Picture 1","Picture 2","Audio 1","Video 1",… ,
  wpis bez `type` -> "Picture N" (fallback), zgodne z oczekiwaniem

CZEKA na wizualne sprawdzenie użytkownika w ComfyUI (CC nie może tego
zaliczyć):
- czy oba przyciski renderują się w toolbarze i `is-active` wygląda OK
- realne przełączenie na "Ref2VA" i z powrotem (wymaga interakcji DOM -
  harness nie może ustawić `currentVariant`), czy lista faktycznie
  pokazuje wpisy ref2va i chowa fl2va
- reset filtrów i zamknięcie detala przy przełączeniu
- wygląd wiersza Ref2VA: 3 miniatury + pill "audio" + "+N more"
- panel szczegółów Ref2VA: miniatury + etykiety pozycyjne, brak sekcji
  refs dla FL2VA
- brak błędów w konsoli przeglądarki
- ZNANY BRAK do M3 (świadomie nietknięty): "Copy prompt" / renderCopyResult
  wciąż używa `ref.label` i mówi "image references" - dla wpisu ref2va
  pokaże "- undefined" i mylący tekst. Naprawa w Fazie M3.

### Domknięcie

Integracja Ref2VA z Cache Managerem (Fazy M1-M5) zakończona i
zweryfikowana end-to-end na żywym ComfyUI+GPU (unifikacja schematu
references z polami type/node_variant, jednorazowy skrypt migracyjny
odpalony i usunięty, wspólne hooki Managera dla FL2VA+Ref2VA, przełącznik
FL2VA/Ref2VA w UI, rekonstrukcja kolejności Picture N/Video N/Audio N
potwierdzona na żywych danych, generalizacja Copy prompt pod N
heterogenicznych referencji). master promowany z feature/ref2video
(55f0f9a) 29.08.2026 - poprzedni stan czysto-FL2VA zamrożony jako branch
FL2V_master (613d234).

### Sesja porządkowa (audyty Grok/GPT/CodeRabbit), 29.08.2026

Osiem grup drobnych poprawek, każda osobny commit (afa1705..ad942e2),
pełny pytest zielony po każdej grupie (197 passed na końcu). Zakres m.in.:
maskowanie wyjątku przez nieudany unload w finally (proxy.py + oba
execute() w nodes.py), obsługa 1- i 4-kanałowych tensorów IMAGE w
thumbnails, sprzątanie plików .tmp-* po nieudanym zapisie w
verbose_store/thumbnails, timeouty w testach współbieżności, poprawki
None-formatting w skryptach diagnostycznych.

Zmiany w web/main.js (Grupa 3: dropdown tagów filtruje po
currentVariant; Grupa 6: wpis "normal" z uszkodzonym verbose.json
renderuje się jak legacy) zweryfikowane BEZ przeglądarki zgodnie ze
stałym workflow: node --check na kopii .mjs (składnia OK) + harness
Node w scratchpadzie (loader hook stubuje /scripts/app.js i
/scripts/api.js, wymusza format module na main.js; minimalny document)
- moduł importuje się bez wyjątku, allNormalTags(entries, variant)
zwraca tagi tylko z danego wariantu i [] gdy wariant pominięty,
filterEntries nie wywala się na verbose=null i traktuje taki wpis jak
legacy (widoczny tylko bez filtrów). NIE zweryfikowane (do sprawdzenia
przez użytkownika w ComfyUI): realny render wiersza uszkodzonego wpisu
przez buildLegacyRow, faktyczne przełączenie wariantu w dropdownie tagów
na żywym DOM, brak błędów w konsoli.

### Nazwy plików referencji: Ref2VA zrealizowane, FL2VA nadal odłożone

Rozważone i ODŁOŻONE (nie odrzucone na stałe): śledzenie nazwy pliku
źródłowego dla referencji first_frame/last_frame. Prosta wersja (ukryte
inputy PROMPT/EXTRA_PNGINFO w węźle H3-cached, prześledzenie wstecz przez
graf czy bezpośrednim źródłem jest LoadImage) byłaby krucha i sięgałaby
poza własny kontrakt węzła - ten sam powód co analogiczna decyzja przy
Ref2VA (MANAGER_TODO_ref2video.md punkt 10: węzeł ma zostać zgodny z
oryginalnym kontraktem stockowego węzła, żadnych dodatkowych pól).
MANAGER_TODO_ref2video.md był dokumentem roboczym z etapu planowania i
nigdy nie trafił do tego repozytorium (git log --all po tej ścieżce jest
pusty) - odwołanie zostaje jako ślad ówczesnego uzasadnienia, nie jako
wskazówka gdzie szukać tego pliku.

Poprawna implementacja wymagałaby zamiast tego osobnych, dedykowanych
wrapperów na węzły ładujące (LoadImage/LoadVideo/LoadAudio i pochodne),
analogicznych do MiniMaxH3CLIPCached* - takie węzły przekazywałyby nazwę
pliku jawnie, jako część własnego kontraktu wyjścia, zamiast węzeł
H3-cached zgadywał to z grafu przez introspekcję. To realny, ale znacznie
większy projekt niż drobna poprawka (osobna rodzina wrapperów, nie jedna
linijka) - może zostać podjęty kiedyś, jeśli okaże się wystarczająco
wartościowy, nie jest odrzucony na stałe.

Wariant Ref2VA ZOSTAŁ JEDNAK ZREALIZOWANY - inną drogą niż zakładał
akapit powyżej: bez rodziny wrapperów, dokładnie tą introspekcją grafu,
która została wyżej odrzucona. minimaxh3_clipcache/provenance.py chodzi
po API-formatowym grafie wstecz od każdego slotu ref_* do węzła-liścia
(prawdziwej ładowarki) i odczytuje jego literalną nazwę pliku;
_sync_ref_sources() w nodes.py dopisuje wynik jako system.ref_sources w
sidecarze verbose, a Cache Manager pokazuje go pod każdą referencją w
panelu szczegółów.

Pierwotne zastrzeżenie nie zostało zignorowane - obeszło je zawężenie
zakresu na tyle mocne, że waga tego zastrzeżenia spadła. Nic z
prowenancji nie wchodzi do compute_fingerprint(), nieudany przejazd po
grafie nie jest w stanie naruszyć zapisanego encode'u
(collect_ref_sources() nigdy nie rzuca, a wynik None oznacza "zostaw
to, co już jest"), całość jest pomocą nawigacyjną dla UI Managera, nie
częścią kontraktu cache'a. Wyjście "poza kontrakt węzła" ograniczyło się
do dwóch ukrytych inputów (PROMPT/UNIQUE_ID) na parze węzłów Ref2VA -
publiczny kontrakt wejść/wyjść względem stockowego węzła się nie zmienił.

Granice tego zakresu są opisane autorytatywnie w docstringu modułu
minimaxh3_clipcache/provenance.py (reguła liścia, dlaczego wynik jest
kluczowany nazwą slotu a nie pozycją, rozróżnienie None vs {}) - czytać
stamtąd, nie powielać ich tutaj.

Nadal ODŁOŻONE i objęte akapitami powyżej: first_frame/last_frame w
FL2VA. Węzły FL2VA nie deklarują bloku "hidden", więc w ogóle nie
dostają grafu do przejścia - szczegóły w TODO.md.

### comfyui_version w verbose metadata (informacyjnie)

_sync_verbose_metadata() w nodes.py zapisuje pole "comfyui_version" w
bloku "system" sidecara (best-effort, try/except - nigdy nie wywala
funkcji). WYŁĄCZNIE informacyjne: nie wchodzi do compute_fingerprint(),
nie wpływa na HIT/MISS. Ma pomóc przy diagnozie "dlaczego stary wpis w
cache wygląda inaczej" po aktualizacji ComfyUI.

comfyui_version w verbose metadata to tani, informacyjny kompromis. Węższy,
ale realnie działający wariant "encoding ABI fingerprint" (wchodzący do
HIT/MISS, w przeciwieństwie do pola informacyjnego wyżej) został dodany -
patrz sekcja niżej.

### Encoder ABI fingerprint (audit punkt 1) - dodany 2026-08-29

Nowy moduł minimaxh3_clipcache/encoder_abi.py: get_encoder_abi_id() zwraca
(abi_id, available), gdzie abi_id = "{comfyui_version}:{sha256 pliku
comfy/text_encoders/minimax.py}", liczone i cache'owane RAZ na proces
(plik nie zmienia się w trakcie działania ComfyUI; cache'owany jest też
wynik porażki, a WARNING loguje się raz na sesję). compute_fingerprint()
dostał wymagany, keyword-only parametr encoder_abi_id (bez wartości
domyślnej - nie da się go po cichu pominąć); CachedClipProxy ma go z
testowym defaultem "test-abi-id" (nigdy nie trafia na produkcję - nodes.py
zawsze podaje jawną wartość). Gdy ABI jest niedostępne (available=False),
oba węzły wymuszają realny encode niezależnie od cache_mode
(force_refresh=True, encoder_abi_id="unavailable") i IS_CHANGED zwraca
NaN - nigdy nie serwujemy ani nie zapisujemy HIT-a policzonego pod
niezweryfikowaną implementacją tokenizera (np. po zmianie w rodzaju
PR #15808 "Minimax-H3: Add missing special tokens").

Świadomy efekt uboczny: JEDNORAZOWA inwalidacja całego istniejącego
cache'a przy tym wdrożeniu - każdy dotychczasowy wpis przestaje być
trafiany, niezależnie od realnej wartości ABI, bo sam kształt danych
wchodzących w fingerprint się zmienił. To nie bug, to oczekiwany koszt
jednorazowy.

Zakres świadomie zawężony do comfyui_version + hash
comfy/text_encoders/minimax.py - NIE obejmuje łańcucha zależności
(qwen3vl.py -> qwen_vl.py -> qwen35.py -> llama.py -> sd1_clip.py, których
minimax.py używa). Zmiana WYŁĄCZNIE w jednym z tych współdzielonych plików,
nietykająca minimax.py, nie zostanie wykryta - zaakceptowany residual risk
w zamian za brak budowania pełnego trackera zależności ComfyUI i brak
inwalidacji cache'a przy każdej niezwiązanej zmianie upstream.

### Generation-ID w store.py (audyt punkt 4) + CACHE_SCHEMA_VERSION=2 - dodane 2026-08-29

Każdy wpis cache'a dostaje generation_id (uuid4 hex) generowany raz na
wywołanie save_conditioning() i zapisywany do OBU artefaktów: do payloadu
JSON ({"generation_id": ..., "skeleton": ...} zamiast gołego skeletonu)
oraz do metadanych .safetensors ("cache_generation_id"). load_conditioning()
porównuje oba - czytając tylko nagłówek safetensors przez safe_open(), przed
załadowaniem jakichkolwiek tensorów - i każde niedopasowanie traktuje jako
jednoznaczny MISS. To łapie rozerwaną parę po nieudanym drugim os.replace()
(nowy .safetensors pod starym .json) nawet gdy oba skeletony są
strukturalnie identyczne (przy refreshu tych samych wejść prawie zawsze są,
więc samo unflatten_tensors() by tego nie zauważyło). CACHE_SCHEMA_VERSION
1 -> 2, bo format na dysku faktycznie się zmienia (w odróżnieniu od encoder
ABI, gdzie format się NIE zmienił i stąd osobny mechanizm zamiast bumpa).

Połączone z niedawną inwalidacją przez encoder ABI - jedna fala odbudowy
cache'a zamiast dwóch osobnych.

gc_orphaned_cache_files() nadal automatycznie sprząta tylko .safetensors bez
.json. Para z niedopasowanym generation_id (oba pliki obecne, różne
generacje) jest teraz jawnie klasyfikowana przez Cache Manager jako
`inconsistent` i może zostać usunięta z UI; load_conditioning() nadal
odrzuca ją jako MISS i kolejne udane użycie tego fingerprintu ją nadpisuje.

### Floating launcher: przeciąganie + trzyrzędowa etykieta + styl - dodane 2026-08-30

installLauncher() w web/main.js dostał drag-to-move na pointer events
(pointerdown/move/up/cancel - jedno API dla myszy, dotyku i pióra).
Próg klik-vs-drag to Math.hypot(dx, dy) >= 5 px liczone od pointerdown;
poniżej progu gest jest zwykłym kliknięciem i dalej otwiera panel,
powyżej - przeciąganiem, a wynikowy syntetyczny `click` jest tłumiony
(flaga suppressClick, zerowana przez setTimeout(0) zaraz po ustawieniu,
żeby nie połknąć żadnej późniejszej niezwiązanej aktywacji, oraz przez
każdy kolejny pointerdown). Przy pierwszym realnym ruchu pozycjonowanie
przełącza się z CSS-owego right/bottom na inline left/top (pinToLeftTop,
ustawia right/bottom na "auto"). Pozycja jest clampowana do
window.innerWidth/innerHeight przy każdym pointermove i przy evencie
resize (clampLauncherPosition, eksportowana - jak reszta czystych
helperów w tym pliku - razem z readLauncherPosition/writeLauncherPosition).

Persystencja: localStorage, klucz "h3cm-launcher-position", wartość
{left, top} w px zapisywana na pointerup kończącym drag. Odczyt w
installLauncher(): pozycja jest przywracana TYLKO gdy po clampie do
bieżącego viewportu wychodzi identyczna jak zapisana (czyli w całości się
mieści); brak klucza, zły JSON, nie-skończone liczby albo niedostępny
localStorage (tryb prywatny) są łykane po cichu - zostajemy przy domyślnym
rogu right:20/bottom:20, żaden wyjątek nie przerywa setup().

Etykieta: `textContent = "H3 Cache"` zamienione na trzy `<span
class="h3cm-launcher-line">` ("H3" / "Prompt" / "Cache"); button jest
teraz `display:flex; flex-direction:column`. title zmieniony na "Open
MiniMax H3 Prompt Cache Manager". NIE ruszane: h3cm-title w modalu,
menuCommands ["Extensions", "MiniMax H3 Cache Manager"], command label.

Styl (blok .h3cm-floating-launcher w web/styles.css): subtelny
liniowy gradient w obecnej ciemnej palecie (#2f333e -> #22252d),
border-radius 14px, transition na transform/background/box-shadow, hover
z transform: scale(1.04) i jaśniejszym gradientem, klasa .is-dragging
(cursor: grabbing, lekkie scale, mocniejszy cień), touch-action: none +
user-select: none na buttonie. Bez żadnych zewnętrznych grafik/ikon.

Zweryfikowane BEZ przeglądarki (stały workflow): `node --check` na kopii
.mjs (składnia OK), brace-balance styles.css, oraz harness Node w
scratchpadzie (loader hook wymusza format module na web/main.js i stubuje
/scripts/app.js -> rejestruje ext z setup(), /scripts/api.js; fake DOM z
localStorage) - 20 asercji: import bez wyjątku; clamp (in-bounds bez
zmian, ujemne -> 0, cap do viewportu, element większy niż okno -> 0);
read (brak klucza / zły JSON / zły kształt / nie-skończone -> null bez
rzutu); write+read round-trip; write łyka błąd storage; etykieta = trzy
rzędy H3/Prompt/Cache + nowy title; start bez inline left/top; drag ponad
próg rusza button i przełącza na left/top + klasa is-dragging; pointerup
czyści klasę i zapisuje sclampowaną pozycję; syntetyczny click po dragu
NIE otwiera panelu; klik poniżej progu otwiera panel (createPanel dokleja
h3cm-root do body); resize wciąga button z powrotem do okna; restore
honoruje zapisaną pozycję która się mieści i ignoruje tę która nie.

NIE zweryfikowane (do sprawdzenia przez użytkownika w żywym ComfyUI):
realny render trzech rzędów i wygląd gradientu/hover/scale, faktyczne
złapanie i przeciągnięcie przycisku myszą/dotykiem, że zwykły klik nadal
otwiera panel, przetrwanie pozycji przez odświeżenie strony, zachowanie
przy zmniejszeniu okna, brak błędów w konsoli. Znane ograniczenie: przy
pierwszym paint stylesheet (`<link>` z injectStyles) może jeszcze nie być
zaaplikowany, więc offsetWidth/Height użyte w clampie na ścieżce restore
mogą chwilowo odbiegać - kolejny resize/drag mierzy już poprawnie.
