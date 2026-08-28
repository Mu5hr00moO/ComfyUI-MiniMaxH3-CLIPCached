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
  patrz notatka R1 o length), w /home/kamil/ComfyUI. Wersję sprawdzać przez
  comfyui_version.py / pyproject.toml (oba: "0.34.2") oraz git describe.
- UWAGA: użytkownik utrzymuje lokalne monkey-patche na czysty ComfyUI
  (git stash "MiniMax H3 local monkey patches before master update" w repo
  ComfyUI /home/kamil/ComfyUI, NIE w tym repo). Łatki dotyczą: (a) widgetu
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

Pełny plan w CACHE_MANAGER_PLAN.md. Kluczowy niezmiennik: cache jest
source of truth, manager jest warstwą indeksującą, fingerprint = ID
wpisu, prompt jest read-only w managerze.
