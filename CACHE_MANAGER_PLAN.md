# MiniMax H3 CLIP Cache Manager — plan wdrożenia (v2, po review kodu)

Rozszerzenie istniejącego `MiniMaxH3CLIPCachedImageToVideo` o czytelne metadata
cache, miniaturki referencji i interaktywny manager.

Dokument zastępuje wcześniejszy plan „Biblioteka nazwanych promptów + Manager +
Prewarming cache”. Aktualna architektura jest prostsza:

> **Encode-cache jest źródłem prawdy. Manager jest warstwą indeksującą,
> opisującą i zarządzającą istniejącymi wpisami cache.**

Nie istnieje osobna biblioteka presetów niezależna od cache.

---

## 0. Status weryfikacji wobec kodu (v2)

Ten dokument to wersja 1 planu (bez zmian architektonicznych) plus sekcja
**23**, dopisana po realnym przejrzeniu repo `ComfyUI-MiniMaxH3-CLIPCached`
(`fingerprint.py`, `proxy.py`, `store.py`, `nodes.py`, `loader.py`,
`serialize.py`, `__init__.py`) na commicie `origin/master`:

```text
bb97275  Validate encoded conditioning hidden dim (guard against wrong clip_name)
```

**Przed branchowaniem zweryfikuj lokalnie**, że to ten sam commit
(`git rev-parse HEAD` na `master`) — zgodnie z §22: nic tu nie jest oparte na
pamięci ani wyłącznie na publicznym GitHubie bez lokalnego potwierdzenia.

Wynik review: **plan jest dobrze osadzony w kodzie**, żadna decyzja
architektoniczna z sekcji 1–22 nie wymaga zmiany. Domknięte zostały tylko
trzy konkretne luki techniczne i jedna jawna decyzja projektowa, których plan
nie precyzował — patrz sekcja **23**. W treści sekcji 1–22 dodano wyłącznie
krótkie odnośniki „(patrz §23.x)” w miejscach, których to dotyczy; sama treść
merytoryczna tych sekcji jest niezmieniona.

Praca toczy się na branchu `feature/cache-manager`, wydzielonym z `master`
na powyższym commicie.

---

## 1. Kontekst i cel

Punkt wyjścia: działający disk-cache wyników
`clip.encode_from_tokens_scheduled()` dla `MiniMaxH3ImageToVideo` (FLF),
zbudowany wokół:

- `CachedClipProxy`,
- `compute_fingerprint()`,
- istniejącego store `.safetensors` + `.json`,
- stockowego `MiniMaxH3ImageToVideo.execute()` jako źródła prawdy dla
  preprocessingu i tokenizacji.

Problemy do rozwiązania:

1. Cache jest skuteczny technicznie, ale mało czytelny dla użytkownika.
2. Trudno znaleźć wcześniejszy conditioning po treści promptu.
3. Nie wiadomo z poziomu UI, czy wpis został zbudowany z `first_frame`,
   `last_frame`, czy bez referencji.
4. Brakuje nazw, tagów, ulubionych wpisów i notatek.
5. Brakuje wygodnego sposobu sprawdzenia aktualnego stanu cache i usunięcia
   konkretnego wpisu.

Prewarming **nie jest częścią projektu**. Realne testy pokazały, że MISS jest
wystarczająco szybki, a po udanym encode CLIP jest zwalniany; normalne użycie
cached noda samo naturalnie przygotowuje cache.

---

## 2. Główny invariant

Każdy wpis managera odpowiada dokładnie jednemu technicznemu wpisowi cache:

```text
prompt
+ tokenize kwargs / obrazy referencyjne
+ encoder identity
+ cache schema
        ↓
   fingerprint
        ↓
<fingerprint>.json
<fingerprint>.safetensors
```

Dlatego:

```text
fingerprint = ID wpisu managera
```

Nie dodajemy:

- osobnego `uuid4`,
- `library_id`,
- `linked_fingerprints`,
- niezależnej bazy presetów.

Jeżeli prompt, referencje albo encoder powodują inny fingerprint, powstaje
inny wpis cache i tym samym inna pozycja managera.

Potwierdzone w kodzie: `fingerprint.py:54-89`, `compute_fingerprint()` bierze
wyłącznie `(prompt, tokenize_kwargs, clip_name, clip_file_size, clip_mtime_ns,
cache_schema_version)` — żadnego `uuid4` ani osobnej tożsamości nigdzie w
repo.

---

## 3. Cache jako source of truth

Normalna pozycja managera istnieje tylko wtedy, gdy istnieje odpowiadający jej
właściwy encode-cache.

Nie ma przepływu:

```text
zapisz preset
→ może kiedyś powstanie cache
```

Jest:

```text
RUN
→ HIT lub MISS
→ istniejący / nowo zapisany cache
→ verbose metadata
→ Manager / Check
```

Konsekwencje:

- nie ma przycisku „Zapisz jako…” przed wykonaniem grafu,
- prompt w managerze jest opisem realnego conditioning cache,
- manager nie może zmieniać promptu „w miejscu”,
- zmiana promptu odbywa się w normalnym widgetcie noda; kolejne wykonanie
  tworzy odpowiedni nowy fingerprint/cache entry.

---

## 4. Pliki jednego wpisu

Proponowany układ:

```text
cache/
├── <fingerprint>.safetensors
├── <fingerprint>.json
├── <fingerprint>.verbose.json
└── thumbnails/
    ├── <fingerprint>_0.jpg
    └── <fingerprint>_1.jpg
```

Znaczenie:

```text
<fingerprint>.safetensors
<fingerprint>.json
= właściwy encode-cache
= krytyczna ścieżka HIT/MISS

<fingerprint>.verbose.json
thumbnails/*
= metadata i UI managera
= warstwa pomocnicza
```

Uszkodzenie lub brak `verbose` nie może uszkodzić poprawnego conditioning cache
ani zmienić istniejącej polityki HIT/MISS.

Potwierdzone w kodzie: `nodes.py:23` — `CACHE_DIR = os.path.join(REPO_ROOT,
"cache")`, płaski katalog, dokładnie ten układ. `thumbnails/` jest nowym
podkatalogiem do dodania w Fazie 3.

---

## 5. `verbose` metadata

`verbose` przechowuje dwa logicznie różne zestawy danych:

1. **system** — opis tego, co faktycznie zostało zcache'owane,
2. **user** — dane organizacyjne edytowane przez użytkownika.

Przykład:

```json
{
  "fingerprint": "abc123...",
  "system": {
    "prompt": "...",
    "clip_name": "qwen3vl_...",
    "clip_file_size": 27141342152,
    "clip_mtime_ns": 123456789,
    "cache_schema_version": 1,
    "comfyui_version": "informacyjnie, jeśli łatwo dostępne",
    "references": [
      {
        "index": 0,
        "label": "first_frame",
        "thumbnail": "thumbnails/abc123_0.jpg"
      },
      {
        "index": 1,
        "label": "last_frame",
        "thumbnail": "thumbnails/abc123_1.jpg"
      }
    ]
  },
  "user": {
    "name": "",
    "notes": "",
    "tags": [],
    "favorite": false
  }
}
```

### 5.1. Dane systemowe

System metadata mają być read-only z punktu widzenia managera.

Minimalnie:

- `fingerprint`,
- dokładny `prompt`,
- `clip_name`,
- informacje identyfikacyjne encodera dostępne już przy fingerprintowaniu,
- `cache_schema_version`,
- informacja o referencjach,
- ścieżki miniaturek,
- opcjonalnie informacyjna wersja ComfyUI.

Wersja ComfyUI w `verbose` jest diagnostyczna. Nie rozwiązuje automatycznie
problemu pełnej kompatybilności implementacji tokenizera i nie musi w tej fazie
wchodzić do fingerprintu.

### 5.2. Dane użytkownika

Edytowalne:

- `name`,
- `notes`,
- `tags`,
- `favorite`.

Nie dodajemy:

- `negative_prompt`,
- `revision`,
- `last_used_at`,
- `use_count`.

### 5.3. Merge przy odświeżeniu metadata

Jeżeli system ponownie zapisuje / backfilluje `verbose` dla tego samego
fingerprintu:

- część `system` może zostać odświeżona z aktualnych, zweryfikowanych danych,
- część `user` musi zostać zachowana,
- zapis ma być atomowy (`tmp` + `os.replace`).

Wzorzec nazwy pliku tymczasowego bierzemy z istniejącego `store.py`
(`_tmp_name()` z PID + `uuid4().hex` w nazwie tymczasowej). Sam zapis jest
jednak prostszy niż w core cache: Faza 1 pisze własny, minimalny atomowy
zapis `tmp write` + `os.replace()` dla POJEDYNCZEGO pliku (`.verbose.json`).
W przeciwieństwie do core cache nie ma tu dwóch sprzężonych artefaktów
(`.safetensors` + `.json`) do skoordynowanego posprzątania przy błędzie —
`os.replace()` na jednym pliku jest atomowy sam w sobie, więc logika
rollbacku z `save_conditioning()` (śledzenie utworzonych plików i kasowanie
ich w `except`) jest tu zbędna. (Historycznie `store.py` miał osobny helper
`_atomic_write_bytes`; został wchłonięty inline do `save_conditioning()`,
gdy doszedł tam wieloplikowy cleanup — `_tmp_name()` pozostał.) Patrz §23.3.

Kto woła zapis/backfill i skąd wie, że core cache faktycznie powstał —
patrz **§23.1**.

---

## 6. Prompt jest read-only w managerze

Prompt w managerze opisuje istniejący cache.

Nie wolno zrobić:

```text
fingerprint A
prompt A
↓ ręczna edycja metadata
prompt B
```

bo wtedy opis nie odpowiadałby conditioningowi.

Manager pozwala:

- wyświetlić cały prompt,
- skopiować / `Load` prompt do głównego noda,
- zmieniać nazwę, tagi, favorite i notes.

Zmiana samego promptu odbywa się po załadowaniu go do normalnego widgetu
`prompt` w `MiniMaxH3CLIPCachedImageToVideo`.

Po kolejnym uruchomieniu zmodyfikowany prompt naturalnie tworzy właściwy
fingerprint i ewentualnie nowy cache entry.

---

## 7. Backfill `verbose` dla istniejącego cache

Stare wpisy cache mogą nie mieć `.verbose.json`.

Z samego fingerprintu nie da się odzyskać promptu ani referencji.

Jednak przy ponownym użyciu takiego wpisu `CachedClipProxy` zna już:

- `fingerprint`,
- `prompt`,
- `clip_name`,
- `tokenize_kwargs`,
- wynik HIT/MISS.

Dlatego:

```text
HIT
+ core cache istnieje
+ verbose brak
→ zapisz / backfill verbose
→ zwróć conditioning normalnie
```

Backfill nie ładuje CLIP-a — używa danych dostępnych na ścieżce aktualnego HIT-a.

Stary cache, który nie został jeszcze ponownie użyty, może być widoczny po
`Check` jako wpis legacy z niedostępnymi szczegółami.

---

## 8. MISS, zapis cache i kolejność side-effectów

Na MISS:

```text
encode
→ save_conditioning()
→ core cache zapisany?
```

Dopiero po potwierdzonym zapisie właściwego cache tworzymy / odświeżamy
`verbose`.

Jeżeli:

```text
encode zakończony
core cache write failed
```

istniejąca polityka zostaje bez zmian:

```text
WARNING
→ zwróć gotowe cond
```

Nie tworzymy jednak normalnego wpisu managera, bo właściwy cache nie istnieje.

Jeżeli:

```text
core cache zapisany
verbose write failed
```

to:

```text
WARNING
→ conditioning/cache pozostają poprawne
```

Kolejny HIT może ponowić backfill `verbose`.

To "core cache zapisany?" nie ma dziś w kodzie żadnego jawnego sygnału poza
brakiem wyjątku — patrz **§23.1** po konkretną zmianę w `proxy.py`.

---

## 9. Polityka błędów cache

Zostaje świadoma asymetria istniejącego projektu.

### Odczyt

Korupcja / niekompletny wpis, który zgodnie z kontraktem oznacza brak używalnego
cache:

```text
→ MISS
```

Przykłady:

- brak skeleton JSON,
- brak odpowiadającego pliku tensorów,
- uszkodzony JSON,
- znane uszkodzenie formatu cache.

Prawdziwe błędy środowiskowe, np. uprawnienia / realne I/O, powinny być
widoczne użytkownikowi zamiast udawać MISS.

### Zapis po drogim encode

Po udanym encode błąd zapisu cache:

```text
→ WARNING
→ zachowaj i zwróć gotowy cond
```

Strata poprawnie obliczonego wyniku jest gorsza niż brak trwałego wpisu cache.

---

## 10. Miniaturki referencji

Dla bieżącego zakresu FLF interesują nas tylko obrazy, które faktycznie trafiają
do ścieżki CLIP:

- `first_frame`,
- `last_frame`.

`MiniMaxH3AddGuide` pozostaje poza zakresem, ponieważ nie korzysta z CLIP-a.

### 10.1. Gdzie ustalać label

Sam proxy widzi `images=[...]`, ale nie zawsze potrafi jednoznacznie odtworzyć,
czy pojedynczy obraz był `first_frame`, czy `last_frame`.

Wrapper głównego noda zna dokładnie wejścia:

```text
first_frame
last_frame
```

Dlatego labelowanie miniaturek powinno korzystać z informacji na poziomie
głównego wrappera, nie z heurystyki w proxy.

Potwierdzone w kodzie: `nodes.py:60-75` — `execute()` ma `first_frame` i
`last_frame` jako osobne, nazwane parametry, zanim jeszcze zawoła stockowy
`MiniMaxH3ImageToVideo.execute()`. To wystarcza do labelowania bez sięgania do
`kwargs["images"]` przechwyconych przez proxy — **z którego obrazu realnie
robimy miniaturkę, patrz §23.2 (jawna decyzja projektowa)**.

### 10.2. Zakres danych

Przechowujemy:

- miniaturkę,
- indeks,
- label `first_frame` / `last_frame`.

Nie przechowujemy:

- pełnego obrazu źródłowego,
- `source_filename`,
- automatycznie rekonstruowanej ścieżki `LoadImage`.

---

## 11. `Check` — centralna synchronizacja managera

Manager ma przycisk:

```text
Check
```

`Check` skanuje aktualny katalog cache i odświeża listę na podstawie rzeczywistego
stanu dysku.

`Check`:

- nie ładuje CLIP-a,
- niczego nie enkoduje,
- nie tworzy nowego conditioning,
- nie wykonuje preprocessingu H3.

Powinien:

1. znaleźć wpisy core cache,
2. znaleźć odpowiadające `.verbose.json`,
3. odczytać lekkie metadata,
4. sklasyfikować wpisy,
5. odświeżyć listę managera,
6. przeliczyć globalny rozmiar cache i liczbę wpisów.

### 11.1. Klasyfikacja

#### Normalny wpis

```text
<fingerprint>.json          ✓
<fingerprint>.safetensors   ✓
<fingerprint>.verbose.json  ✓
```

→ pełna pozycja managera.

#### Legacy

```text
core cache                  ✓
verbose                     ✗
```

→ cache istnieje, ale prompt / referencje mogą być niedostępne do czasu
ponownego użycia i backfillu.

#### Osierocony plik tensorów

```text
.json          ✗
.safetensors   ✓
```

→ nie jest normalnym HIT-em; może być uwzględniony diagnostycznie / w rozmiarze,
ale nie jako normalna pozycja biblioteki.

Potwierdzone w kodzie: `store.py:66-67` — `load_conditioning()` zwraca `None`
natychmiast, gdy brak `.json`, nigdy nawet nie sprawdzając `.safetensors`. Ta
klasyfikacja jest więc dokładnie spójna z realnym zachowaniem HIT/MISS, nie
tylko opisem na papierze.

#### Verbose bez core cache

```text
core cache       ✗
verbose          ✓
```

→ nie jest normalnym wpisem managera. Cache pozostaje źródłem prawdy.

### 11.2. Zakres weryfikacji

`Check` ma być szybki. Domyślnie skanuje filesystem i JSON metadata; nie powinien
ładować wszystkich dużych tensorów tylko po to, żeby odświeżyć UI.

Pełna integralność payloadu tensorowego pozostaje odpowiedzialnością normalnej
ścieżki cache przy realnym HIT.

---

## 12. Brak centralnego `index.json`

Nie tworzymy drugiej bazy, która mogłaby się rozjechać ze stanem cache.

```text
filesystem + core cache = source of truth
```

Po `Check` backend zwraca aktualny stan do frontendu.

Dane użytkownika (`name`, `notes`, `tags`, `favorite`) mieszkają w sidecarze
konkretnego fingerprintu, nie w centralnym indeksie.

---

## 13. Manager UI

Docelowy panel:

```text
┌───────────────────────────────────────────────┐
│ Search: [____________________]                │
│ Tag: [ All tags ▼ ]      [★ Favorites]       │
│                                               │
│ [ Check ]        Cache: 183 entries / 3.7 GB │
├───────────────────────────────────────────────┤
│ ★ Sidewalk interview S1       [night][dialog] │
│   [thumb] [thumb]                             │
│                                               │
│   Character intro             [portrait]      │
│   [thumb]                                     │
├───────────────────────────────────────────────┤
│ FULL PROMPT                                   │
│                                               │
│ [Shot 1] Live-action...                       │
│ ...                                           │
│                                               │
│ Notes                                         │
│ [___________________________________________] │
│ [___________________________________________] │
│                                               │
│ Tags: [night] [dialogue] [+]                  │
│ Name: [Sidewalk interview S1_______________] │
│ Favorite: ★                                   │
│                                               │
│ [ Load ]                         [ Delete ]    │
└───────────────────────────────────────────────┘
```

### 13.1. Search

Wyszukiwanie na żywo po:

- `name`,
- pełnym `prompt`,
- `notes`,
- `tags`.

Może działać po stronie JS po pobraniu aktualnej listy z backendu.

### 13.2. Tags

Tagi są listą stringów.

Manager ma:

- możliwość dodawania/usuwania tagów,
- combobox z filtrowaniem po istniejących tagach,
- wartość `All tags`.

Lista dostępnych wartości może być zbudowana jako unikalna suma tagów wszystkich
aktualnie załadowanych wpisów.

### 13.3. Favorite

Każdy wpis ma:

```json
"favorite": false
```

Manager pozwala:

- przełączyć gwiazdkę,
- filtrować tylko ulubione.

### 13.4. Notes

Każdy wpis ma edytowalny notatnik `notes`.

Notes wchodzą do wyszukiwania.

### 13.5. Pełny prompt

Kliknięcie elementu listy pokazuje **cały prompt niżej w tym samym oknie**.

Nie stosujemy tylko krótkiego preview jako głównego sposobu odczytu.

### 13.6. Globalny cache status

Wystarczy globalna informacja:

```text
Cache: <count> entries / <size>
```

Nie dodajemy cache statusu per preset / wpis.

---

## 14. `Load`

`Load`:

1. kopiuje dokładny prompt wybranego wpisu **do schowka systemowego**
   (`navigator.clipboard.writeText`), **nie** do widgetu w grafie.

   Powód (odkryty przy realnej implementacji, faza 6 część 3): próba
   automatycznego odnalezienia i zapisania do widgetu `prompt` na nodzie
   w grafie okazała się niepewna w dwóch normalnych, częstych
   konfiguracjach, nie edge case'ach:
   - **subgrafy** — `app.graph.findNodesByType()` nie schodzi rekurencyjnie
     do subgrafów, więc node `MiniMaxH3CLIPCachedImageToVideo` wewnątrz
     subgrafu jest dla niej niewidoczny;
   - **prompt przekonwertowany z widgetu na input** i podłączony z
     osobnego noda tekstowego — `node.widgets` nie zawiera wtedy takiego
     widgetu do ustawienia.

   Kopiowanie do schowka działa bezwarunkowo, niezależnie od struktury
   grafu, i jest bliżej ducha tej sekcji niż automatyczny zapis do
   widgetu ("Nie próbujemy automatycznie ... rekonstruować grafu" niżej).
2. nie próbuje automatycznie odtwarzać pełnych obrazów referencyjnych,
3. jeśli wpis został utworzony z referencjami, **musi wyświetlić użytkownikowi
   wyraźną informację o ich obecności**.

Przykład komunikatu:

```text
This cache entry was created with image references:
- first_frame
- last_frame
```

Jeżeli dostępne są miniaturki, powinny być widoczne przy komunikacie / wpisie.

Cel:

> użytkownik nie może dostać samego promptu i odnieść wrażenia, że identyczny
> fingerprint/cache zostanie odtworzony bez wymaganych obrazów.

Nie próbujemy automatycznie:

- odnajdywać oryginalnych plików,
- przepinać `LoadImage`,
- rekonstruować grafu.

---

## 15. `Delete`

W aktualnej architekturze `Delete` oznacza:

> **usuń cały odpowiadający wpis cache**, nie tylko metadata managera.

Przed usunięciem UI musi pokazać **confirmation**.

Po potwierdzeniu usuwamy:

1. `<fingerprint>.json`,
2. `<fingerprint>.safetensors`,
3. `<fingerprint>.verbose.json`,
4. odpowiadające miniaturki.

Kolejność core cache jest celowa:

```text
najpierw skeleton JSON
→ wpis natychmiast przestaje być HIT-em
→ potem tensor payload
```

Jeżeli po usunięciu zostanie osierocony plik pomocniczy, nie może on przywrócić
normalnego wpisu przy kolejnym `Check`.

Nie implementujemy obecnie osobnego „Hide”.

---

## 16. Backend — proponowane REST routes

Finalne nazwy endpointów można dopasować do aktualnego stylu repo po inspekcji
kodowej. Funkcjonalnie potrzebujemy:

| Route | Metoda | Rola |
|---|---|---|
| `/h3_cache_manager/check` | GET lub POST | skan cache + pełny aktualny model listy + global size/count |
| `/h3_cache_manager/get?fingerprint=` | GET | szczegóły jednego wpisu |
| `/h3_cache_manager/update` | POST | update tylko `user`: `name`, `notes`, `tags`, `favorite` |
| `/h3_cache_manager/delete` | POST | delete całego cache entry po confirmie wykonanym w UI |
| `/h3_cache_manager/thumbnail?...` | GET | serwowanie miniaturki, jeśli potrzebne przez frontend |

Nie potrzebujemy:

- `/save` tworzącego niezależny preset,
- `list?q=` jeśli wyszukiwanie wykonuje JS,
- `library_id`,
- osobnego node'a biblioteki.

Rejestracja zgodna ze standardowym mechanizmem `PromptServer.instance.routes`.

Potwierdzone w kodzie: grep po `PromptServer`/`routes`/`aiohttp` w całym repo
nie zwraca żadnych wyników — zero istniejących endpointów, zero ryzyka
kolizji nazw. To będzie pierwszy kod sieciowy w tym projekcie (patrz §23.5).

---

## 17. Zakres innych node'ów H3

### `MiniMaxH3AddGuide`

Poza zakresem.

Nie ma wejścia CLIP i nie wpływa na fingerprint tej warstwy.

### `MiniMaxH3ReferenceToVideo`

Poza bieżącą implementacją.

Istniejący cache/proxy/fingerprint mogą potencjalnie zostać rozszerzone także na
ReferenceToVideo, ale wymaga to osobnej, lokalnej weryfikacji aktualnej
implementacji i kontraktu `clip.tokenize()`.

Nie projektujemy teraz managera pod pełne RefToVideo kosztem komplikowania FLF.

---

## 18. Usunięte z poprzedniego planu

Świadomie usuwamy:

- osobną bibliotekę `prompt_library/`,
- `uuid4`,
- `negative_prompt`,
- `linked_fingerprints`,
- przycisk „Zapisz jako…” przed encode,
- `library_id`,
- node `MiniMaxH3PromptLibrary`,
- `revision`,
- `last_used_at`,
- `use_count`,
- per-entry cache status,
- import/export,
- prewarm,
- `MiniMaxH3PrewarmCacheFLF`,
- `MiniMaxH3PrewarmCacheReference`,
- automatyczne LRU/eviction,
- automatyczne śledzenie źródłowych `LoadImage`.

---

## 19. „Może kiedyś”

Na liście możliwych przyszłych dodatków pozostaje:

- **`revision` / historia zmian user metadata** — dopiero jeśli pojawi się realna
  potrzeba wersjonowania; obecnie częste ręczne edycje i cofanie nie uzasadniają
  prostego rosnącego licznika bez prawdziwej historii.

Nie jest to część MVP ani obecnego kontraktu danych.

---

## 20. Techniczne przypomnienia

### Fingerprint

Aktualny fingerprint pozostaje technicznym kluczem cache i managera.

Nie próbujemy tanio „udawać” pełnego hasha całej implementacji tokenizera.
Problem kompatybilności implementacji pozostaje świadomym ograniczeniem.

Możliwe lekkie częściowe zabezpieczenie w przyszłości:

- jawny `CACHE_COMPAT_VERSION`,
- kontrolowany `ENCODER_IMPL_VERSION`,
- wersja ComfyUI jako dodatkowy salt.

Wersja ComfyUI może być zapisana już teraz w `verbose` informacyjnie, bez
uznawania jej za pełne rozwiązanie problemu kompatybilności.

### Core cache write

Zachować istniejącą bezpieczną kolejność:

```text
tensory
→ skeleton JSON
```

### Core cache delete

Odwrotnie:

```text
skeleton JSON
→ tensory
```

### Stock H3 jako źródło prawdy

Nie kopiować preprocessingu `_resize`, tokenizacji ani logiki H3 do managera.

Cache nadal ma powstawać przez istniejącą, zweryfikowaną ścieżkę stockowego
`MiniMaxH3ImageToVideo.execute()` + `CachedClipProxy`.

Manager nie może stać się drugim równoległym systemem encode/cache.

---

## 21. Plan faz implementacyjnych

### Faza 1 — verbose metadata store

Dodać:

- model danych `.verbose.json`,
- odczyt,
- atomowy zapis,
- merge `system` + zachowanie `user`,
- testy bez GPU.

Układ pliku/testów — patrz **§23.3**, **§23.4**.

### Faza 2 — integracja proxy / core cache

Dodać minimalny stan potrzebny do powiązania wykonania z fingerprintem, np.:

- ostatni fingerprint,
- HIT/MISS,
- informacja czy core cache faktycznie istnieje / został zapisany.

Obsłużyć:

- verbose na nowym MISS po udanym zapisie core cache,
- backfill verbose na HIT,
- brak wpływu błędu verbose na poprawność core cache.

Konkretne nazwy pól i logika — patrz **§23.1**.

Testy jednostkowe.

### Faza 3 — thumbnails

Dodać helper:

```text
IMAGE tensor
→ PIL
→ resize
→ thumbnail
```

Labelowanie:

- `first_frame`,
- `last_frame`.

Źródło tensora do miniaturki — patrz **§23.2**.

Testy na sztucznych tensorach.

### Faza 4 — cache scanner / Check

Dodać scanner:

- normal,
- legacy,
- orphan,
- global size/count.

Bez ładowania CLIP i bez hurtowego ładowania tensorów.

Testy filesystemowe.

### Faza 5 — REST backend

Dodać:

- check,
- get,
- update user metadata,
- delete core cache,
- thumbnail serving jeśli potrzebne.

Testy backendu — patrz **§23.4** (stub `PromptServer` w `conftest.py`).

### Faza 6 — Manager JS

Dodać:

- `Check`,
- live search,
- tag combobox,
- Favorites,
- listę + miniaturki,
- pełny prompt pod listą,
- `name`,
- `notes`,
- `tags`,
- `favorite`,
- globalny cache size/count,
- `Load`,
- informację o image refs przy `Load`,
- `Delete` + confirmation.

Frontend testować osobno od backendu.

### Faza 7 — realny E2E

Scenariusz:

```text
MISS
→ encode
→ core cache
→ verbose
→ thumbnails
→ Check
→ wpis widoczny
→ name/tags/notes/favorite
→ Load
→ informacja o refs
→ kolejny RUN = HIT
→ Delete + confirm
→ Check
→ wpis zniknął
→ kolejny RUN = MISS
```

Dodatkowo:

```text
stary core cache bez verbose
→ Check = legacy
→ ponowne użycie = HIT + backfill
→ Check = pełny wpis
```

---

## 22. Zasady pracy przy implementacji

Przed każdą zmianą sprawdzamy aktualny lokalny stan repo.

Nie opieramy patcha na pamięci ani na publicznym GitHubie, jeśli dokładny lokalny
kod nie został potwierdzony.

Praca sekwencyjna:

```text
sprawdź lokalny baseline
→ jedna mała zmiana
→ najmniejszy test
→ rzeczywisty wynik
→ diff
→ commit / push stabilnego checkpointu
→ kolejny etap
```

Nie zakładamy powodzenia poprzedniego kroku bez wyniku.

---

## 23. Ustalenia po review kodu (v2)

Ta sekcja jest w całości nowa względem wersji 1. Nie zmienia żadnej decyzji
architektonicznej z sekcji 1–22 — domyka tylko konkretne szczegóły techniczne,
które plan zostawiał otwarte, a które trzeba rozstrzygnąć przed napisaniem
pierwszej linijki kodu Fazy 1/2/3.

### 23.1. Rozszerzenia `CachedClipProxy` wymagane w Fazie 2

Dziś `encode_from_tokens_scheduled()` (`proxy.py:46-86`) liczy `fingerprint`
lokalnie i nigdy go nie zwraca, a błąd zapisu cache jest tylko logowany
(`except Exception as e: logger.warning(...)`, `proxy.py:79-85`) — bez
żadnego śladu w stanie obiektu. To za mało dla §7/§8: `nodes.py` musi po
powrocie ze stockowego `MiniMaxH3ImageToVideo.execute()` wiedzieć, **z jakim
fingerprintem** operacja się skończyła, **czy to był HIT czy MISS**, i **czy
core cache faktycznie został zapisany** — dopiero to pozwala zdecydować, czy
zapisać nowy `verbose`, zrobić backfill, czy nic nie robić.

Proponowane, minimalne dodanie (bez zmiany istniejącej sygnatury żadnej
metody publicznej):

```python
# w CachedClipProxy.__init__:
self.last_fingerprint = None
self.last_hit = None              # True / False, None dopóki nic nie policzono
self.last_core_cache_written = None  # True / False / None (write nie było próbowane)
```

Ustawiane na końcu `encode_from_tokens_scheduled()`:

- `last_fingerprint = fingerprint` — zawsze, na HIT i MISS/REFRESH,
- `last_hit = True` na ścieżce `load_conditioning() is not None`,
  `last_hit = False` na ścieżce MISS/REFRESH,
- `last_core_cache_written`:
  - `True`, jeśli `save_conditioning()` zwróciło się bez wyjątku,
  - `False`, jeśli złapany `except Exception` (dokładnie ta sama gałąź co
    dziś — WARNING zostaje bez zmian),
  - `None` na ścieżce HIT, gdzie `save_conditioning()` w ogóle nie jest
    wołane.

`store.save_conditioning()` **zostaje bez zmian** — nic nie zwraca, nadal
albo się udaje, albo rzuca. Cała logika "czy się udało" zostaje lokalnie w
`proxy.py`, w istniejącym `try/except`, tylko rozszerzonym o `else:` (albo
przypisanie przed/po bloku) ustawiające `last_core_cache_written`. To
zachowuje zasadę z §20 "Stock H3 jako źródło prawdy" w duchu: nie poszerzamy
kontraktu store'a bez potrzeby, zmiana zostaje możliwie mała i lokalna.

`nodes.py.execute()` czyta te trzy pola z `proxy` **po** wywołaniu
stockowego `execute()`, w tym samym `finally`, gdzie dziś już sprawdza
`proxy.did_load_real_clip` (linia 84) — naturalne miejsce na wywołanie
zapisu/backfillu `verbose` (Faza 2) i thumbnaili (Faza 3), zanim `proxy`
zostanie `del`-owany.

### 23.2. Źródło danych dla thumbnaila (decyzja)

Plan (§10.2) mówi "przechowujemy miniaturkę", nie precyzując z którego
tensora. Są dwie możliwości:

1. surowy `first_frame` / `last_frame`, taki jaki wszedł do
   `nodes.py.execute()` — **przed** resize,
2. obraz **po resize**, który realnie trafił do `clip.tokenize()` jako
   `images=[...]` — to on wchodzi do fingerprintu (README, sekcja "Cache
   key": "the exact list of images the stock node's frame-resize step
   produced").

**Decyzja: opcja 1 — surowy tensor z `nodes.py`.**

Powody:

- proxy nie gwarantuje 1:1 odpowiedniości między `kwargs["images"]` a
  `(first_frame, last_frame)` (np. gdy tylko jeden z nich jest podany, lista
  może mieć inną długość/kolejność niż zakładana) — sięganie po nią
  wymagałoby wiedzy o wewnętrznej konwencji stockowego noda, czyli
  dokładnie tego, czego §20 zabrania ("Stock H3 jako źródło prawdy... Nie
  kopiować preprocessingu `_resize`... do managera"),
- miniaturka to czysto UI-owa pomoc pamięciowa, element warstwy pomocniczej
  (§4: "verbose + thumbnails = metadata i UI managera"), nie element źródła
  prawdy — nie musi bit-for-bit odpowiadać temu, co poszło do encodera,
- `nodes.py.execute()` i tak już ma bezpośredni dostęp do obu tensorów jako
  osobne, nazwane parametry (§10.1) — zero dodatkowej złożoności.

Konsekwencja dla Fazy 3: helper `thumbnails.py` przyjmuje gotowy
`IMAGE`-tensor (dokładnie ten z parametru `execute()`), nie zna nic o
`tokenize_kwargs` ani o proxy.

### 23.3. Proponowany układ modułów

Trzymając się istniejącej konwencji nazewniczej repo (`fingerprint.py`,
`store.py`, `serialize.py`, `loader.py` — każdy jeden, wąski cel):

```text
minimaxh3_clipcache/
├── verbose_store.py     # Faza 1: read/write .verbose.json, merge system+user, atomowo
├── thumbnails.py         # Faza 3: IMAGE tensor -> PIL -> resize -> .jpg
├── scanner.py            # Faza 4: klasyfikacja normal/legacy/orphan + size/count
└── routes.py             # Faza 5: PromptServer.instance.routes, importowany z __init__.py

web/
└── js/
    └── h3_cache_manager.js   # Faza 6, rejestrowany przez WEB_DIRECTORY w __init__.py
```

`proxy.py` i `nodes.py` dostają wyłącznie minimalne rozszerzenia z §23.1 —
żadnej nowej odpowiedzialności poza tym.

### 23.4. Konwencja testów dla nowych modułów

Każdy nowy moduł dostaje swój plik testów w tym samym stylu co istniejące:

```text
tests/test_verbose_store.py   # obok tests/test_store.py
tests/test_thumbnails.py
tests/test_scanner.py
tests/test_routes.py
```

Zero GPU, zero realnego ComfyUI — dokładnie jak obecne 45 testów. Dla Fazy 5
(`test_routes.py`) potrzebny będzie stub `PromptServer.instance.routes` w
`tests/conftest.py`, analogiczny do istniejących stubów `comfy`/
`folder_paths`, które już pozwalają importować `nodes.py` pod pytest bez
prawdziwego ComfyUI (patrz `tests/conftest.py` i wzorzec importu z
`importlib.util` w `tests/test_node.py`).

### 23.5. REST routes — brak konfliktów

Potwierdzone grepem po całym repo na baseline `bb97275`: zero wystąpień
`PromptServer`, `routes`, `aiohttp`. Endpointy z §16
(`/h3_cache_manager/check`, `/get`, `/update`, `/delete`, `/thumbnail`) będą
pierwszym kodem sieciowym w tym projekcie — żadnego ryzyka kolizji nazw ani
istniejącej rejestracji do wzięcia pod uwagę.

---

# Finalny invariant

Manager ma odpowiadać na pytanie:

> **Jakie conditioning CLIP mam faktycznie zapisane na dysku, z jakiego promptu
> i referencji powstały, i jak chcę je sobie nazwać / otagować / opisać?**

Nie ma odpowiadać na pytanie:

> „Jak wygląda cały workflow generacji?”

Architektura docelowa:

```text
realny H3 run
    ↓
CachedClipProxy
    ↓
fingerprint
    ↓
core cache (.json + .safetensors)
    ↓
verbose + thumbnails
    ↓
Check
    ↓
Cache Manager
```

Właściwości:

```text
cache = source of truth
fingerprint = identity
prompt = read-only metadata
user metadata = name/tags/favorite/notes
Check = rebuild/refresh widoku
Load = prompt + ostrzeżenie o referencjach
Delete = pełny cache entry + confirmation
prewarm = usunięty
```
