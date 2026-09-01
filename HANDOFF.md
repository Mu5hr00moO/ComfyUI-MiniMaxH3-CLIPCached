# HANDOFF

## Stan na: 2026-09-01 / branch master / commit 8a1ac81

## Ostatnio zrobione (rozpoznanie do ZLECENIA: embedding: w promptcie a tożsamość cache)

Tylko rozpoznanie (ZLECENIE punkt 1). Zero zmian w kodzie produkcyjnym.
Skrypt eksploracyjny w scratchpadzie (niecommitowany), uruchomiony w
comfyenv na żywym łańcuchu tokenizera MiniMax H3.

Cel ZLECENIA: dziś podmiana pliku textual-inversion pod tą samą nazwą,
przy niezmienionym promptcie i checkpoincie, może dać cichy stary HIT --
bo `compute_fingerprint()` liczy fingerprint z samego stringa promptu +
`tokenize_kwargs`, a `embedding:name` jest rozwiązywane do pliku dopiero
wewnątrz prawdziwego tokenizera na realnym MISS.

## Ustalenia istotne dla Chat

- **Gdzie stock parsuje `embedding:`** -- `comfy/sd1_clip.py:594`,
  `SDTokenizer.tokenize_with_weights`:
  `re.split(r'(?<=\s)embedding:', to_tokenize)` (plus `embedding:` na
  początku stringa). Identyfikator `"embedding:"` -- `sd1_clip.py:537`.
- **Czyszczenie nazwy** -- `SDTokenizer._try_get_embedding`
  (`sd1_clip.py:543`): `name.split()[0]`, ucięcie na pierwszym `<` lub `[`
  (`re.search(r'[<\[]')`), ponowna próba po zdjęciu przecinków z końca.
- **Rozwiązanie do pliku** -- `load_embed` (`sd1_clip.py:415`), funkcja
  MODUŁOWA (nie metoda): iteruje `expand_directory_list(embedding_directory)`
  (chodzi też po podkatalogach, `sd1_clip.py:397`),
  `os.path.join(embed_dir, name)` z zabezpieczeniem `os.path.commonpath`
  przed wyjściem z katalogu; próbuje literalnej ścieżki, potem dokleja
  `.safetensors`, `.pt`, `.bin` (`sd1_clip.py:431`). Zwraca **tensor**
  (albo None) -- **nigdy ścieżki**.
- **Łańcuch MiniMax** -- `MiniMaxH3Tokenizer` (`comfy/text_encoders/minimax.py:136`)
  -> `SD1Tokenizer` z sub-tokenizerem `MiniMaxQwenSDTokenizer` pod atrybutem
  `qwen3vl_32b`, `embedding_size=5120`, `embedding_key="qwen3vl_32b"`.
  `tokenize_with_weights` (`minimax.py:148`) puszcza tekst użytkownika
  WYŁĄCZNIE przez `add_text(text)` ->
  `self.qwen3vl_32b.tokenize_with_weights(s, return_word_ids=False, disable_weights=True)`
  (`minimax.py:155`). `<Picture N>: ` / `<Audio N>: ` / `<Video N>: ` /
  `<N.N seconds>` to literały format-stringów -- składnia `embedding:` może
  wejść tylko z pola `prompt` węzła.
- **Sub-tokenizer da się zbudować bez enkodera 27 GB.** Prawdziwy CLIP
  woła `MiniMaxH3Tokenizer(embedding_directory=..., tokenizer_data={})` --
  dla MINIMAX nie ma żadnych `tokenizer_data` z checkpointu
  (`grep` po `comfy/sd.py`: tylko minimax_music). `embedding_directory` =
  `folder_paths.get_folder_paths("embeddings")` (`folder_paths.py:34`,
  `[<models_dir>/embeddings]`) -- dokładnie ta sama wartość, którą
  `loader.build_clip_loader_fn` już podaje do `comfy.sd.load_clip`.
- **Koszt (zmierzony w comfyenv):** budowa tokenizera ~0.26-0.32 s
  (ładuje wolny `transformers.Qwen2Tokenizer` z dołączonego
  `comfy/text_encoders/qwen25_tokenizer`, 4.3 MB). Cache'owalne raz na
  proces (vocab/merges nie zmieniają się w runtime -- ten sam wzorzec co
  `encoder_abi.get_encoder_abi_id()`). Skan pojedynczego promptu:
  0.1-1.6 ms Z ładowaniem pliku embeddingu. Plik TI dla MiniMax to tensor
  `(5120,)` lub `(N, 5120)` -- ~20 KB na jeden wektor.
- **`SDTokenizer.tokenize_with_weights(prompt, return_word_ids=False, disable_weights=True)`
  zwraca `(token, weight)`; rozwiązany embedding to `(torch.Tensor, weight)`
  -- jeden wpis na wektor przy TI wielowektorowym (`sd1_clip.py:608-611`).**
  Przejście po wyniku i zebranie tokenów-tensorów w kolejności daje
  dokładnie tę treść embeddingu, która trafi do enkodera, przy ZEROWEJ
  reimplementacji regexa/czyszczenia nazwy/rozwiązywania pliku.
- **Zweryfikowane empirycznie:** podmiana treści pliku pod tą samą nazwą
  -> inny tensor (`torch.equal` = False). Brak pliku -> stock loguje
  `warning, embedding:X does not exist, ignoring` i nie emituje tensora
  (hash wtedy równy przypadkowi bez embeddingu -- zgodnie z realnym
  zachowaniem stocka). Stock sam obsługuje: `embedding:` na początku,
  końcówkę `.safetensors`, przecinek na końcu, `embedding:name<...>`,
  `embedding:subdir/name` -- bez żadnego naszego regexa.
- **IS_CHANGED** dostaje literalny `prompt` w `**kwargs` (`execution.py:91`
  -> `get_input_data(..., execution_list=None)`; `_is_changed_common`
  (`nodes.py:263`) już bierze `**kwargs`). OGRANICZENIE: jeśli `prompt`
  jest przełączony z widgetu na wejście-link, `get_input_data` z
  `execution_list=None` oznacza go jako missing -> `IS_CHANGED` widzi
  `prompt=None` (ta sama klasa ograniczenia co istniejąca ścieżka
  `clip_name=None`). Fingerprint w proxy jest na to odporny (zawsze widzi
  rozwiązany prompt przez `tokenize()`), więc ochrona na poziomie
  fingerprintu jest szczelna niezależnie; degraduje się tylko wtórna
  osłona "nie pozwól ComfyUI pominąć węzła" dla linkowanego promptu.
- Wyjątek w IS_CHANGED jest łapany przez ComfyUI -> NaN -> re-egzekucja
  (`execution.py:96-98`). Bezpieczny fallback.

## Otwarte pytania

- **OTWARTE_PYTANIA_DO_CLAUDE w raporcie** -- rozwidlenie: (A) hash
  rozwiązanych tensorów embeddingu (zero reimplementacji, content-addressed,
  bez ścieżki) vs (B) `name + stat()` jak `resolve_clip_stat()` (lżejsze,
  ale stock nie wystawia ścieżki -> trzeba by odtworzyć pętlę `load_embed`,
  ~20 linii cudzych internali -- przed czym ZLECENIE ostrzega). Plus:
  czy nowy komponent ma być NIEOBECNY dla promptów bez `embedding:` (zero
  unieważnienia cache) czy bump `CACHE_SCHEMA_VERSION` (jednorazowe
  przesunięcie wszystkiego). Rekomendacja CC: A + wariant nieobecny.
  Implementacja WSTRZYMANA do decyzji.

## Sugestie (nie polecenia)

- Wariant A da się w całości oprzeć na module-level funkcji `load_embed`
  + realnym sub-tokenizerze -- prowenancja czysta, łatwo wyłączalne
  (guard `"embedding:" in prompt`), zgodne z regułą "reuse proven
  mechanics".
