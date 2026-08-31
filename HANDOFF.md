# HANDOFF

## Stan na: 2026-08-31 / branch master

## Ostatnio zrobione
- Rozpoznanie: czy `width`/`height` w stockowych `MiniMaxH3ImageToVideo`
  i `MiniMaxH3ReferenceToVideo` (`comfy_extras/nodes_minimax_h3.py`,
  lokalny ComfyUI 0.34.2) wpływa na wejście `clip.tokenize(...)`, czy jest
  wyłącznie metadaną/rozmiarem zwracanej LATENT. Czysty odczyt przepływu
  danych w źródle — jednoznaczny, bez potrzeby testu empirycznego. Kodu
  nie pisano.
- (Poprzednia sesja) Rozpoznanie mechanizmu typu COMBO pod node
  "MiniMax H3 CLIP Name" — patrz commit 0c26627 w historii; wynik:
  `RETURN_TYPES = (folder_paths.get_filename_list("text_encoders"),)`,
  literalny `("COMBO",)` pada na Queue.

## Ustalenia istotne dla Chat

### FL2VA — MiniMaxH3ImageToVideo.execute() (nodes_minimax_h3.py:134-159)
- `width`/`height` wpływa na `clip.tokenize()` **TAK, ale tylko gdy podano
  first_frame i/lub last_frame**:
  - linia 143: `_resize(first_frame[:1], width, height, "disabled")` —
    plain stretch DOKŁADNIE do `(width, height)`.
  - linia 148: `_resize(last_frame[:1], width, height, "center")` —
    cover-crop do `(width, height)`.
  - linia 152: `clip.tokenize(prompt, images=images)` — `images` zawiera
    te już-przeskalowane tensory. To jest PRZED tokenize. Klasa (a).
- Czysty t2va (brak keyframes): `images=[]` → `width`/`height` NIE dotyka
  `tokenize()`. Wtedy jedyne użycie to linia 137 `_empty_av_latent(width,
  height, length)` → kształt zwracanej pustej LATENT (`height//16`,
  `width//16`). Klasa (b).
- linia 157: `vae.encode(kf.pop("image"))` — latenty keyframe, ścieżka
  VAE, PO `encode_from_tokens_scheduled`, doklejane jako
  `minimax_keyframes` przez `conditioning_set_values` (linia 158). Klasa
  (b). Rozmiaro-zależne (obraz był `_resize`'owany do width/height), ale
  to nie CLIP.
- `prompt` przekazywany bez zmian; `width`/`height` NIGDY nie wchodzą do
  tekstu promptu.

### Ref2VA — MiniMaxH3ReferenceToVideo.execute() (nodes_minimax_h3.py:285-355)
- `width`/`height` wpływa na `clip.tokenize()` **TAK, ale tylko przez ref
  images w trybie `ref_image_size="match"` (domyślny) i tylko gdy ref
  obraz jest WIĘKSZY powierzchniowo niż `width*height`**:
  - linia 299: `scale = min(1.0, math.sqrt((width*height)/(w*h)))` —
    liczy się WYŁĄCZNIE iloczyn `width*height` (pole), nie proporcje.
    Gdy `width*height >= w*h` → `scale = 1.0` → ref zachowuje własny
    rozmiar (do 32) → `width`/`height` bez efektu.
  - linie 302-304: `tw`, `th`, `_resize(img[:1], tw, th, "disabled")`.
  - linia 306: `ref_items.append({"type": "image", "data": resized})`.
  - linia 351: `clip.tokenize(prompt, minimax_ref_items=ref_items)`.
    PRZED tokenize. Klasa (a).
- `ref_image_size="max"`: linia 301 używa `REF_IMAGE_SHORT_EDGE` (2048),
  `width`/`height` bez efektu na `tokenize()`.
- ref videos: **NIE** — linia 316 `adapt_canvas(vw, vh)` liczy kanwę
  wyłącznie z wymiarów samego wideo referencyjnego + stałych (768,
  768*1344); `width`/`height` node'a nie wchodzi. `qwen_frames` (linia
  337) → `ref_items` (linia 338) są w rozmiarze pochodnym od wideo, nie
  od width/height.
- ref audio: brak `width`/`height`.
- linia 288: `_empty_av_latent(width, height, length)` → kształt
  zwracanej LATENT. Klasa (b).
- linie 305/329/347 `vae.encode(...)` → `ref_blocks` → `minimax_refs`
  doklejane PO encode (linie 353-354). Klasa (b). `latent_h`/`latent_w`
  w ref_blocks (linia 307) pochodzą z `th`/`tw` → w trybie "match"
  pośrednio zależą od `width*height`, ale to payload DiT (VAE), nie CLIP.
- `prompt` bez zmian; `width`/`height` nie wchodzą do tekstu.

### Implikacja dla planowanej funkcji "cond dla upscalingu latenów"
- Hidden states z Qwena (`encode_from_tokens_scheduled`) są niezależne od
  `width`/`height` TYLKO w: FL2VA bez keyframes; Ref2VA bez ref images;
  Ref2VA z ref images w trybie "max"; Ref2VA gdzie każdy ref obraz jest
  mniejszy powierzchniowo niż `width*height` w trybie "match".
- W pozostałych przypadkach zmiana `width`/`height` zmienia piksele
  wchodzące do enkodera → inny CONDITIONING → inny fingerprint; nie da
  się przenieść na inny rozmiar bez ponownego encode.
- Część conda doklejana po encode (`minimax_keyframes` / `minimax_refs`,
  latenty VAE) jest rozmiaro-zależna ZAWSZE — nawet gdy hidden states
  nie są.

## Otwarte pytania
- Kruchość wariantu COMBO z gołą listą (patrz poprzednia sesja):
  `RETURN_TYPES` liczone raz przy imporcie; dodanie/usunięcie pliku
  w `models/text_encoders` w sesji rozjeżdża listę do restartu.
- Frontendowa część rozpoznania COMBO to odczyt source-map 1.49.6, nie
  test w żywym UI — do domknięcia jeden fizyczny test.
