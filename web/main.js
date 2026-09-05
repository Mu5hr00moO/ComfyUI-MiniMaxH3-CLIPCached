// MiniMax H3 Cache Manager frontend.
//
// Phase 6 part 1: floating launcher, modal shell, "Check", raw entry list.
// Phase 6 part 2: client-side search / tag / favorite filtering, reference
// thumbnails, inline detail+edit panel (name / notes / tags / favorite).
// Phase 6 part 3 (this file): "Copy prompt" puts a cached entry's prompt on
// the clipboard (writing it into a graph widget was tried and dropped --
// findNodesByType() does not descend into subgraphs, and a prompt converted
// to an input has no widget to set), still showing the image-reference
// notice; "Delete" removes a whole cache
// entry after a window.confirm().
//
// Structure follows the local convention in
// custom_nodes/MiniMaxH3-Prompt-Writer/web/main.js: a singleton panel built
// once, injectStyles via a <link> guarded by a data-attribute,
// app.registerExtension with commands/menuCommands/setup, and every HTTP
// call through api.fetchApi so a ComfyUI mounted on a sub-path still works.
// Markup, CSS and naming here are our own. Kept as one file -- still small.

import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

const EXTENSION_NAME = "minimax.h3.cache.manager";
const API_PREFIX = "/h3_cache_manager"; // must match minimaxh3_clipcache/routes.py
const ALL_TAGS = ""; // sentinel select value meaning "no tag filter"

let panel = null; // built once -- see createPanel()
let lastCheckResult = null; // last /check response, for client-side filtering
let currentVariant = "fl2va"; // "fl2va" | "ref2va" -- which node's entries the list shows
let openDetailFingerprint = null; // fingerprint whose detail panel is shown
let objectUrls = []; // list-row thumbnail blob URLs to revoke on the next list render
let detailRefObjectUrls = []; // detail-panel reference thumbnail blob URLs, revoked
// only when renderDetailRefs() rebuilds -- kept apart from objectUrls so a
// list re-render (search/tag/favorite typing) never revokes a live detail thumb
let renderGeneration = 0; // bumped each render so stale async thumbnails bail
let checkGeneration = 0; // only the newest overlapping /check may update UI state
let copyResultObjectUrls = []; // thumbnail blob URLs shown in the "Copy prompt" result box

// --- styles -----------------------------------------------------------------

function injectStyles() {
  if (document.querySelector("link[data-h3cm-styles]")) return;
  const link = document.createElement("link");
  link.rel = "stylesheet";
  link.href = new URL("./styles.css", import.meta.url).href;
  link.dataset.h3cmStyles = "true";
  document.head.appendChild(link);
}

// --- helpers ---------------------------------------------------------------

export function formatBytes(bytes) {
  if (typeof bytes !== "number" || !isFinite(bytes) || bytes < 0) return "—";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${unit === 0 ? value : value.toFixed(1)} ${units[unit]}`;
}

// Parse a number that arrived from outside this module -- a text <input>, a
// localStorage value, an API field -- and return null (never NaN, never a
// substituted default) when it is not a finite number inside [min, max].
// Returning null is what lets a caller tell "not configured" apart from
// "configured to something unusable". positiveBytes() answers a different
// question (it substitutes 0, which entryOwnSizeBytes() renders as "nothing
// to show") and sits on the entry-size render path, so it is left alone.
export function parseFiniteNumber(raw, { min = -Infinity, max = Infinity } = {}) {
  let value;
  if (typeof raw === "number") {
    value = raw;
  } else if (typeof raw === "string") {
    // Number("") and Number("   ") are both 0, which would silently turn an
    // empty input box into a real value.
    if (raw.trim() === "") return null;
    value = Number(raw);
  } else {
    return null;
  }
  if (!Number.isFinite(value)) return null;
  return value >= min && value <= max ? value : null;
}

export function parseTags(raw) {
  return String(raw || "")
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean);
}

export function shortPrompt(prompt, max = 60) {
  const text = String(prompt || "");
  return text.length > max ? `${text.slice(0, max)}…` : text;
}

export function entryLabel(entry) {
  const user = (entry.verbose && entry.verbose.user) || {};
  const system = (entry.verbose && entry.verbose.system) || {};
  if (user.name && user.name.trim()) return user.name.trim();
  const p = shortPrompt(system.prompt);
  return p || "(no prompt)";
}

export function formatCreatedAt(value) {
  if (typeof value !== "string" || !value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "—";
  try {
    return new Intl.DateTimeFormat(undefined, {
      year: "numeric", month: "2-digit", day: "2-digit",
      hour: "2-digit", minute: "2-digit",
    }).format(date);
  } catch (e) {
    return "—";
  }
}

export function formatGenerationSize(system) {
  const width = system && system.width;
  const height = system && system.height;
  if (typeof width !== "number" || typeof height !== "number") return "";
  const megapixels = typeof (system && system.megapixels) === "number"
    ? system.megapixels
    : (width * height) / 1_000_000;
  return `${width}×${height} (${megapixels.toFixed(2)} MP)`;
}

// Tooltip for the whole entry meta line (formatEntryMetaLine). It is built
// from independent sentences, one per field on the line that needs
// explaining, so a field that is not on screen contributes nothing:
//
//   resolution -- created_at is fixed at first write while
//     system.width/.height/.megapixels track the most recent run, so the two
//     halves of the line come from different moments. The resolution trio is
//     informational only, never part of the fingerprint or the HIT/MISS
//     decision.
//   pair total -- the byte figure covers BOTH halves of a folded
//     dual-resolution pair, while the Delete beside it takes one fingerprint.
//     Without this the number reads as a promise about that button.
//
// The pair-total sentence is gated on `size`, NOT on the resolution: an entry
// can carry a size with no width/height at all (nothing connected to the node
// records a resolution), and that entry still needs the warning. Returns ""
// when neither sentence applies, so nothing gets an empty tooltip attribute
// with a stray hover target.
export function entryMetaTooltip(system, size = null) {
  const sentences = [];
  if (formatGenerationSize(system)) {
    sentences.push(
      "Creation date and generation resolution come from two different " +
      "moments: the date is fixed when the entry is first written, while the " +
      "resolution is that of the most recent run that used it. One cached " +
      "encode serves every resolution when no keyframes are connected -- the " +
      "encode itself does not depend on width/height.",
    );
  }
  if (size && size.isPairTotal) {
    sentences.push(
      "The size shown is the pair total: this entry plus its rescaled " +
      "partner. Delete removes only this entry -- the partner is listed " +
      "under the \"+ rescaled to\" badge and has its own Delete button.",
    );
  }
  return sentences.join(" ");
}

function positiveBytes(value) {
  return typeof value === "number" && isFinite(value) && value > 0 ? value : 0;
}

// One entry's own on-disk bytes, as reported by scan_cache(): exactly the
// files a Delete of that entry frees (scanner.entry_file_paths()). Anything
// that is not a positive finite number -- a "/check" response from a build
// before the field existed, a 0 -- counts as "nothing to show", which
// formatEntryMetaLine() renders by leaving the meta line untouched.
export function entryOwnSizeBytes(entry) {
  return positiveBytes(entry && entry.size_bytes);
}

// What ONE line should report for `entry`: { bytes, isPairTotal }.
//
// A valid dual-resolution pair is folded into a single visible row
// (resolvePairing() -> "valid"), so that row answers for both halves and gets
// the sum. Every other status -- none / orphaned / inconsistent-pair /
// role-unknown -- renders each side as its own row, so each reports only
// itself. The partner strip under a folded row skips this function and takes
// entryOwnSizeBytes() instead: it is the second line describing the same pair,
// and repeating the total there would read as double counting.
//
// `isPairTotal` travels WITH the number rather than being re-derived by each
// caller, because the two must never disagree: the flag is what puts the
// "(pair total)" marker on the line and the matching warning in the tooltip,
// and a sum shown without it reads as a promise about the adjacent Delete
// button -- which acts on this one fingerprint, not on the pair.
export function entryDisplaySize(entry, pairing) {
  const own = entryOwnSizeBytes(entry);
  if (!pairing || pairing.status !== "valid" || !pairing.partner) {
    return { bytes: own, isPairTotal: false };
  }
  return { bytes: own + entryOwnSizeBytes(pairing.partner), isPairTotal: true };
}

// `sizeBytes` and the pair-total flag are supplied by the caller rather than
// derived here: only the caller knows whether its line stands for one entry or
// for a folded pair (see entryDisplaySize). This function never looks at
// pairing. The third argument is normally the entryDisplaySize() descriptor
// itself, which carries `isPairTotal`; the partner strip passes a bare byte
// count and so gets no marker.
export function formatEntryMetaLine(system, sizeBytes = 0, { isPairTotal = false } = {}) {
  const sizeText = formatGenerationSize(system);
  const dateText = formatCreatedAt(system && system.created_at);
  const line = sizeText ? `${dateText} · ${sizeText}` : dateText;
  const bytes = positiveBytes(sizeBytes);
  if (!bytes) return line;
  return `${line} - ${formatBytes(bytes)}${isPairTotal ? " (pair total)" : ""}`;
}

// Which entries survive the current toolbar state. Reads the module-level
// currentVariant for the FL2VA/Ref2VA cutoff (there is no per-call variant
// arg); everything else is a pure function of its arguments.
export function filterEntries(entries, { search, tag, favoritesOnly }) {
  const q = String(search || "").trim().toLowerCase();
  const unfiltered = q === "" && (tag || ALL_TAGS) === ALL_TAGS && !favoritesOnly;

  return (entries || []).filter((entry) => {
    // Variant cutoff first: an entry written before the schema migration has
    // no node_variant and is treated as "fl2va" (only that node existed
    // then). A legacy entry has no verbose at all -> also "fl2va".
    const sys = (entry.verbose && entry.verbose.system) || {};
    if ((sys.node_variant || "fl2va") !== currentVariant) return false;

    if (!entry.verbose) {
      // Nothing to match on when there is no verbose block -- a real legacy
      // entry, or a "normal" entry whose verbose.json is unreadable. Either
      // way show it only in the no-filters view (plan section 11.1 / 7).
      return unfiltered;
    }

    const user = (entry.verbose && entry.verbose.user) || {};
    const system = (entry.verbose && entry.verbose.system) || {};
    const tags = Array.isArray(user.tags) ? user.tags : [];

    const matchesSearch =
      q === "" ||
      String(user.name || "").toLowerCase().includes(q) ||
      String(system.prompt || "").toLowerCase().includes(q) ||
      String(user.notes || "").toLowerCase().includes(q) ||
      tags.some((t) => String(t).toLowerCase().includes(q));

    const matchesTag = (tag || ALL_TAGS) === ALL_TAGS || tags.includes(tag);
    const matchesFavorite = !favoritesOnly || user.favorite === true;

    return matchesSearch && matchesTag && matchesFavorite;
  });
}

export function allNormalTags(entries, variant) {
  const set = new Set();
  for (const entry of entries || []) {
    if (entry.classification !== "normal") continue;
    const entryVariant = (entry.verbose && entry.verbose.system && entry.verbose.system.node_variant) || "fl2va";
    if (entryVariant !== variant) continue;
    const tags = (entry.verbose && entry.verbose.user && entry.verbose.user.tags) || [];
    for (const t of tags) set.add(String(t));
  }
  return [...set].sort((a, b) => a.localeCompare(b));
}

// Legacy/inconsistent entries (no verbose, or classification !== "normal")
// have nothing meaningful to sort by in either mode -- they always sink to
// the end rather than landing above real entries by accident (e.g. a
// legacy row's entryLabel() fallback text sorting alphabetically before
// real prompts).
function normalCreatedAtTime(entry) {
  if (entry.classification !== "normal") return null;
  const value = entry.verbose && entry.verbose.system && entry.verbose.system.created_at;
  if (typeof value !== "string" || !value) return null;
  const t = new Date(value).getTime();
  return Number.isNaN(t) ? null : t;
}

function compareByDateDesc(a, b) {
  const ta = normalCreatedAtTime(a);
  const tb = normalCreatedAtTime(b);
  if (ta === null && tb === null) return 0;
  if (ta === null) return 1;
  if (tb === null) return -1;
  return tb - ta; // newest first
}

function compareByNameAsc(a, b) {
  const sortableA = a.classification === "normal";
  const sortableB = b.classification === "normal";
  if (!sortableA && !sortableB) return 0;
  if (!sortableA) return 1;
  if (!sortableB) return -1;
  return entryLabel(a).localeCompare(entryLabel(b), undefined, { sensitivity: "base", numeric: true });
}

export function sortEntries(entries, mode) {
  const copy = [...(entries || [])];
  copy.sort(mode === "name" ? compareByNameAsc : compareByDateDesc);
  return copy;
}

// --- dual-resolution pairing --------------------------------------------
//
// A dual-resolution node run (MiniMaxH3CLIPCached{FL2VA,Ref2VA}DualRes)
// encodes the same prompt at two target resolutions and lands on two
// separate cache entries. The backend cross-links them: each entry's
// verbose.system carries paired_fingerprint / paired_width / paired_height
// (the OTHER side's fingerprint and pixel size) and is_upscale_target
// (THIS entry's own role -- false = base resolution, true = upscale
// resolution). See verbose_store.add_pairing() and
// nodes._pair_verbose_entries().
//
// The Cache Manager folds a valid pair into a single visible row -- the
// base entry, carrying a "+ rescaled to WxH" badge that expands a small
// read-only strip for the upscale entry -- so the (identical) prompt is
// never listed twice. resolvePairing() decides, for one entry, whether
// that treatment applies. It returns a status object; the caller keys off
// `.status`:
//
//   "none"          No paired_fingerprint at all: an ordinary entry.
//   "valid"         Mutual pointer + opposite explicit roles. Also carries
//                   { partner, entryIsUpscale, partnerIsUpscale }.
//   "orphaned"      Has a paired_fingerprint, but the partner is not in
//                   `entriesByFingerprint` (deleted -- Delete does not
//                   cascade, by design) or no longer points back (a later
//                   dual-res run with a different second resolution
//                   repointed the base entry, stranding the old upscale
//                   entry with a stale one-way pointer).
//   "role-unknown"  Mutual pointer, but is_upscale_target is missing on one
//                   or both sides (a pair written before that flag existed)
//                   or the two roles are equal (should never happen). The
//                   role is not trustworthy, so both sides render as plain
//                   separate rows -- the role is NEVER guessed from
//                   paired_width * paired_height.
//   "inconsistent-pair"
//                   Mutual pointer confirmed, but at least one side is not a
//                   plain "normal" entry with a readable verbose sidecar (it
//                   is "inconsistent", "legacy", or its verbose.json failed
//                   to load). The pair is NOT folded: folding would paint a
//                   misleading "+ rescaled to WxH" badge for a partner that
//                   store.load_conditioning() would reject, and it would hide
//                   an otherwise-good "normal" side behind a base row drawn by
//                   buildInconsistentRow() (which knows nothing about
//                   pairing). Both sides render as their own explicit rows --
//                   the non-normal side through its legacy/inconsistent row,
//                   the normal side as an ordinary standalone entry.
//
// `entriesByFingerprint` MUST be built from the full entry list of the last
// /check, not from the post-search/tag/favorite subset -- otherwise an
// active text filter that hides the partner would make a real pair look
// orphaned.
export function resolvePairing(entry, entriesByFingerprint) {
  const system = (entry && entry.verbose && entry.verbose.system) || {};
  const partnerFingerprint = system.paired_fingerprint;
  if (typeof partnerFingerprint !== "string" || partnerFingerprint === "") {
    return { status: "none" };
  }

  const partner = entriesByFingerprint.get(partnerFingerprint);
  if (!partner) return { status: "orphaned" };

  const partnerSystem = (partner.verbose && partner.verbose.system) || {};
  if (partnerSystem.paired_fingerprint !== entry.fingerprint) {
    return { status: "orphaned" };
  }

  // Fold only when BOTH sides are ordinary "normal" entries with a readable
  // verbose sidecar. If either is "inconsistent" / "legacy" / has an
  // unreadable sidecar, folding would mislead (a "+ rescaled to" badge for an
  // unservable partner) or hide a good entry -- see the "inconsistent-pair"
  // note above. Render both sides plainly instead.
  const bothNormal =
    entry.classification === "normal" &&
    !!entry.verbose &&
    partner.classification === "normal" &&
    !!partner.verbose;
  if (!bothNormal) return { status: "inconsistent-pair" };

  const entryIsUpscale = system.is_upscale_target;
  const partnerIsUpscale = partnerSystem.is_upscale_target;
  if (typeof entryIsUpscale !== "boolean" || typeof partnerIsUpscale !== "boolean") {
    return { status: "role-unknown" };
  }
  if (entryIsUpscale === partnerIsUpscale) return { status: "role-unknown" };

  return { status: "valid", partner, entryIsUpscale, partnerIsUpscale };
}

// --- reference provenance ---------------------------------------------------
//
// system.ref_sources (Ref2VA only, added in the graph-provenance phase) maps
// a node INPUT SLOT NAME -- "ref_image_0", "ref_video_2", ... -- to the list
// of files traced back through the graph for that slot:
//
//   { "ref_image_0": [ { annotated: "foto.png", path: "/abs/input/foto.png" } ],
//     "ref_image_2": [ { annotated: "a.png", path: "..." },
//                      { annotated: "b.png", path: "..." } ] }
//
// The value is ALWAYS a list. More than one entry is normal, not an edge
// case: an asymmetric fan-in (an ImageBatch, a composite that also pulls in
// a mask) legitimately traces one slot to several files. `path` is omitted
// when folder_paths.get_annotated_filepath() rejected the value.
//
// The only key shared with system.references[i] is `slot`. The compacted
// `index` must NOT be used to line the two up: a gap in the wired slots
// (ref_image_0, ref_image_2, ref_image_5 -> indices 0, 1, 2) makes index and
// slot number diverge.
export function refSourcesForReference(ref, refSources) {
  const slot = ref && typeof ref.slot === "string" ? ref.slot : "";
  if (!slot || !refSources || typeof refSources !== "object") return [];
  const list = refSources[slot];
  if (!Array.isArray(list)) return [];
  return list.filter(
    (entry) => entry && typeof entry.annotated === "string" && entry.annotated !== "",
  );
}

// The short fingerprint(s) shown in the detail action bar. 12 hex chars --
// exactly what ComfyUI logs ("[CACHE HIT] 1390b9927e5b", proxy.py) -- so a
// value here pastes straight into a log search. A valid dual-resolution pair
// (resolvePairing) contributes two lines, its own and its partner's, each
// tagged "base" / "rescaled" so the two can be told apart; any other entry
// contributes just one, untagged.
export function detailFingerprintLines(entry, pairing) {
  const ownFull = (entry && entry.fingerprint) || "";
  const own = { role: "", full: ownFull, short: ownFull.slice(0, 12) };
  if (!pairing || pairing.status !== "valid" || !pairing.partner) return [own];

  const partnerFull = pairing.partner.fingerprint || "";
  own.role = pairing.entryIsUpscale ? "rescaled" : "base";
  return [
    own,
    {
      role: pairing.partnerIsUpscale ? "rescaled" : "base",
      full: partnerFull,
      short: partnerFull.slice(0, 12),
    },
  ];
}

// --- HTTP ----------------------------------------------------------------

async function fetchJson(path, options) {
  const response = await api.fetchApi(path, options);
  if (!response.ok) {
    let message = `HTTP ${response.status}`;
    try {
      const body = await response.json();
      if (body && body.error) message = body.error;
    } catch (_) {
      /* keep the HTTP status */
    }
    throw new Error(message);
  }
  return response.json();
}

function postUpdate(body) {
  return fetchJson(`${API_PREFIX}/update`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

// --- panel ------------------------------------------------------------------

function createPanel() {
  if (panel) return panel;
  injectStyles();

  const root = document.createElement("div");
  root.className = "h3cm-root";
  root.setAttribute("aria-hidden", "true");
  root.innerHTML = `
    <div class="h3cm-backdrop" data-h3cm-close></div>
    <section class="h3cm-modal" role="dialog" aria-modal="true" aria-label="MiniMax H3 Cache Manager">
      <header class="h3cm-header">
        <span class="h3cm-title">MiniMax H3 Cache Manager</span>
        <button type="button" class="h3cm-icon-button" data-h3cm-close aria-label="Close">✕</button>
      </header>
      <div class="h3cm-toolbar">
        <button type="button" class="h3cm-button" data-h3cm-check>Check</button>
        <div class="h3cm-variant-toggle" role="group" aria-label="Node variant">
          <button type="button" data-h3cm-variant="fl2va" class="h3cm-variant-btn is-active">FL2VA</button>
          <button type="button" data-h3cm-variant="ref2va" class="h3cm-variant-btn">Ref2VA</button>
        </div>
        <input type="text" class="h3cm-search" data-h3cm-search placeholder="Search…">
        <select class="h3cm-select" data-h3cm-sort aria-label="Sort by">
          <option value="date">Date</option>
          <option value="name">Name</option>
        </select>
        <select class="h3cm-select" data-h3cm-tag-filter aria-label="Filter by tag">
          <option value="">All tags</option>
        </select>
        <label class="h3cm-check-label">
          <input type="checkbox" data-h3cm-favorites-only> ★ Favorites only
        </label>
        <span class="h3cm-status" data-h3cm-status>Cache: — entries / —</span>
      </div>
      <div class="h3cm-list" data-h3cm-list></div>
      <section class="h3cm-detail" data-h3cm-detail hidden>
        <div class="h3cm-detail-head">
          <span class="h3cm-detail-title" data-h3cm-detail-title></span>
          <button type="button" class="h3cm-button" data-h3cm-detail-close>Close details</button>
        </div>
        <span class="h3cm-detail-created" data-h3cm-detail-created></span>
        <div class="h3cm-prompt-wrap">
          <div class="h3cm-prompt-toolbar">
            <span class="h3cm-refs-hint" data-h3cm-refs-hint></span>
            <button type="button" class="h3cm-prompt-copy" data-h3cm-prompt-copy
              title="Copy prompt" aria-label="Copy prompt">
              <svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true">
                <rect x="4" y="4" width="9" height="9" rx="1" fill="none" stroke="currentColor" stroke-width="1.3"/>
                <path d="M2.5 9.5V2.5A1 1 0 0 1 3.5 1.5H10.5" fill="none" stroke="currentColor" stroke-width="1.3"/>
              </svg>
            </button>
          </div>
          <pre class="h3cm-prompt" data-h3cm-detail-prompt></pre>
        </div>
        <div class="h3cm-detail-refs" data-h3cm-detail-refs hidden></div>
        <label class="h3cm-field">Name
          <input type="text" data-h3cm-edit-name>
        </label>
        <label class="h3cm-field">Notes
          <textarea rows="3" data-h3cm-edit-notes></textarea>
        </label>
        <label class="h3cm-field">Tags (comma-separated)
          <input type="text" data-h3cm-edit-tags>
        </label>
        <label class="h3cm-check-label">
          <input type="checkbox" data-h3cm-edit-favorite> Favorite
        </label>
        <div class="h3cm-detail-actions">
          <button type="button" class="h3cm-button" data-h3cm-save>Save</button>
          <button type="button" class="h3cm-button" data-h3cm-copy-prompt>Copy prompt</button>
          <button type="button" class="h3cm-button h3cm-danger" data-h3cm-delete>Delete</button>
          <span class="h3cm-detail-status" data-h3cm-detail-status></span>
          <div class="h3cm-detail-fingerprint" data-h3cm-detail-fingerprint></div>
        </div>
        <div class="h3cm-copy-result" data-h3cm-copy-result hidden></div>
      </section>
    </section>
  `;
  document.body.appendChild(root);

  panel = {
    root,
    statusEl: root.querySelector("[data-h3cm-status]"),
    listEl: root.querySelector("[data-h3cm-list]"),
    searchEl: root.querySelector("[data-h3cm-search]"),
    sortEl: root.querySelector("[data-h3cm-sort]"),
    tagFilterEl: root.querySelector("[data-h3cm-tag-filter]"),
    favoritesOnlyEl: root.querySelector("[data-h3cm-favorites-only]"),
    variantBtns: [...root.querySelectorAll("[data-h3cm-variant]")],
    detailEl: root.querySelector("[data-h3cm-detail]"),
  };

  root.querySelectorAll("[data-h3cm-close]").forEach((el) => el.addEventListener("click", closePanel));
  root.querySelector("[data-h3cm-check]").addEventListener("click", () => runCheck());
  panel.variantBtns.forEach((btn) => btn.addEventListener("click", () => switchVariant(btn.dataset.h3cmVariant)));
  panel.searchEl.addEventListener("input", renderList);
  panel.tagFilterEl.addEventListener("change", renderList);
  panel.favoritesOnlyEl.addEventListener("change", renderList);
  panel.sortEl.addEventListener("change", renderList);
  root.querySelector("[data-h3cm-detail-close]").addEventListener("click", closeDetail);
  root.querySelector("[data-h3cm-save]").addEventListener("click", saveDetail);
  root.querySelector("[data-h3cm-edit-favorite]").addEventListener("change", onDetailFavoriteChange);
  root.querySelector("[data-h3cm-copy-prompt]").addEventListener("click", copyPrompt);
  root.querySelector("[data-h3cm-delete]").addEventListener("click", onDetailDeleteClick);
  root
    .querySelector("[data-h3cm-prompt-copy]")
    .addEventListener("click", (event) => copyPromptText(event.currentTarget));

  return panel;
}

function openPanel() {
  const p = createPanel();
  p.root.classList.add("is-open");
  p.root.setAttribute("aria-hidden", "false");
  runCheck();
}

function closePanel() {
  if (!panel) return;
  panel.root.classList.remove("is-open");
  panel.root.setAttribute("aria-hidden", "true");
}

// Switch which node's entries the list shows. Purely client-side: /check
// already returned every entry, so this only changes the filter and
// re-renders. By prior decision the toolbar filters are NOT kept per tab --
// switching resets search / tag / favorites / sort to their defaults and
// closes any open detail panel, so each tab always opens on a clean,
// unfiltered view.
function switchVariant(variant) {
  if (variant !== "fl2va" && variant !== "ref2va") return;
  currentVariant = variant;
  panel.variantBtns.forEach((btn) =>
    btn.classList.toggle("is-active", btn.dataset.h3cmVariant === variant));

  panel.searchEl.value = "";
  panel.tagFilterEl.value = ALL_TAGS;
  panel.favoritesOnlyEl.checked = false;
  panel.sortEl.value = "date";
  refreshTagFilterOptions(); // rebuild the dropdown from the new variant's tags
  if (openDetailFingerprint) closeDetail();

  renderList();
}

// --- list rendering -------------------------------------------------------

function refreshTagFilterOptions() {
  const { tagFilterEl } = panel;
  const previous = tagFilterEl.value;
  const tags = lastCheckResult ? allNormalTags(lastCheckResult.entries, currentVariant) : [];

  tagFilterEl.innerHTML = '<option value="">All tags</option>';
  for (const tag of tags) {
    const option = document.createElement("option");
    option.value = tag;
    option.textContent = tag;
    tagFilterEl.appendChild(option);
  }
  // Keep the previous selection if that tag still exists.
  tagFilterEl.value = tags.includes(previous) ? previous : ALL_TAGS;
}

function revokeThumbnailUrls() {
  for (const url of objectUrls) URL.revokeObjectURL(url);
  objectUrls = [];
}

async function loadThumbnail(imgEl, fingerprint, index, generation, bucket = objectUrls) {
  try {
    const response = await api.fetchApi(
      `${API_PREFIX}/thumbnail?fingerprint=${encodeURIComponent(fingerprint)}&index=${encodeURIComponent(index)}`,
    );
    if (!response.ok || generation !== renderGeneration) return;
    const blob = await response.blob();
    if (generation !== renderGeneration) return; // list re-rendered meanwhile
    const url = URL.createObjectURL(blob);
    bucket.push(url);
    imgEl.src = url;
    imgEl.classList.add("is-loaded");
  } catch (_) {
    /* leave the placeholder in place */
  }
}

function buildTagChips(tags) {
  const wrap = document.createElement("span");
  wrap.className = "h3cm-chips";
  for (const tag of tags) {
    const chip = document.createElement("span");
    chip.className = "h3cm-chip";
    chip.textContent = tag;
    wrap.appendChild(chip);
  }
  return wrap;
}

function buildThumbnails(fingerprint, references, generation) {
  const wrap = document.createElement("span");
  wrap.className = "h3cm-thumbs";
  for (const ref of references) {
    const img = document.createElement("img");
    img.className = "h3cm-thumb";
    img.alt = ref.label || `reference ${ref.index}`;
    img.title = ref.label || "";
    wrap.appendChild(img);
    loadThumbnail(img, fingerprint, ref.index, generation);
  }
  return wrap;
}

// Ref2VA rows can carry up to 15 references, so the row shows only the first
// three (thumbnail for image/video, a small "audio" pill for audio, which
// never has a thumbnail) and a "+N more" count for the rest. The full,
// positionally-labelled breakdown lives in the detail panel.
function buildRef2vaThumbnails(fingerprint, references, generation) {
  const wrap = document.createElement("span");
  wrap.className = "h3cm-thumbs";
  for (const ref of references.slice(0, 3)) {
    if ((ref.type || "image") === "audio") {
      const pill = document.createElement("span");
      pill.className = "h3cm-audio-pill";
      pill.textContent = "audio";
      wrap.appendChild(pill);
      continue;
    }
    const img = document.createElement("img");
    img.className = "h3cm-thumb";
    img.alt = `reference ${ref.index}`;
    wrap.appendChild(img);
    loadThumbnail(img, fingerprint, ref.index, generation);
  }
  if (references.length > 3) {
    const more = document.createElement("span");
    more.className = "h3cm-thumbs-more";
    more.textContent = `+${references.length - 3} more`;
    wrap.appendChild(more);
  }
  return wrap;
}

// Shared skeleton for the two row kinds that have no detail panel (legacy
// and inconsistent): a truncated fingerprint, a status badge, a one-line
// hint, and a Delete button. Both kinds still correspond to real
// .json/.safetensors files the user may want to remove (plan section 15
// makes no exception), so the Delete button and its stopPropagation +
// deleteEntry listener are identical; only the row/badge/hint classes and
// the badge/hint text differ.
function buildSimpleRow(entry, { rowClass, badgeClass, badgeText, hintClass, hintText }) {
  const row = document.createElement("div");
  row.className = `h3cm-row ${rowClass}`;

  const fp = document.createElement("span");
  fp.className = "h3cm-fp";
  fp.textContent = `${String(entry.fingerprint || "").slice(0, 12)}…`;

  const badge = document.createElement("span");
  badge.className = `h3cm-badge ${badgeClass}`;
  badge.textContent = badgeText;

  const hint = document.createElement("span");
  hint.className = hintClass;
  hint.textContent = hintText;

  const del = document.createElement("button");
  del.type = "button";
  del.className = "h3cm-button h3cm-danger h3cm-row-delete";
  del.textContent = "Delete";
  del.addEventListener("click", (event) => {
    event.stopPropagation();
    deleteEntry(entry.fingerprint, null);
  });

  row.append(fp, badge, hint);
  // A legacy or inconsistent row has no meta line to hang the size off, but
  // these entries can be the largest ones on disk and Delete is usually the
  // only thing anyone does with them -- so the size goes next to the hint.
  // Never a pair total: neither classification is ever folded into a pair
  // (resolvePairing() rejects a partner that is not a readable "normal"
  // entry), so what is shown here is exactly what this Delete frees.
  const sizeBytes = entryOwnSizeBytes(entry);
  if (sizeBytes) {
    const size = document.createElement("span");
    size.className = "h3cm-row-size";
    size.textContent = formatBytes(sizeBytes);
    row.appendChild(size);
  }
  row.appendChild(del);
  return row;
}

function buildLegacyRow(entry) {
  return buildSimpleRow(entry, {
    rowClass: "is-legacy",
    badgeClass: "h3cm-badge-legacy",
    badgeText: "legacy",
    hintClass: "h3cm-legacy-hint",
    hintText: "Use this entry once to populate details",
  });
}

function buildInconsistentRow(entry) {
  const hintText = {
    missing_json: "Core JSON is missing",
    json_unreadable: "Core JSON cannot be read",
    invalid_json_envelope: "Core JSON has an invalid envelope",
    missing_safetensors: "Tensor file is missing",
    safetensors_unreadable: "Tensor header cannot be read",
    generation_mismatch: "Interrupted refresh: generation IDs do not match",
  }[entry.reason] || "Core cache files are inconsistent";
  return buildSimpleRow(entry, {
    rowClass: "is-inconsistent",
    badgeClass: "h3cm-badge-inconsistent",
    badgeText: "inconsistent",
    hintClass: "h3cm-inconsistent-hint",
    hintText,
  });
}

// The base row of a valid dual-resolution pair carries a "+ rescaled to WxH"
// toggle. WxH is the partner (upscale) entry's pixel size, which the backend
// mirrored onto this entry as paired_width / paired_height. Clicking the
// toggle expands one compact, read-only strip describing the partner -- its
// short fingerprint, date, pixel size, and a Delete button acting on the
// PARTNER's fingerprint (the same deleteEntry() path as every other Delete,
// only a different target). The prompt is deliberately never repeated here:
// the two entries share it by construction.
function appendRescaledBadge(row, baseSystem, partner) {
  const rescaledTo = formatGenerationSize({
    width: baseSystem.paired_width,
    height: baseSystem.paired_height,
  });

  const toggle = document.createElement("button");
  toggle.type = "button";
  toggle.className = "h3cm-badge h3cm-badge-paired h3cm-pair-toggle";
  toggle.textContent = rescaledTo ? `+ rescaled to ${rescaledTo}` : "+ rescaled";
  toggle.setAttribute("aria-expanded", "false");

  const strip = document.createElement("div");
  strip.className = "h3cm-pair-strip";
  strip.hidden = true;

  const partnerSystem = (partner.verbose && partner.verbose.system) || {};

  const fp = document.createElement("span");
  fp.className = "h3cm-fp";
  fp.textContent = `${String(partner.fingerprint || "").slice(0, 12)}…`;

  const meta = document.createElement("span");
  meta.className = "h3cm-pair-strip-meta";
  meta.textContent = formatEntryMetaLine(partnerSystem, entryOwnSizeBytes(partner));

  const del = document.createElement("button");
  del.type = "button";
  del.className = "h3cm-button h3cm-danger h3cm-row-delete";
  del.textContent = "Delete";
  del.addEventListener("click", (event) => {
    event.stopPropagation();
    deleteEntry(partner.fingerprint, null);
  });

  strip.append(fp, meta, del);

  toggle.addEventListener("click", (event) => {
    event.stopPropagation();
    const opening = strip.hidden;
    strip.hidden = !opening;
    toggle.classList.toggle("is-open", opening);
    toggle.setAttribute("aria-expanded", opening ? "true" : "false");
  });

  row.append(toggle, strip);
}

function buildNormalRow(entry, generation, lastUsedFingerprint, pairing = null) {
  const user = (entry.verbose && entry.verbose.user) || {};
  const system = (entry.verbose && entry.verbose.system) || {};
  const tags = Array.isArray(user.tags) ? user.tags : [];
  const references = Array.isArray(system.references) ? system.references : [];

  const row = document.createElement("div");
  row.className = "h3cm-row is-normal";
  row.dataset.fingerprint = entry.fingerprint; // so attachDetailAfterRow() can find this row
  // After a dual-resolution run the freshly saved fingerprint is the upscale
  // side, which has no row of its own -- so the visible base row also lights
  // up when the "last used" fingerprint is its hidden pair partner.
  const isLastUsed =
    !!lastUsedFingerprint &&
    (entry.fingerprint === lastUsedFingerprint ||
      (pairing &&
        pairing.status === "valid" &&
        pairing.partner.fingerprint === lastUsedFingerprint));
  if (isLastUsed) {
    row.classList.add("is-last-used");
  }

  const star = document.createElement("button");
  star.type = "button";
  star.className = "h3cm-star";
  star.dataset.h3cmFavoriteToggle = entry.fingerprint;
  star.textContent = user.favorite === true ? "★" : "☆";
  star.classList.toggle("is-on", user.favorite === true);
  star.title = user.favorite === true ? "Remove from favorites" : "Add to favorites";
  star.addEventListener("click", (event) => {
    event.stopPropagation();
    toggleFavorite(entry.fingerprint, user.favorite === true);
  });

  const label = document.createElement("span");
  label.className = "h3cm-row-label";
  label.textContent = entryLabel(entry);

  const size = entryDisplaySize(entry, pairing);
  const created = document.createElement("span");
  created.className = "h3cm-row-created";
  created.textContent = formatEntryMetaLine(system, size.bytes, size);
  created.title = entryMetaTooltip(system, size);

  row.append(star, label, created);
  if (tags.length) row.appendChild(buildTagChips(tags));
  if (references.length) {
    const variant = system.node_variant || "fl2va";
    row.appendChild(
      variant === "ref2va"
        ? buildRef2vaThumbnails(entry.fingerprint, references, generation)
        : buildThumbnails(entry.fingerprint, references, generation),
    );
  }

  if (pairing && pairing.status === "valid" && !pairing.entryIsUpscale) {
    // Valid pair, base side: the expandable "+ rescaled to WxH" strip.
    appendRescaledBadge(row, system, pairing.partner);
  } else if (pairing && pairing.status === "orphaned") {
    // Had a pairing pointer, but the partner is gone or no longer points
    // back. Render as a normal, fully visible row plus a warning tag so the
    // user can judge it by hand -- Delete is not cascaded (deliberate).
    // NOTE: a "role-unknown" pairing gets NO badge -- it is not broken,
    // only missing the is_upscale_target flag (a pre-flag pair).
    const badge = document.createElement("span");
    badge.className = "h3cm-badge h3cm-badge-orphaned";
    badge.textContent = "⚠ pairing partner missing";
    badge.title =
      "Created by a dual-resolution run, but the paired entry is no longer "
      + "cross-linked (deleted, or re-paired by a later run). Shown as a "
      + "normal entry -- delete it here if you no longer need it.";
    row.appendChild(badge);
  }

  row.addEventListener("click", () => openDetail(entry.fingerprint));
  return row;
}

// renderList() clears panel.listEl, and attachDetailAfterRow() parents the one
// detail node *inside* a row, so every re-render physically removes the detail
// panel from the DOM. Any caller that re-renders on its own (the search / tag /
// favorite listeners go straight to renderList(), not through runCheck()) would
// otherwise leave the panel orphaned. Re-attach it here, after the rows exist:
// if a row was actually rendered for its entry, rebuild + re-attach it;
// otherwise close it.
//
// The re-attach must key off a row *actually being in the DOM*, not off the
// entry surviving the filter: renderList() emits no row for the upscale side
// of a valid dual-resolution pair, so an open detail panel whose entry has
// just become that hidden side (a later dual-res run paired it) is still in
// `filtered` but has nowhere to anchor. Anchoring is what attachDetailAfterRow()
// now reports back -- a false return means "no row", i.e. close the panel
// rather than leave it detached from the DOM showing stale data.
function reattachOpenDetailAfterRender() {
  if (!openDetailFingerprint) return;
  const entry = findNormalEntry(openDetailFingerprint);
  if (entry) {
    // Background re-render: keep any unsaved name / notes / tags / favorite
    // edit the user has in progress -- only openDetail() (an explicit click)
    // resets those.
    populateDetail(entry, { preserveEditableFields: true });
  }
  if (!entry || !attachDetailAfterRow(openDetailFingerprint)) {
    closeDetail();
  }
}

function renderList() {
  if (!panel || !lastCheckResult) return;

  renderGeneration += 1;
  const generation = renderGeneration;
  revokeThumbnailUrls();

  const state = {
    search: panel.searchEl.value,
    tag: panel.tagFilterEl.value,
    favoritesOnly: panel.favoritesOnlyEl.checked,
  };
  const entries = lastCheckResult.entries || [];
  // Indexed over the FULL entry set, not `filtered`: resolvePairing() must be
  // able to see a pair partner even while an active search / tag filter hides
  // it, otherwise a real pair would look orphaned (see resolvePairing()).
  const entriesByFingerprint = new Map();
  for (const e of entries) {
    if (e && typeof e.fingerprint === "string") entriesByFingerprint.set(e.fingerprint, e);
  }
  const lastUsedFingerprint = (lastCheckResult.last_used || {})[currentVariant] || null;
  const filtered = sortEntries(filterEntries(entries, state), panel.sortEl.value);

  panel.listEl.innerHTML = "";

  if (filtered.length === 0) {
    const empty = document.createElement("div");
    empty.className = "h3cm-empty";
    empty.textContent = entries.length === 0
      ? "No cache entries found."
      : "No entries match the current search / filters.";
    panel.listEl.appendChild(empty);
    reattachOpenDetailAfterRender();
    return;
  }

  for (const entry of filtered) {
    if (entry.classification === "inconsistent") {
      panel.listEl.appendChild(buildInconsistentRow(entry));
      continue;
    }
    if (!entry.verbose) {
      // No verbose block -> the simplified row. That covers a real legacy
      // entry (predates the sidecar) AND a "normal" entry whose verbose.json
      // is unreadable (load_verbose() returned null): buildNormalRow() would
      // give it a detail panel / favorite / save that all 404, so render it
      // the same stripped-down way. _sync_verbose_metadata() backfills both
      // cases identically the next time the entry is used.
      panel.listEl.appendChild(buildLegacyRow(entry));
      continue;
    }
    const pairing = resolvePairing(entry, entriesByFingerprint);
    // The upscale side of a valid dual-resolution pair has no row of its
    // own: its data is reached by expanding the "+ rescaled to" badge on
    // the base row (buildNormalRow -> appendRescaledBadge).
    if (pairing.status === "valid" && pairing.entryIsUpscale) continue;
    panel.listEl.appendChild(
      buildNormalRow(entry, generation, lastUsedFingerprint, pairing),
    );
  }

  reattachOpenDetailAfterRender();
}

// --- cache size threshold --------------------------------------------------
//
// Display only: the limit lives in the browser and the backend never learns
// about it. Nothing is evicted, no write is blocked -- the status line just
// changes colour once the cache passes the configured size.
//
// The figure being judged is "total_size_bytes" from /check, i.e. the whole
// cache directory (both node variants, thumbnails and stray files included),
// which is exactly the number the status line already prints. There is no
// per-variant equivalent to compare against: scan_cache() measures the
// directory as a whole (minimaxh3_clipcache/scanner.py, _dir_size_bytes) and
// the per-entry "size_bytes" values deliberately do not add up to it.
//
// This turn ships no UI for the values; they are written into localStorage
// by hand.

const CACHE_SIZE_OPTIONS_KEY = "h3cm-cache-size-options";
const WARNING_PERCENT_MIN = 1;
const WARNING_PERCENT_MAX = 100;

// Read the persisted { limitBytes, warningPercent }. Returns null on any
// problem -- missing key, malformed JSON, a missing or unusable field, or
// localStorage being unavailable (private mode, storage disabled) -- so the
// caller falls back to "no threshold configured" instead of throwing. Same
// shape of guard as readLauncherPosition().
//
// A limitBytes of 0 is accepted and passed through: classifyCacheSize() reads
// it as the off switch. A negative one is rejected here as malformed data,
// which reaches the same "off" result by the other route.
export function readCacheSizeOptions() {
  try {
    const raw = window.localStorage.getItem(CACHE_SIZE_OPTIONS_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed !== "object") return null;
    const limitBytes = parseFiniteNumber(parsed.limitBytes, { min: 0 });
    const warningPercent = parseFiniteNumber(parsed.warningPercent, {
      min: WARNING_PERCENT_MIN,
      max: WARNING_PERCENT_MAX,
    });
    if (limitBytes === null || warningPercent === null) return null;
    return { limitBytes, warningPercent };
  } catch (err) {
    return null;
  }
}

export function writeCacheSizeOptions(options) {
  try {
    window.localStorage.setItem(CACHE_SIZE_OPTIONS_KEY, JSON.stringify(options));
  } catch (err) {
    /* storage unavailable / over quota -- the options just won't persist */
  }
}

// "off" | "ok" | "warning" | "alert" for a cache of `totalBytes` under
// `options`. Pure: no DOM, no storage, no clamping of its inputs.
//
// Both thresholds compare with ">=", so landing exactly on one counts as
// reaching it rather than as staying below it.
export function classifyCacheSize(totalBytes, options) {
  if (!options) return "off";
  const limitBytes = parseFiniteNumber(options.limitBytes);
  const warningPercent = parseFiniteNumber(options.warningPercent, {
    min: WARNING_PERCENT_MIN,
    max: WARNING_PERCENT_MAX,
  });
  if (limitBytes === null || warningPercent === null) return "off";
  if (limitBytes <= 0) return "off"; // zero / negative -- no limit configured
  // A /check payload without a usable total (an older build, a truncated
  // response) is treated the way the entry count already is: nothing to
  // report. formatBytes() prints "—" for it in the same line.
  const total = parseFiniteNumber(totalBytes, { min: 0 });
  if (total === null) return "off";
  if (total >= limitBytes) return "alert";
  if (total >= (limitBytes * warningPercent) / 100) return "warning";
  return "ok";
}

const CACHE_STATUS_LEVEL_CLASS = {
  warning: "h3cm-status-warning",
  alert: "h3cm-status-alert",
};
const CACHE_STATUS_LEVEL_PREFIX = {
  warning: "⚠ ",
  alert: "ALERT: ",
};

// The only writer of the panel's status line. Text and threshold level are
// applied together on purpose: kept apart, a level left over from the
// previous /check outlives the text it belonged to and paints an unrelated
// message ("Cache: checking…", "Delete failed (…)") red. Every path
// that is not a successful /check passes "off", which clears it.
function setCacheStatus(text, level) {
  const el = panel.statusEl;
  el.textContent = (CACHE_STATUS_LEVEL_PREFIX[level] || "") + text;
  for (const cls of Object.values(CACHE_STATUS_LEVEL_CLASS)) el.classList.remove(cls);
  const active = CACHE_STATUS_LEVEL_CLASS[level];
  if (active) el.classList.add(active);
}

// --- check ---------------------------------------------------------------

async function runCheck() {
  if (!panel) return;
  const generation = ++checkGeneration;
  setCacheStatus("Cache: checking…", "off");

  try {
    const data = await fetchJson(`${API_PREFIX}/check`);
    if (generation !== checkGeneration) return;
    lastCheckResult = data;
    const count = typeof data.total_count === "number" ? data.total_count : "—";
    setCacheStatus(
      `Cache: ${count} entries / ${formatBytes(data.total_size_bytes)}`,
      classifyCacheSize(data.total_size_bytes, readCacheSizeOptions()),
    );

    refreshTagFilterOptions();
    renderList(); // also re-attaches the open detail panel, or closes it when
    // that entry has no rendered row -- see reattachOpenDetailAfterRender()
  } catch (err) {
    if (generation !== checkGeneration) return;
    lastCheckResult = null;
    setCacheStatus(`Cache: check failed (${err && err.message ? err.message : err})`, "off");
    panel.listEl.innerHTML = "";
  }
}

// --- favorite toggle (row) ----------------------------------------------

async function toggleFavorite(fingerprint, currentlyFavorite) {
  try {
    await postUpdate({ fingerprint, favorite: !currentlyFavorite });
    await runCheck(); // re-render from the real state, never patch JS state
  } catch (err) {
    setCacheStatus(`Update failed (${err && err.message ? err.message : err})`, "off");
  }
}

// The "Favorite" checkbox in the detail panel saves the moment it is toggled,
// the same way the row star does -- it no longer waits for the "Save" button.
// "Save" still sends favorite along with name / notes / tags, but by then it
// is only re-sending the value already persisted here, so there is no
// conflict. On a failed save the checkbox is reverted to match reality.
async function onDetailFavoriteChange(event) {
  if (!openDetailFingerprint) return;
  const checkbox = event.currentTarget;
  const statusEl = panel.detailEl.querySelector("[data-h3cm-detail-status]");
  try {
    await postUpdate({ fingerprint: openDetailFingerprint, favorite: checkbox.checked });
    await runCheck(); // re-renders the row star + re-populates this same detail panel
  } catch (err) {
    checkbox.checked = !checkbox.checked; // revert - the save didn't happen
    statusEl.textContent = `Update failed (${err && err.message ? err.message : err})`;
  }
}

// --- detail + edit panel -------------------------------------------------

function findNormalEntry(fingerprint) {
  if (!lastCheckResult) return null;
  return (lastCheckResult.entries || []).find(
    (e) => e.fingerprint === fingerprint && e.classification === "normal" && e.verbose,
  ) || null;
}

// resolvePairing() needs to look up a pair partner by fingerprint over the
// FULL last-/check entry set (renderList() builds the same map for the list
// rows). The detail panel builds it here so it can resolve pairing for the
// open entry without depending on renderList() having run.
function entriesByFingerprintFromLastCheck() {
  const map = new Map();
  const entries = (lastCheckResult && lastCheckResult.entries) || [];
  for (const e of entries) {
    if (e && typeof e.fingerprint === "string") map.set(e.fingerprint, e);
  }
  return map;
}

// Paint the short-fingerprint block in the detail action bar from
// detailFingerprintLines() -- one <code> per line, each optionally tagged
// with its "base" / "rescaled" role for a dual-resolution pair.
function renderDetailFingerprint(container, lines) {
  container.innerHTML = "";
  for (const line of lines) {
    const row = document.createElement("span");
    row.className = "h3cm-detail-fp-line";
    if (line.role) {
      const role = document.createElement("span");
      role.className = "h3cm-detail-fp-role";
      role.textContent = line.role;
      row.appendChild(role);
    }
    const value = document.createElement("code");
    value.className = "h3cm-fp";
    value.textContent = line.short;
    if (line.full) value.title = line.full;
    row.appendChild(value);
    container.appendChild(row);
  }
}

// Ref2VA only: the "<Picture N>" / "<Video N>" / "<Audio N>" tags in the
// prompt are positional, counted per type over the reference list in order.
// The count is always derived here in JS, never read from a stored field.
const REF_TYPE_LABEL = { image: "Picture", video: "Video", audio: "Audio" };

// The render model for the Ref2VA detail grid: one cell per reference, in
// order, carrying the positional label ("Picture 2", counted per type here
// and nowhere else) and the file-provenance entries joined from
// system.ref_sources by slot name. Kept pure and separate from the DOM
// painting in renderDetailRefs() so the join and the counting are testable.
export function detailRefCells(references, refSources) {
  const counters = { image: 0, video: 0, audio: 0 };
  return references.map((ref) => {
    const type = ref.type || "image";
    counters[type] = (counters[type] || 0) + 1;
    return {
      type,
      index: ref.index,
      posLabel: `${REF_TYPE_LABEL[type] || "Reference"} ${counters[type]}`,
      sources: refSourcesForReference(ref, refSources),
    };
  });
}

// The file-provenance block under one reference's position label. Always
// rendered, even with no sources: an explicit muted "no file source" tells a
// correctly file-less reference (VAEDecode / EmptyImage output) apart from a
// trace that failed. The raw slot name (ref_image_5) is never shown -- it is
// the fifth input but perhaps the third picture, which misleads. The file
// name is clipped to a single line in the tile, so the tooltip carries the
// full name and, when the graph trace resolved one, the absolute path on a
// second line; that path is the value a click copies (via
// copyToClipboardWithFeedback). A source with no resolved path shows and
// tooltips its annotated name only and is not clickable.
function buildRefSourceLines(sources) {
  const wrap = document.createElement("div");
  wrap.className = "h3cm-detail-ref-sources";

  if (sources.length === 0) {
    const empty = document.createElement("span");
    empty.className = "h3cm-detail-ref-file is-empty";
    empty.textContent = "no file source";
    wrap.appendChild(empty);
    return wrap;
  }

  for (const source of sources) {
    const line = document.createElement("span");
    line.className = "h3cm-detail-ref-file";
    line.textContent = source.annotated;

    const path = typeof source.path === "string" ? source.path : "";
    // Full name always in the tooltip (the tile clips it); the path, when
    // resolved, on a second line. This same string is what the title reverts
    // to after the transient "Copied!" -- passing the bare path there would
    // drop the name from the tooltip on the first copy.
    const fullTitle = path ? `${source.annotated}\n${path}` : source.annotated;
    line.title = fullTitle;
    if (path) {
      line.classList.add("is-copyable");
      line.setAttribute("role", "button");
      line.setAttribute("tabindex", "0");
      const copy = () => copyToClipboardWithFeedback(line, path, fullTitle);
      line.addEventListener("click", copy);
      line.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          copy();
        }
      });
    }
    wrap.appendChild(line);
  }
  return wrap;
}

function renderDetailRefs(container, fingerprint, variant, references, refSources) {
  container.innerHTML = "";
  // Detail thumbnails live in their own blob-URL pool (see the declaration of
  // detailRefObjectUrls): revoke the previous set here, where the <img>
  // elements holding them have just been discarded, and nowhere else.
  for (const url of detailRefObjectUrls) URL.revokeObjectURL(url);
  detailRefObjectUrls = [];
  // FL2VA is unchanged: its first_frame / last_frame already show in the list
  // row and, on a Copy-prompt click, in the result box below -- no persistent
  // breakdown here. Only Ref2VA gets one.
  if (variant !== "ref2va" || references.length === 0) {
    container.hidden = true;
    return;
  }
  container.hidden = false;

  const heading = document.createElement("div");
  heading.className = "h3cm-detail-refs-head";
  heading.textContent = `${references.length} reference${references.length === 1 ? "" : "s"}`;
  container.appendChild(heading);

  const grid = document.createElement("div");
  grid.className = "h3cm-detail-refs-grid";

  for (const cellModel of detailRefCells(references, refSources)) {
    const cell = document.createElement("div");
    cell.className = "h3cm-detail-ref-cell";

    if (cellModel.type === "audio") {
      const pill = document.createElement("span");
      pill.className = "h3cm-audio-pill";
      pill.textContent = "audio";
      cell.appendChild(pill);
    } else {
      const img = document.createElement("img");
      img.className = "h3cm-thumb";
      img.alt = cellModel.posLabel;
      cell.appendChild(img);
      loadThumbnail(img, fingerprint, cellModel.index, renderGeneration, detailRefObjectUrls);
    }

    const cap = document.createElement("span");
    cap.className = "h3cm-detail-ref-label";
    cap.textContent = cellModel.posLabel;
    cell.appendChild(cap);

    cell.appendChild(buildRefSourceLines(cellModel.sources));
    grid.appendChild(cell);
  }
  container.appendChild(grid);
}

// preserveEditableFields: when true, the name / notes / tags inputs are
// left exactly as the user has them and only the read-only parts
// (title, prompt, references) are refreshed from server data. The favorite
// checkbox is deliberately excluded from this protection -- it is not a
// draft, it saves to the server the instant it is toggled (see
// onDetailFavoriteChange), so it must always mirror the server state.
// Every background re-render funnels through reattachOpenDetailAfterRender() ->
// populateDetail(), for reasons unrelated to the detail panel's own data
// (a search keystroke, a tag-filter change, the favorites toggle, the
// runCheck() after Save or a favorite toggle) -- without this guard each of
// those silently discards an in-progress, unsaved edit. openDetail() (an
// explicit click onto a row, possibly a different entry) passes it false so
// everything resets from scratch.
function populateDetail(entry, { preserveEditableFields = false } = {}) {
  const { detailEl } = panel;
  const user = (entry.verbose && entry.verbose.user) || {};
  const system = (entry.verbose && entry.verbose.system) || {};

  // One resolution for the whole panel: the meta line's byte figure and the
  // fingerprint list must describe the same pair.
  const pairing = resolvePairing(entry, entriesByFingerprintFromLastCheck());

  detailEl.querySelector("[data-h3cm-detail-title]").textContent = entryLabel(entry);
  const detailCreated = detailEl.querySelector("[data-h3cm-detail-created]");
  const size = entryDisplaySize(entry, pairing);
  detailCreated.textContent = formatEntryMetaLine(system, size.bytes, size);
  detailCreated.title = entryMetaTooltip(system, size);
  detailEl.querySelector("[data-h3cm-detail-prompt]").textContent = system.prompt || "(no prompt)";
  renderDetailRefs(
    detailEl.querySelector("[data-h3cm-detail-refs]"),
    entry.fingerprint,
    system.node_variant || "fl2va",
    Array.isArray(system.references) ? system.references : [],
    system.ref_sources && typeof system.ref_sources === "object" ? system.ref_sources : {},
  );
  renderDetailFingerprint(
    detailEl.querySelector("[data-h3cm-detail-fingerprint]"),
    detailFingerprintLines(entry, pairing),
  );
  if (!preserveEditableFields) {
    detailEl.querySelector("[data-h3cm-edit-name]").value = user.name || "";
    detailEl.querySelector("[data-h3cm-edit-notes]").value = user.notes || "";
    detailEl.querySelector("[data-h3cm-edit-tags]").value = (Array.isArray(user.tags) ? user.tags : []).join(", ");
  }
  detailEl.querySelector("[data-h3cm-edit-favorite]").checked = user.favorite === true;
  detailEl.querySelector("[data-h3cm-detail-status]").textContent = "";
  detailEl.hidden = false;
}

// Move the single detail node so it sits right after the row it describes,
// and report whether that row exists. .after() detaches the node from
// wherever it currently is first, so there is never a duplicate -- it is
// always the same one node from the template. A false return means the list
// has no row for this fingerprint (see reattachOpenDetailAfterRender()).
function attachDetailAfterRow(fingerprint) {
  const rowEl = panel.listEl.querySelector(
    `[data-fingerprint="${CSS.escape(fingerprint)}"]`,
  );
  if (rowEl) rowEl.after(panel.detailEl);
  return rowEl !== null;
}

function openDetail(fingerprint) {
  const entry = findNormalEntry(fingerprint);
  if (!entry) return; // legacy / missing -- nothing to show
  openDetailFingerprint = fingerprint;
  resetCopyUI(); // fresh entry -> drop any leftover "Copy prompt" result
  populateDetail(entry);
  attachDetailAfterRow(fingerprint);
  panel.detailEl.scrollIntoView({ block: "nearest" });
}

function closeDetail() {
  openDetailFingerprint = null;
  if (!panel) return;
  resetCopyUI();
  panel.detailEl.hidden = true;
}

function revokeCopyResultUrls() {
  for (const url of copyResultObjectUrls) URL.revokeObjectURL(url);
  copyResultObjectUrls = [];
}

// Clear the "Copy prompt" result box. Called when the detail panel switches
// entries or closes -- NOT on a plain runCheck() refresh, so a copy result
// stays visible until the user acts.
function resetCopyUI() {
  const resultEl = panel.detailEl.querySelector("[data-h3cm-copy-result]");
  revokeCopyResultUrls();
  resultEl.innerHTML = "";
  resultEl.hidden = true;
  // The reference hint is set only by a Copy-prompt click, so clear it when
  // the panel switches entries or closes -- it must never leak onto a
  // freshly opened entry.
  panel.detailEl.querySelector("[data-h3cm-refs-hint]").textContent = "";
}

async function saveDetail() {
  if (!openDetailFingerprint) return;
  const { detailEl } = panel;
  const statusEl = detailEl.querySelector("[data-h3cm-detail-status]");

  const body = {
    fingerprint: openDetailFingerprint,
    name: detailEl.querySelector("[data-h3cm-edit-name]").value,
    notes: detailEl.querySelector("[data-h3cm-edit-notes]").value,
    tags: parseTags(detailEl.querySelector("[data-h3cm-edit-tags]").value),
    favorite: detailEl.querySelector("[data-h3cm-edit-favorite]").checked,
  };

  statusEl.textContent = "Saving…";
  try {
    await postUpdate(body);
    await runCheck(); // re-renders list + re-populates this same detail panel
    panel.detailEl.querySelector("[data-h3cm-detail-status]").textContent = "Saved.";
  } catch (err) {
    statusEl.textContent = `Save failed (${err && err.message ? err.message : err})`;
  }
}

// --- Copy prompt: put the prompt on the clipboard ---------------------
//
// Writing the prompt straight into a graph widget was tried and dropped:
// app.graph.findNodesByType() does not descend into subgraphs, and a
// "prompt" converted from a widget to an input has no widget to set. The
// clipboard works regardless of graph structure. Only classification="normal"
// reaches here (legacy has no detail panel and no "Copy prompt" button).

async function loadCopyResultThumbnail(imgEl, linkEl, dimsEl, fingerprint, index) {
  imgEl.onload = () => {
    if (imgEl.naturalWidth) {
      dimsEl.textContent = `${imgEl.naturalWidth}×${imgEl.naturalHeight}px`;
    }
  };
  try {
    const response = await api.fetchApi(
      `${API_PREFIX}/thumbnail?fingerprint=${encodeURIComponent(fingerprint)}&index=${encodeURIComponent(index)}`,
    );
    if (!response.ok) return;
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    copyResultObjectUrls.push(url);
    imgEl.src = url;
    // The link opens this exact blob -- the cached thumbnail, capped at
    // 256px on its longer side. That is the highest resolution available;
    // the original image file is never stored (plan section 10.2 / 14).
    if (linkEl) linkEl.href = url;
    imgEl.classList.add("is-loaded");
  } catch (_) {
    /* leave the placeholder and the "—" dimensions */
  }
}

// The lines shown under the reference hint in the Copy-prompt result box.
// FL2VA keeps its concrete per-reference labels ("first_frame" / "last_frame")
// -- still the most useful form with 0-2 references. Ref2VA has no meaningful
// per-slot label, so it gets one line counted per type over the whole list.
function referenceSummaryLines(variant, references) {
  if (variant !== "ref2va") {
    return references.map((ref) => `- ${ref.label || `reference ${ref.index}`}`);
  }
  const counts = { image: 0, video: 0, audio: 0 };
  for (const ref of references) {
    const t = ref.type || "image";
    counts[t] = (counts[t] || 0) + 1;
  }
  const parts = [];
  if (counts.image) parts.push(`${counts.image} image${counts.image > 1 ? "s" : ""}`);
  if (counts.video) parts.push(`${counts.video} video${counts.video > 1 ? "s" : ""}`);
  if (counts.audio) parts.push(`${counts.audio} audio`);
  return [parts.join(", ") + "."];
}

function renderCopyResult(fingerprint, verbose, headline) {
  const resultEl = panel.detailEl.querySelector("[data-h3cm-copy-result]");
  const references =
    (verbose.system && Array.isArray(verbose.system.references) && verbose.system.references) || [];
  const isRef2va = currentVariant === "ref2va";

  revokeCopyResultUrls();
  resultEl.innerHTML = "";
  resultEl.hidden = false;

  if (references.length === 0) {
    resultEl.textContent = headline;
    return;
  }

  const heading = document.createElement("div");
  heading.textContent = isRef2va
    ? `${headline} This cache entry was created with these references:`
    : `${headline} This cache entry was created with image references:`;
  resultEl.appendChild(heading);

  const list = document.createElement("ul");
  list.className = "h3cm-ref-list";
  for (const line of referenceSummaryLines(currentVariant, references)) {
    const item = document.createElement("li");
    item.textContent = line;
    list.appendChild(item);
  }
  resultEl.appendChild(list);

  const limitNote = document.createElement("div");
  limitNote.className = "h3cm-copy-note";
  limitNote.textContent = isRef2va
    ? "The original reference files are never stored."
    : "This is the only visual reference this cache entry has — the original image file is never stored.";
  resultEl.appendChild(limitNote);

  // Ref2VA's per-type breakdown with thumbnails already lives in the
  // persistent renderDetailRefs() section above, so this box stays text-only
  // for it -- no point re-fetching the same thumbnail blobs here.
  if (!isRef2va) {
    const thumbs = document.createElement("div");
    thumbs.className = "h3cm-thumbs";
    for (const ref of references) {
      const cell = document.createElement("div");
      cell.className = "h3cm-thumb-cell";

      const link = document.createElement("a");
      link.target = "_blank";
      link.rel = "noopener";
      link.title = "Open this reference thumbnail (max 256px) in a new tab";

      const img = document.createElement("img");
      img.className = "h3cm-thumb";
      img.alt = ref.label || `reference ${ref.index}`;

      const dims = document.createElement("span");
      dims.className = "h3cm-thumb-dims";
      dims.textContent = "—";

      link.appendChild(img);
      cell.append(link, dims);
      thumbs.appendChild(cell);
      loadCopyResultThumbnail(img, link, dims, fingerprint, ref.index);
    }
    resultEl.appendChild(thumbs);
  }

  const note = document.createElement("div");
  note.className = "h3cm-copy-note";
  note.textContent =
    "Load these references manually into the matching inputs on the node.";
  resultEl.appendChild(note);
}

// The transient "Copied!" affordance every click-to-copy control in the panel
// shares: an `is-copied` class plus a title swap that reverts after
// COPY_FEEDBACK_MS. It lives in separate pieces rather than one call because
// the cancel has to run *before* the copy attempt, which is async, while the
// outcome is only known after it.
//
// Both pieces that settle an outcome cancel again on the way in, and that is
// not redundant with the cancel before the await. Two clicks can both clear a
// still-empty handle before either has scheduled anything -- the copy attempt
// is async, so the first click may not have reached its timer yet when the
// second one looks for it. Both would then schedule a revert while the element
// remembers only the later handle, orphaning the earlier timer to fire in the
// middle of the later window. Cancelling as the outcome is applied is the last
// moment that timer can still be caught. The earlier cancel guards something
// else: a timer from an already settled click that would otherwise fire while
// this one waits on a permission prompt. Cancelling is idempotent, so keeping
// both costs nothing.
//
// Left unhandled on purpose: if overlapping attempts settle out of order and
// the earlier one failed, the element ends up reading "Copy failed" even
// though the later attempt succeeded. Numbering generations to tell those
// apart would be more machinery than a 1.5 s tooltip is worth.
const COPY_FEEDBACK_MS = 1500;
const COPY_FAILED_TITLE = "Copy failed - select the text manually";

// Cancel any pending revert on this element. The handle rides on the element,
// which the detail-panel rebuild discards whole.
export function cancelCopyRevert(el) {
  if (!el._h3cmCopyRevertTimer) return;
  clearTimeout(el._h3cmCopyRevertTimer);
  el._h3cmCopyRevertTimer = null;
}

// Show the confirmation on `el` and schedule the revert back to `revertTitle`.
export function markCopied(el, revertTitle) {
  cancelCopyRevert(el);
  el.classList.add("is-copied");
  el.title = "Copied!";
  el._h3cmCopyRevertTimer = setTimeout(() => {
    el.classList.remove("is-copied");
    el.title = revertTitle;
    el._h3cmCopyRevertTimer = null;
  }, COPY_FEEDBACK_MS);
}

// Show the failure on `el`. The class has to come off explicitly: a preceding
// click may have left the element looking copied, and the revert that would
// have cleaned it up is exactly what the cancel above just dropped.
export function markCopyFailed(el) {
  cancelCopyRevert(el);
  el.classList.remove("is-copied");
  el.title = COPY_FAILED_TITLE;
}

// Write `text` to the clipboard and give `el` that affordance, so every
// click-to-copy control in the panel behaves the same way. `revertTitle` is
// what the title returns to.
async function copyToClipboardWithFeedback(el, text, revertTitle) {
  cancelCopyRevert(el);
  try {
    await navigator.clipboard.writeText(text);
  } catch (_) {
    markCopyFailed(el);
    return false;
  }
  markCopied(el, revertTitle);
  return true;
}

// The small icon on the prompt box is just a compact trigger for the same
// action as the big "Copy prompt" button -- it defers entirely to
// copyPrompt() and only adds its own transient "Copied!" affordance. It
// cannot go through copyToClipboardWithFeedback(): the clipboard write
// happens inside copyPrompt(), together with side effects this button does
// not own, so it shares the affordance helpers instead.
async function copyPromptText(button) {
  cancelCopyRevert(button);
  const ok = await copyPrompt();
  if (!ok) {
    markCopyFailed(button);
    return;
  }
  markCopied(button, "Copy prompt");
}

// Copies the open entry's prompt to the clipboard, renders the reference
// panel below, and sets the toolbar hint -- all only on this click, never
// on panel open. Returns true on a successful clipboard write, false
// otherwise.
async function copyPrompt() {
  if (!openDetailFingerprint) return false;
  const entry = findNormalEntry(openDetailFingerprint);
  if (!entry) return false;

  const prompt = (entry.verbose.system && entry.verbose.system.prompt) || "";
  const references =
    (entry.verbose.system
      && Array.isArray(entry.verbose.system.references)
      && entry.verbose.system.references)
    || [];

  let ok;
  try {
    await navigator.clipboard.writeText(prompt);
    renderCopyResult(entry.fingerprint, entry.verbose, "Copied to clipboard.");
    ok = true;
  } catch (err) {
    renderCopyResult(
      entry.fingerprint,
      entry.verbose,
      "Couldn't copy automatically - select the prompt above and copy it manually.",
    );
    ok = false;
  }

  const hintEl = panel.detailEl.querySelector("[data-h3cm-refs-hint]");
  hintEl.textContent =
    references.length > 0
      ? `${references.length > 1 ? "References" : "Reference"} detected — see below.`
      : "";

  return ok;
}

// --- Delete: remove a whole cache entry --------------------------------

async function deleteEntry(fingerprint, statusEl) {
  const confirmed = window.confirm(
    "Delete this cache entry? This removes the cached result permanently and cannot be undone.",
  );
  if (!confirmed) return;

  try {
    await fetchJson(`${API_PREFIX}/delete`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ fingerprint }),
    });
    if (openDetailFingerprint === fingerprint) closeDetail();
    await runCheck(); // entry disappears from the list
  } catch (err) {
    const message = `Delete failed (${err && err.message ? err.message : err})`;
    if (statusEl) statusEl.textContent = message;
    else setCacheStatus(message, "off");
  }
}

function onDetailDeleteClick() {
  if (!openDetailFingerprint) return;
  deleteEntry(openDetailFingerprint, panel.detailEl.querySelector("[data-h3cm-detail-status]"));
}

// --- launcher + wiring -------------------------------------------------

const LAUNCHER_POSITION_KEY = "h3cm-launcher-position";
const LAUNCHER_DRAG_THRESHOLD_PX = 5;

// Read a persisted {left, top} (px). Returns null on any problem -- missing
// key, malformed JSON, non-finite numbers, or localStorage being unavailable
// (private mode, storage disabled) -- so the caller can silently fall back to
// the default corner without an exception escaping setup().
export function readLauncherPosition() {
  try {
    const raw = window.localStorage.getItem(LAUNCHER_POSITION_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    if (
      !parsed ||
      typeof parsed.left !== "number" ||
      typeof parsed.top !== "number" ||
      !Number.isFinite(parsed.left) ||
      !Number.isFinite(parsed.top)
    ) {
      return null;
    }
    return { left: parsed.left, top: parsed.top };
  } catch (err) {
    return null;
  }
}

export function writeLauncherPosition(pos) {
  try {
    window.localStorage.setItem(LAUNCHER_POSITION_KEY, JSON.stringify(pos));
  } catch (err) {
    /* storage unavailable / over quota -- the position just won't persist */
  }
}

// Keep the button fully inside the current viewport.
export function clampLauncherPosition(left, top, el) {
  const maxLeft = Math.max(0, window.innerWidth - el.offsetWidth);
  const maxTop = Math.max(0, window.innerHeight - el.offsetHeight);
  return {
    left: Math.min(Math.max(0, left), maxLeft),
    top: Math.min(Math.max(0, top), maxTop),
  };
}

function installLauncher() {
  if (document.querySelector("[data-h3cm-launcher]")) return;

  const button = document.createElement("button");
  button.type = "button";
  button.className = "h3cm-floating-launcher";
  button.dataset.h3cmLauncher = "true";
  button.title = "Open MiniMax H3 Prompt Cache Manager";
  for (const text of ["H3", "Prompt", "Cache"]) {
    const line = document.createElement("span");
    line.className = "h3cm-launcher-line";
    line.textContent = text;
    button.appendChild(line);
  }
  document.body.appendChild(button);

  // Switch anchoring from the CSS default (right/bottom) to left/top. Called
  // the first time the button actually moves, and when restoring a position.
  function pinToLeftTop(left, top) {
    button.style.left = `${left}px`;
    button.style.top = `${top}px`;
    button.style.right = "auto";
    button.style.bottom = "auto";
  }

  // Restore a saved position, but only if it still fits this viewport --
  // otherwise stay at the default corner.
  const saved = readLauncherPosition();
  if (saved) {
    const clamped = clampLauncherPosition(saved.left, saved.top, button);
    if (clamped.left === saved.left && clamped.top === saved.top) {
      pinToLeftTop(saved.left, saved.top);
    }
  }

  // --- drag to move (pointer events cover mouse, touch and stylus at once) ---
  let tracking = false; // following a pointerdown
  let dragging = false; // movement has passed the click-vs-drag threshold
  let suppressClick = false; // swallow the click synthesized right after a drag
  let startX = 0;
  let startY = 0;
  let originLeft = 0;
  let originTop = 0;

  button.addEventListener("pointerdown", (event) => {
    if (event.button > 0) return; // primary mouse button / touch / pen only
    tracking = true;
    dragging = false;
    suppressClick = false; // clear any stale flag from a drag that fired no click
    startX = event.clientX;
    startY = event.clientY;
    const rect = button.getBoundingClientRect();
    originLeft = rect.left;
    originTop = rect.top;
  });

  window.addEventListener("pointermove", (event) => {
    if (!tracking) return;
    const dx = event.clientX - startX;
    const dy = event.clientY - startY;
    if (!dragging) {
      if (Math.hypot(dx, dy) < LAUNCHER_DRAG_THRESHOLD_PX) return;
      dragging = true;
      button.classList.add("is-dragging");
    }
    const next = clampLauncherPosition(originLeft + dx, originTop + dy, button);
    pinToLeftTop(next.left, next.top);
    event.preventDefault();
  });

  function endDrag() {
    if (!tracking) return;
    tracking = false;
    if (!dragging) return; // under the threshold -- leave it for the click handler
    dragging = false;
    button.classList.remove("is-dragging");
    suppressClick = true;
    // Only the click the browser synthesizes from this gesture (dispatched
    // synchronously, before this timer) should be swallowed -- not some
    // unrelated later activation.
    setTimeout(() => {
      suppressClick = false;
    }, 0);
    const rect = button.getBoundingClientRect();
    const clamped = clampLauncherPosition(rect.left, rect.top, button);
    pinToLeftTop(clamped.left, clamped.top);
    writeLauncherPosition(clamped);
  }

  window.addEventListener("pointerup", endDrag);
  window.addEventListener("pointercancel", endDrag);

  button.addEventListener("click", (event) => {
    if (suppressClick) {
      suppressClick = false;
      event.preventDefault();
      event.stopPropagation();
      return;
    }
    openPanel();
  });

  // If the window shrinks under the button, pull it back into view.
  window.addEventListener("resize", () => {
    if (!button.style.left) return; // still at the default corner -- nothing to clamp
    const rect = button.getBoundingClientRect();
    const clamped = clampLauncherPosition(rect.left, rect.top, button);
    pinToLeftTop(clamped.left, clamped.top);
  });
}

document.addEventListener("keydown", (event) => {
  if (event.key !== "Escape") return;
  if (!panel || !panel.root.classList.contains("is-open")) return;
  event.preventDefault();
  if (!panel.detailEl.hidden) closeDetail();
  else closePanel();
});

app.registerExtension({
  name: EXTENSION_NAME,
  commands: [
    {
      id: "h3-cache-manager.open",
      label: "Open MiniMax H3 Cache Manager",
      function: openPanel,
    },
  ],
  menuCommands: [
    { path: ["Extensions", "MiniMax H3 Cache Manager"], commands: ["h3-cache-manager.open"] },
  ],
  async setup() {
    injectStyles();
    installLauncher();
  },
});
