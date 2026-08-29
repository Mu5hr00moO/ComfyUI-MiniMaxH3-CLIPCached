// MiniMax H3 Cache Manager frontend.
//
// Phase 6 part 1: floating launcher, modal shell, "Check", raw entry list.
// Phase 6 part 2: client-side search / tag / favorite filtering, reference
// thumbnails, inline detail+edit panel (name / notes / tags / favorite).
// Phase 6 part 3 (this file): "Copy prompt" puts a cached entry's prompt on
// the clipboard (writing it into a graph widget was tried and dropped --
// findNodesByType() does not descend into subgraphs, and a prompt converted
// to an input has no widget to set; see CACHE_MANAGER_PLAN.md section 14),
// still showing the image-reference notice; "Delete" removes a whole cache
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
let renderGeneration = 0; // bumped each render so stale async thumbnails bail
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

    if (entry.classification === "legacy") {
      // Nothing to match a legacy entry on -- show it only in the
      // no-filters view (plan section 11.1 / 7).
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

export function allNormalTags(entries) {
  const set = new Set();
  for (const entry of entries || []) {
    if (entry.classification !== "normal") continue;
    const tags = (entry.verbose && entry.verbose.user && entry.verbose.user.tags) || [];
    for (const t of tags) set.add(String(t));
  }
  return [...set].sort((a, b) => a.localeCompare(b));
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
  root.querySelector("[data-h3cm-detail-close]").addEventListener("click", closeDetail);
  root.querySelector("[data-h3cm-save]").addEventListener("click", saveDetail);
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
// switching resets search / tag / favorites to their defaults and closes any
// open detail panel, so each tab always opens on a clean, unfiltered view.
function switchVariant(variant) {
  if (variant !== "fl2va" && variant !== "ref2va") return;
  currentVariant = variant;
  panel.variantBtns.forEach((btn) =>
    btn.classList.toggle("is-active", btn.dataset.h3cmVariant === variant));

  panel.searchEl.value = "";
  panel.tagFilterEl.value = ALL_TAGS;
  panel.favoritesOnlyEl.checked = false;
  if (openDetailFingerprint) closeDetail();

  renderList();
}

// --- list rendering -------------------------------------------------------

function refreshTagFilterOptions() {
  const { tagFilterEl } = panel;
  const previous = tagFilterEl.value;
  const tags = lastCheckResult ? allNormalTags(lastCheckResult.entries) : [];

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

async function loadThumbnail(imgEl, fingerprint, index, generation) {
  try {
    const response = await api.fetchApi(
      `${API_PREFIX}/thumbnail?fingerprint=${encodeURIComponent(fingerprint)}&index=${encodeURIComponent(index)}`,
    );
    if (!response.ok || generation !== renderGeneration) return;
    const blob = await response.blob();
    if (generation !== renderGeneration) return; // list re-rendered meanwhile
    const url = URL.createObjectURL(blob);
    objectUrls.push(url);
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

function buildLegacyRow(entry) {
  const row = document.createElement("div");
  row.className = "h3cm-row is-legacy";

  const fp = document.createElement("span");
  fp.className = "h3cm-fp";
  fp.textContent = `${String(entry.fingerprint || "").slice(0, 12)}…`;

  const badge = document.createElement("span");
  badge.className = "h3cm-badge h3cm-badge-legacy";
  badge.textContent = "legacy";

  const hint = document.createElement("span");
  hint.className = "h3cm-legacy-hint";
  hint.textContent = "Use this entry once to populate details";

  // Legacy entries have no detail panel, but they still correspond to real
  // .json/.safetensors files the user may want to remove (plan section 15
  // makes no exception for legacy).
  const del = document.createElement("button");
  del.type = "button";
  del.className = "h3cm-button h3cm-danger h3cm-row-delete";
  del.textContent = "Delete";
  del.addEventListener("click", (event) => {
    event.stopPropagation();
    deleteEntry(entry.fingerprint, null);
  });

  row.append(fp, badge, hint, del);
  return row;
}

function buildNormalRow(entry, generation) {
  const user = (entry.verbose && entry.verbose.user) || {};
  const system = (entry.verbose && entry.verbose.system) || {};
  const tags = Array.isArray(user.tags) ? user.tags : [];
  const references = Array.isArray(system.references) ? system.references : [];

  const row = document.createElement("div");
  row.className = "h3cm-row is-normal";
  row.dataset.fingerprint = entry.fingerprint; // so attachDetailAfterRow() can find this row

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

  row.append(star, label);
  if (tags.length) row.appendChild(buildTagChips(tags));
  if (references.length) {
    const variant = system.node_variant || "fl2va";
    row.appendChild(
      variant === "ref2va"
        ? buildRef2vaThumbnails(entry.fingerprint, references, generation)
        : buildThumbnails(entry.fingerprint, references, generation),
    );
  }

  row.addEventListener("click", () => openDetail(entry.fingerprint));
  return row;
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
  const filtered = filterEntries(entries, state);

  panel.listEl.innerHTML = "";

  if (filtered.length === 0) {
    const empty = document.createElement("div");
    empty.className = "h3cm-empty";
    empty.textContent = entries.length === 0
      ? "No cache entries found."
      : "No entries match the current search / filters.";
    panel.listEl.appendChild(empty);
    return;
  }

  for (const entry of filtered) {
    panel.listEl.appendChild(
      entry.classification === "legacy"
        ? buildLegacyRow(entry)
        : buildNormalRow(entry, generation),
    );
  }
}

// --- check ---------------------------------------------------------------

async function runCheck() {
  if (!panel) return;
  panel.statusEl.textContent = "Cache: checking…";

  try {
    const data = await fetchJson(`${API_PREFIX}/check`);
    lastCheckResult = data;
    const count = typeof data.total_count === "number" ? data.total_count : "—";
    panel.statusEl.textContent = `Cache: ${count} entries / ${formatBytes(data.total_size_bytes)}`;

    refreshTagFilterOptions();
    renderList();

    // Keep the detail panel on the same entry across a refresh (e.g. after
    // a Save or a favorite toggle); close it if that entry is gone.
    // renderList() above rebuilt every row from scratch, so the detail node
    // has to be re-attached under the (new) row for its fingerprint --
    // otherwise it drifts back to the end of the list on each refresh.
    if (openDetailFingerprint) {
      const still = findNormalEntry(openDetailFingerprint);
      if (still) {
        populateDetail(still);
        attachDetailAfterRow(openDetailFingerprint);
      } else {
        closeDetail();
      }
    }
  } catch (err) {
    lastCheckResult = null;
    panel.statusEl.textContent = `Cache: check failed (${err && err.message ? err.message : err})`;
    panel.listEl.innerHTML = "";
  }
}

// --- favorite toggle (row) ----------------------------------------------

async function toggleFavorite(fingerprint, currentlyFavorite) {
  try {
    await postUpdate({ fingerprint, favorite: !currentlyFavorite });
    await runCheck(); // re-render from the real state, never patch JS state
  } catch (err) {
    panel.statusEl.textContent = `Update failed (${err && err.message ? err.message : err})`;
  }
}

// --- detail + edit panel -------------------------------------------------

function findNormalEntry(fingerprint) {
  if (!lastCheckResult) return null;
  return (lastCheckResult.entries || []).find(
    (e) => e.fingerprint === fingerprint && e.classification === "normal" && e.verbose,
  ) || null;
}

// Ref2VA only: the "<Picture N>" / "<Video N>" / "<Audio N>" tags in the
// prompt are positional, counted per type over the reference list in order.
// The count is always derived here in JS, never read from a stored field.
const REF_TYPE_LABEL = { image: "Picture", video: "Video", audio: "Audio" };

function renderDetailRefs(container, fingerprint, variant, references) {
  container.innerHTML = "";
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

  const counters = { image: 0, video: 0, audio: 0 };
  for (const ref of references) {
    const type = ref.type || "image";
    counters[type] = (counters[type] || 0) + 1;
    const posLabel = `${REF_TYPE_LABEL[type] || "Reference"} ${counters[type]}`;

    const cell = document.createElement("div");
    cell.className = "h3cm-detail-ref-cell";

    if (type === "audio") {
      const pill = document.createElement("span");
      pill.className = "h3cm-audio-pill";
      pill.textContent = "audio";
      cell.appendChild(pill);
    } else {
      const img = document.createElement("img");
      img.className = "h3cm-thumb";
      img.alt = posLabel;
      cell.appendChild(img);
      loadThumbnail(img, fingerprint, ref.index, renderGeneration);
    }

    const cap = document.createElement("span");
    cap.className = "h3cm-detail-ref-label";
    cap.textContent = posLabel;
    cell.appendChild(cap);

    grid.appendChild(cell);
  }
  container.appendChild(grid);
}

function populateDetail(entry) {
  const { detailEl } = panel;
  const user = (entry.verbose && entry.verbose.user) || {};
  const system = (entry.verbose && entry.verbose.system) || {};

  detailEl.querySelector("[data-h3cm-detail-title]").textContent = entryLabel(entry);
  detailEl.querySelector("[data-h3cm-detail-prompt]").textContent = system.prompt || "(no prompt)";
  renderDetailRefs(
    detailEl.querySelector("[data-h3cm-detail-refs]"),
    entry.fingerprint,
    system.node_variant || "fl2va",
    Array.isArray(system.references) ? system.references : [],
  );
  detailEl.querySelector("[data-h3cm-edit-name]").value = user.name || "";
  detailEl.querySelector("[data-h3cm-edit-notes]").value = user.notes || "";
  detailEl.querySelector("[data-h3cm-edit-tags]").value = (Array.isArray(user.tags) ? user.tags : []).join(", ");
  detailEl.querySelector("[data-h3cm-edit-favorite]").checked = user.favorite === true;
  detailEl.querySelector("[data-h3cm-detail-status]").textContent = "";
  detailEl.hidden = false;
}

// Move the single detail node so it sits right after the row it describes.
// .after() detaches it from wherever it currently is first, so there is
// never a duplicate -- it is always the same one node from the template.
function attachDetailAfterRow(fingerprint) {
  const rowEl = panel.listEl.querySelector(
    `[data-fingerprint="${CSS.escape(fingerprint)}"]`,
  );
  if (rowEl) rowEl.after(panel.detailEl);
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
// clipboard works regardless of graph structure. See CACHE_MANAGER_PLAN.md
// section 14. Only classification="normal" reaches here (legacy has no
// detail panel and no "Copy prompt" button).

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
  return [parts.join(", ") + ` reference${references.length > 1 ? "s" : ""}.`];
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

// The small icon on the prompt box is just a compact trigger for the same
// action as the big "Copy prompt" button -- it defers entirely to
// copyPrompt() and only adds its own transient "Copied!" affordance.
async function copyPromptText(button) {
  const ok = await copyPrompt();
  button.classList.toggle("is-copied", ok);
  button.title = ok ? "Copied!" : "Copy failed - select the text manually";
  if (ok) {
    setTimeout(() => {
      button.classList.remove("is-copied");
      button.title = "Copy prompt";
    }, 1500);
  }
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
    else panel.statusEl.textContent = message;
  }
}

function onDetailDeleteClick() {
  if (!openDetailFingerprint) return;
  deleteEntry(openDetailFingerprint, panel.detailEl.querySelector("[data-h3cm-detail-status]"));
}

// --- launcher + wiring -------------------------------------------------

function installLauncher() {
  if (document.querySelector("[data-h3cm-launcher]")) return;
  const button = document.createElement("button");
  button.type = "button";
  button.className = "h3cm-floating-launcher";
  button.dataset.h3cmLauncher = "true";
  button.textContent = "H3 Cache";
  button.title = "Open MiniMax H3 Cache Manager";
  button.addEventListener("click", openPanel);
  document.body.appendChild(button);
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
