// MiniMax H3 Cache Manager -- Phase 6 part 1: floating launcher, modal
// shell, "Check" button, raw entry list. Search / tags / favorites /
// thumbnails / full prompt / name+notes / Load / Delete are deliberately
// out of scope for this step.
//
// Structure follows the local convention in
// custom_nodes/MiniMaxH3-Prompt-Writer/web/main.js (singleton panel built
// once, injectStyles via a <link> guarded by a data-attribute,
// app.registerExtension with commands/menuCommands/setup, all HTTP through
// api.fetchApi so a ComfyUI mounted on a sub-path still works). The content,
// markup, CSS and naming here are our own.

import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

const EXTENSION_NAME = "minimax.h3.cache.manager";
const API_PREFIX = "/h3_cache_manager"; // must match minimaxh3_clipcache/routes.py

let panel = null; // { root, statusEl, listEl } -- built once

function injectStyles() {
  if (document.querySelector("link[data-h3cm-styles]")) return;
  const link = document.createElement("link");
  link.rel = "stylesheet";
  link.href = new URL("./styles.css", import.meta.url).href;
  link.dataset.h3cmStyles = "true";
  document.head.appendChild(link);
}

function formatBytes(bytes) {
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
        <span class="h3cm-status" data-h3cm-status>Cache: — entries / —</span>
      </div>
      <div class="h3cm-list" data-h3cm-list></div>
    </section>
  `;
  document.body.appendChild(root);

  const statusEl = root.querySelector("[data-h3cm-status]");
  const listEl = root.querySelector("[data-h3cm-list]");

  root.querySelectorAll("[data-h3cm-close]").forEach((el) => {
    el.addEventListener("click", closePanel);
  });
  root.querySelector("[data-h3cm-check]").addEventListener("click", () => {
    runCheck();
  });

  panel = { root, statusEl, listEl };
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

function renderEntries(entries) {
  const { listEl } = panel;
  listEl.innerHTML = "";

  if (!Array.isArray(entries) || entries.length === 0) {
    const empty = document.createElement("div");
    empty.className = "h3cm-empty";
    empty.textContent = "No cache entries found.";
    listEl.appendChild(empty);
    return;
  }

  for (const entry of entries) {
    const row = document.createElement("div");
    row.className = "h3cm-row";

    const fp = document.createElement("span");
    fp.className = "h3cm-fp";
    fp.textContent = `${String(entry.fingerprint || "").slice(0, 12)}…`;

    const badge = document.createElement("span");
    const classification = entry.classification === "legacy" ? "legacy" : "normal";
    badge.className = `h3cm-badge h3cm-badge-${classification}`;
    badge.textContent = classification;

    row.appendChild(fp);
    row.appendChild(badge);
    listEl.appendChild(row);
  }
}

async function runCheck() {
  if (!panel) return;
  const { statusEl } = panel;
  statusEl.textContent = "Cache: checking…";

  try {
    const response = await api.fetchApi(`${API_PREFIX}/check`);
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    const data = await response.json();
    const count = typeof data.total_count === "number" ? data.total_count : "—";
    statusEl.textContent = `Cache: ${count} entries / ${formatBytes(data.total_size_bytes)}`;
    renderEntries(data.entries);
  } catch (err) {
    statusEl.textContent = `Cache: check failed (${err && err.message ? err.message : err})`;
    if (panel.listEl) panel.listEl.innerHTML = "";
  }
}

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
  if (panel && panel.root.classList.contains("is-open")) {
    event.preventDefault();
    closePanel();
  }
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
