const search = document.querySelector("#search");
const clearSearch = document.querySelector("#clear-search");
const sections = [...document.querySelectorAll(".concept-section")];
const rows = [...document.querySelectorAll("[data-concept]")];
const count = document.querySelector("#result-count");
const emptyState = document.querySelector("#empty-state");

function filterConcepts() {
  const words = search.value.trim().toLowerCase().split(/\s+/).filter(Boolean);
  let matches = 0;
  rows.forEach(row => {
    row.hidden = !words.every(word => row.dataset.search.includes(word));
    if (!row.hidden) matches++;
  });
  sections.forEach(section => {
    section.hidden = ![...section.querySelectorAll("[data-concept]")].some(row => !row.hidden);
  });
  count.textContent = words.length ? `${matches} of ${rows.length} concepts` : `${rows.length} concepts`;
  count.hidden = !words.length;
  clearSearch.hidden = !words.length;
  emptyState.hidden = matches > 0;
}
function resetSearch() {
  search.value = "";
  filterConcepts();
  search.focus();
}
search.addEventListener("input", filterConcepts);
clearSearch.addEventListener("click", resetSearch);
document.querySelector("#reset-search").addEventListener("click", resetSearch);
search.addEventListener("keydown", event => {
  if (event.key === "Escape") resetSearch();
});

document.querySelectorAll("nav a").forEach(link => {
  link.addEventListener("click", () => {
    if (search.value) {
      search.value = "";
      filterConcepts();
    }
  });
});

const sourceToggle = document.querySelector("#toggle-sources");
sourceToggle.addEventListener("click", () => {
  const show = sourceToggle.getAttribute("aria-pressed") !== "true";
  sourceToggle.setAttribute("aria-pressed", String(show));
  sourceToggle.textContent = show ? "Hide citations" : "Show citations";
  document.querySelectorAll(".cell-sources").forEach(source => { source.hidden = !show; });
});

function revealAnchor() {
  let id;
  try { id = decodeURIComponent(location.hash.slice(1)); } catch { return; }
  const target = document.getElementById(id);
  if (!target) return;
  const details = target.closest("details");
  if (details && !details.open) details.open = true;
  if (id.startsWith("source-") || id === "sources" || id === "mapping-notes") {
    requestAnimationFrame(() => target.scrollIntoView({block: "start"}));
  }
}
window.addEventListener("hashchange", revealAnchor);
document.addEventListener("click", event => {
  const link = event.target.closest('a[href^="#source"]');
  if (link) document.querySelector("#sources").open = true;
});
revealAnchor();

if ("IntersectionObserver" in window) {
  const observer = new IntersectionObserver(entries => {
    entries.forEach(entry => {
      if (!entry.isIntersecting) return;
      document.querySelectorAll("nav a").forEach(link => {
        if (link.hash === `#${entry.target.id}`) link.setAttribute("aria-current", "location");
        else link.removeAttribute("aria-current");
      });
    });
  }, {rootMargin: "-150px 0px -55% 0px"});
  sections.forEach(section => observer.observe(section));
}
