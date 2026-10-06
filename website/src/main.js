import "highlight.js/styles/github-dark.css";
import "./styles.css";

import {
  navigationGroups,
  orderedPages,
  pageByRoute,
  pageBySource,
} from "./navigation.js";
import {
  createMarkdownRenderer,
  extractPageOutline,
} from "./markdown.js";
import {
  createRouter,
  normalizeRoute,
  routeToHref,
} from "./router.js";
import {
  repositoryFileUrl,
  siteConfig,
} from "./site-config.js";

const BASE_URL = import.meta.env.BASE_URL;
const THEME_STORAGE_KEY = "pfgap-docs-theme";

const app = document.querySelector("#app");

if (!app) {
  throw new Error("The website application root '#app' was not found.");
}

const markdownRenderer = createMarkdownRenderer({
  baseUrl: BASE_URL,
  pageBySource,
  routeToHref,
  repositoryUrl: siteConfig.projectRepository,
});

let currentPage = null;
let currentOutline = [];
let searchIndex = [];
let searchIndexPromise = null;
let mobileNavigationOpen = false;

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function joinBaseUrl(relativePath) {
  const cleanPath = relativePath.replace(/^\/+/, "");
  return `${BASE_URL}${cleanPath}`;
}

function createIcon(name) {
  const icons = {
    menu: `
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <path d="M4 7h16M4 12h16M4 17h16" />
      </svg>
    `,
    close: `
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <path d="m6 6 12 12M18 6 6 18" />
      </svg>
    `,
    search: `
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <circle cx="11" cy="11" r="7" />
        <path d="m16.5 16.5 4 4" />
      </svg>
    `,
    github: `
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <path d="M12 2.7a9.3 9.3 0 0 0-2.94 18.12c.47.08.64-.2.64-.45v-1.8c-2.62.57-3.17-1.11-3.17-1.11-.43-1.09-1.05-1.38-1.05-1.38-.86-.59.07-.58.07-.58.95.07 1.45.98 1.45.98.85 1.44 2.22 1.03 2.76.79.09-.61.33-1.03.6-1.27-2.09-.24-4.29-1.05-4.29-4.66 0-1.03.37-1.87.98-2.53-.1-.24-.43-1.2.09-2.5 0 0 .8-.26 2.62.97A9.1 9.1 0 0 1 12 6.96a9 9 0 0 1 2.39.32c1.82-1.23 2.62-.97 2.62-.97.52 1.3.19 2.26.09 2.5.61.66.98 1.5.98 2.53 0 3.62-2.21 4.41-4.31 4.65.34.29.64.87.64 1.75v2.63c0 .25.17.54.65.45A9.3 9.3 0 0 0 12 2.7Z" />
      </svg>
    `,
    sun: `
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <circle cx="12" cy="12" r="4" />
        <path d="M12 2v2M12 20v2M4.93 4.93l1.42 1.42M17.65 17.65l1.42 1.42M2 12h2M20 12h2M4.93 19.07l1.42-1.42M17.65 6.35l1.42-1.42" />
      </svg>
    `,
    moon: `
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <path d="M20.5 14.2A8.5 8.5 0 0 1 9.8 3.5 8.5 8.5 0 1 0 20.5 14.2Z" />
      </svg>
    `,
    chevron: `
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <path d="m9 18 6-6-6-6" />
      </svg>
    `,
    external: `
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <path d="M14 4h6v6M20 4l-9 9" />
        <path d="M18 13v6a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1h6" />
      </svg>
    `,
    copy: `
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <rect x="8" y="8" width="11" height="11" rx="2" />
        <path d="M16 8V5a2 2 0 0 0-2-2H5a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h3" />
      </svg>
    `,
    check: `
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <path d="m5 12 4 4L19 6" />
      </svg>
    `,
  };

  return icons[name] ?? "";
}

function getCurrentTheme() {
  return document.documentElement.dataset.theme === "dark"
    ? "dark"
    : "light";
}

function updateThemeControls() {
  const theme = getCurrentTheme();
  const button = document.querySelector("[data-theme-toggle]");

  if (!button) {
    return;
  }

  const nextTheme = theme === "dark" ? "light" : "dark";
  button.setAttribute("aria-label", `Use ${nextTheme} theme`);
  button.setAttribute("title", `Use ${nextTheme} theme`);
  button.innerHTML = theme === "dark"
    ? createIcon("sun")
    : createIcon("moon");
}

function toggleTheme() {
  const nextTheme = getCurrentTheme() === "dark"
    ? "light"
    : "dark";

  document.documentElement.dataset.theme = nextTheme;
  localStorage.setItem(THEME_STORAGE_KEY, nextTheme);
  updateThemeControls();
}

function renderHeader() {
  return `
    <header class="site-header">
      <button
        class="icon-button mobile-menu-button"
        type="button"
        aria-label="Open documentation navigation"
        aria-expanded="false"
        data-mobile-menu-toggle
      >
        ${createIcon("menu")}
      </button>

      <a
        class="brand"
        href="${escapeHtml(routeToHref("/"))}"
        data-route="/"
        aria-label="PFGAP documentation home"
      >
        <span class="brand-mark" aria-hidden="true">PF</span>
        <span class="brand-copy">
          <strong>PFGAP</strong>
          <span>Documentation</span>
        </span>
      </a>

      <button
        class="header-search-trigger"
        type="button"
        aria-label="Search documentation"
        data-search-open
      >
        ${createIcon("search")}
        <span>Search documentation</span>
        <kbd>/</kbd>
      </button>

      <nav class="header-actions" aria-label="Project links">
        <a
          class="icon-button"
          href="${escapeHtml(siteConfig.projectRepository)}"
          target="_blank"
          rel="noreferrer"
          aria-label="Open the PFGAP repository on GitHub"
          title="GitHub repository"
        >
          ${createIcon("github")}
        </a>

        <button
          class="icon-button"
          type="button"
          data-theme-toggle
        ></button>
      </nav>
    </header>
  `;
}

function renderSidebar() {
  const groups = navigationGroups.map((group) => {
    const items = group.pages.map((page) => `
      <li>
        <a
          class="sidebar-link"
          href="${escapeHtml(routeToHref(page.route))}"
          data-route="${escapeHtml(page.route)}"
        >
          ${escapeHtml(page.title)}
        </a>
      </li>
    `).join("");

    return `
      <section class="sidebar-group">
        <h2>${escapeHtml(group.title)}</h2>
        <ul>${items}</ul>
      </section>
    `;
  }).join("");

  return `
    <div class="mobile-backdrop" data-mobile-backdrop></div>

    <aside class="site-sidebar" aria-label="Documentation navigation">
      <div class="sidebar-mobile-header">
        <span>Documentation</span>
        <button
          class="icon-button"
          type="button"
          aria-label="Close documentation navigation"
          data-mobile-menu-close
        >
          ${createIcon("close")}
        </button>
      </div>

      <nav>${groups}</nav>
    </aside>
  `;
}

function renderBreadcrumbs(page) {
  const group = navigationGroups.find((candidate) =>
    candidate.pages.some((item) => item.route === page.route)
  );

  if (page.route === "/") {
    return `
      <nav class="breadcrumbs" aria-label="Breadcrumb">
        <span>Documentation</span>
      </nav>
    `;
  }

  return `
    <nav class="breadcrumbs" aria-label="Breadcrumb">
      <a href="${escapeHtml(routeToHref("/"))}" data-route="/">
        Documentation
      </a>
      ${createIcon("chevron")}
      <span>${escapeHtml(group?.title ?? "Documentation")}</span>
      ${createIcon("chevron")}
      <span aria-current="page">${escapeHtml(page.title)}</span>
    </nav>
  `;
}

function renderPageNavigation(page) {
  const index = orderedPages.findIndex(
    (candidate) => candidate.route === page.route
  );
  const previous = index > 0 ? orderedPages[index - 1] : null;
  const next = index >= 0 && index < orderedPages.length - 1
    ? orderedPages[index + 1]
    : null;

  return `
    <nav class="page-navigation" aria-label="Previous and next pages">
      ${previous ? `
        <a
          class="page-navigation-link previous"
          href="${escapeHtml(routeToHref(previous.route))}"
          data-route="${escapeHtml(previous.route)}"
        >
          <span>Previous</span>
          <strong>${escapeHtml(previous.title)}</strong>
        </a>
      ` : "<span></span>"}

      ${next ? `
        <a
          class="page-navigation-link next"
          href="${escapeHtml(routeToHref(next.route))}"
          data-route="${escapeHtml(next.route)}"
        >
          <span>Next</span>
          <strong>${escapeHtml(next.title)}</strong>
        </a>
      ` : ""}
    </nav>
  `;
}

function renderTableOfContents(outline) {
  if (outline.length === 0) {
    return `
      <aside class="page-toc" aria-label="On this page">
        <p class="toc-title">On this page</p>
        <p class="toc-empty">No sections</p>
      </aside>
    `;
  }

  return `
    <aside class="page-toc" aria-label="On this page">
      <p class="toc-title">On this page</p>
      <nav>
        <ul>
          ${outline.map((heading) => `
            <li class="toc-level-${heading.level}">
              <a href="#${escapeHtml(heading.id)}">
                ${escapeHtml(heading.text)}
              </a>
            </li>
          `).join("")}
        </ul>
      </nav>
    </aside>
  `;
}

function renderDocumentPage(page, html, outline) {
  const editUrl = repositoryFileUrl(
    `docs/${page.source}`,
    {
      repository: siteConfig.editRepository,
      branch: siteConfig.defaultBranch,
      mode: "edit",
    }
  );

  return `
    <div class="site-layout">
      ${renderSidebar()}

      <main id="main-content" class="document-main" tabindex="-1">
        <div class="document-column">
          ${renderBreadcrumbs(page)}

          <article class="markdown-body" data-document>
            ${html}
          </article>

          <footer class="document-footer">
            <a
              href="${escapeHtml(editUrl)}"
              target="_blank"
              rel="noreferrer"
            >
              Edit this page on GitHub
              ${createIcon("external")}
            </a>
          </footer>

          ${renderPageNavigation(page)}
        </div>

        ${renderTableOfContents(outline)}
      </main>
    </div>
  `;
}

function renderLoadingPage(page) {
  return `
    <div class="site-layout">
      ${renderSidebar()}
      <main id="main-content" class="document-main" tabindex="-1">
        <div class="document-column document-placeholder">
          ${renderBreadcrumbs(page)}
          <div class="placeholder-line placeholder-title"></div>
          <div class="placeholder-line placeholder-title-short"></div>
          <div class="placeholder-line"></div>
          <div class="placeholder-line"></div>
          <div class="placeholder-line placeholder-medium"></div>
        </div>
      </main>
    </div>
  `;
}

function renderErrorPage(title, message) {
  return `
    <div class="site-layout">
      ${renderSidebar()}
      <main id="main-content" class="document-main error-page" tabindex="-1">
        <div class="document-column">
          <p class="eyebrow">Documentation</p>
          <h1>${escapeHtml(title)}</h1>
          <p>${escapeHtml(message)}</p>
          <a
            class="primary-link"
            href="${escapeHtml(routeToHref("/"))}"
            data-route="/"
          >
            Return to documentation home
          </a>
        </div>
      </main>
    </div>
  `;
}

function renderApplicationShell(content) {
  app.innerHTML = `
    ${renderHeader()}
    ${content}
    <div id="search-dialog-root"></div>
  `;
  app.setAttribute("aria-busy", "false");
  updateThemeControls();
}

function updateActiveNavigation(route) {
  for (const link of document.querySelectorAll(".sidebar-link")) {
    const active = link.dataset.route === route;
    link.classList.toggle("active", active);

    if (active) {
      link.setAttribute("aria-current", "page");
    } else {
      link.removeAttribute("aria-current");
    }
  }
}

function closeMobileNavigation() {
  mobileNavigationOpen = false;
  document.body.classList.remove("mobile-navigation-open");

  const toggle = document.querySelector("[data-mobile-menu-toggle]");
  toggle?.setAttribute("aria-expanded", "false");
}

function openMobileNavigation() {
  mobileNavigationOpen = true;
  document.body.classList.add("mobile-navigation-open");

  const toggle = document.querySelector("[data-mobile-menu-toggle]");
  toggle?.setAttribute("aria-expanded", "true");
}

function toggleMobileNavigation() {
  if (mobileNavigationOpen) {
    closeMobileNavigation();
  } else {
    openMobileNavigation();
  }
}

function enhanceCodeBlocks() {
  const blocks = document.querySelectorAll("pre > code");

  for (const code of blocks) {
    const pre = code.parentElement;

    if (!pre || pre.querySelector(".copy-code-button")) {
      continue;
    }

    const languageClass = [...code.classList].find((value) =>
      value.startsWith("language-")
    );
    const language = languageClass
      ? languageClass.replace("language-", "")
      : "text";

    pre.dataset.language = language;

    const button = document.createElement("button");
    button.className = "copy-code-button";
    button.type = "button";
    button.setAttribute("aria-label", "Copy code to clipboard");
    button.setAttribute("title", "Copy code");
    button.innerHTML = createIcon("copy");

    button.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(code.textContent ?? "");
        button.classList.add("copied");
        button.innerHTML = createIcon("check");
        button.setAttribute("aria-label", "Code copied");

        window.setTimeout(() => {
          button.classList.remove("copied");
          button.innerHTML = createIcon("copy");
          button.setAttribute("aria-label", "Copy code to clipboard");
        }, 1600);
      } catch (error) {
        console.error("Could not copy code.", error);
      }
    });

    pre.append(button);
  }
}

function updateTableOfContentsHighlight() {
  const links = [...document.querySelectorAll(".page-toc a")];

  if (links.length === 0) {
    return;
  }

  const headings = currentOutline
    .map((item) => document.getElementById(item.id))
    .filter(Boolean);

  let activeId = headings[0]?.id ?? "";

  for (const heading of headings) {
    if (heading.getBoundingClientRect().top <= 140) {
      activeId = heading.id;
    } else {
      break;
    }
  }

  for (const link of links) {
    link.classList.toggle(
      "active",
      link.getAttribute("href") === `#${activeId}`
    );
  }
}

async function fetchMarkdown(page) {
  const response = await fetch(joinBaseUrl(`docs/${page.source}`), {
    headers: {
      Accept: "text/markdown, text/plain;q=0.9, */*;q=0.1",
    },
  });

  if (!response.ok) {
    throw new Error(
      `Could not load ${page.source}: ${response.status} ${response.statusText}`
    );
  }

  return response.text();
}

function getPageForRoute(route) {
  const normalized = normalizeRoute(route);
  return pageByRoute.get(normalized) ?? null;
}

async function loadRoute(route, { restoreHash = true } = {}) {
  const page = getPageForRoute(route);

  if (!page) {
    currentPage = null;
    currentOutline = [];
    document.title = "Page not found | PFGAP Documentation";
    renderApplicationShell(
      renderErrorPage(
        "Page not found",
        "The requested documentation page does not exist."
      )
    );
    bindInteractiveControls();
    return;
  }

  currentPage = page;
  currentOutline = [];
  renderApplicationShell(renderLoadingPage(page));
  bindInteractiveControls();
  updateActiveNavigation(page.route);

  try {
    const markdown = await fetchMarkdown(page);
    const html = markdownRenderer.render(markdown, page);
    const outline = extractPageOutline(html);

    currentOutline = outline;
    document.title = `${page.title} | PFGAP Documentation`;
    renderApplicationShell(renderDocumentPage(page, html, outline));
    bindInteractiveControls();
    updateActiveNavigation(page.route);
    enhanceCodeBlocks();
    updateTableOfContentsHighlight();

    if (restoreHash && window.location.hash) {
      window.requestAnimationFrame(() => {
        const target = document.getElementById(
          decodeURIComponent(window.location.hash.slice(1))
        );
        target?.scrollIntoView();
      });
    } else {
      window.scrollTo({ top: 0, behavior: "instant" });
      document.querySelector("#main-content")?.focus({
        preventScroll: true,
      });
    }
  } catch (error) {
    console.error(error);
    document.title = "Documentation error | PFGAP Documentation";
    renderApplicationShell(
      renderErrorPage(
        "Unable to load documentation",
        "The documentation page could not be loaded. Try refreshing the page or open the source file in the repository."
      )
    );
    bindInteractiveControls();
  }
}

async function buildSearchIndex() {
  const pages = await Promise.all(
    orderedPages.map(async (page) => {
      try {
        const markdown = await fetchMarkdown(page);
        const plainText = markdown
          .replace(/```[\s\S]*?```/g, " ")
          .replace(/`([^`]+)`/g, "$1")
          .replace(/!?\[([^\]]*)]\([^)]*\)/g, "$1")
          .replace(/[#>*_~|]/g, " ")
          .replace(/\s+/g, " ")
          .trim();

        return {
          ...page,
          searchText: `${page.title} ${plainText}`.toLowerCase(),
          excerpt: plainText.slice(0, 220),
        };
      } catch {
        return {
          ...page,
          searchText: page.title.toLowerCase(),
          excerpt: "",
        };
      }
    })
  );

  searchIndex = pages;
  return searchIndex;
}

function ensureSearchIndex() {
  if (searchIndex.length > 0) {
    return Promise.resolve(searchIndex);
  }

  if (!searchIndexPromise) {
    searchIndexPromise = buildSearchIndex();
  }

  return searchIndexPromise;
}

function renderSearchResults(query) {
  const container = document.querySelector("[data-search-results]");

  if (!container) {
    return;
  }

  const normalizedQuery = query.trim().toLowerCase();

  if (!normalizedQuery) {
    container.innerHTML = `
      <p class="search-empty">Type to search all documentation pages.</p>
    `;
    return;
  }

  const terms = normalizedQuery.split(/\s+/).filter(Boolean);
  const results = searchIndex
    .map((page) => {
      const matches = terms.filter((term) =>
        page.searchText.includes(term)
      ).length;
      const titleMatches = terms.filter((term) =>
        page.title.toLowerCase().includes(term)
      ).length;

      return {
        page,
        score: matches + titleMatches * 3,
      };
    })
    .filter((result) => result.score > 0)
    .sort((a, b) =>
      b.score - a.score || a.page.title.localeCompare(b.page.title)
    )
    .slice(0, 12);

  if (results.length === 0) {
    container.innerHTML = `
      <p class="search-empty">
        No documentation pages matched “${escapeHtml(query.trim())}”.
      </p>
    `;
    return;
  }

  container.innerHTML = results.map(({ page }) => `
    <a
      class="search-result"
      href="${escapeHtml(routeToHref(page.route))}"
      data-route="${escapeHtml(page.route)}"
    >
      <strong>${escapeHtml(page.title)}</strong>
      <span>${escapeHtml(page.excerpt)}</span>
    </a>
  `).join("");
}

async function openSearch() {
  const root = document.querySelector("#search-dialog-root");

  if (!root) {
    return;
  }

  root.innerHTML = `
    <div class="search-overlay" data-search-close></div>
    <section
      class="search-dialog"
      role="dialog"
      aria-modal="true"
      aria-labelledby="search-dialog-title"
    >
      <div class="search-dialog-header">
        ${createIcon("search")}
        <label class="visually-hidden" for="documentation-search">
          Search documentation
        </label>
        <input
          id="documentation-search"
          type="search"
          placeholder="Search documentation"
          autocomplete="off"
          spellcheck="false"
          data-search-input
        />
        <button
          class="search-close-button"
          type="button"
          data-search-close
        >
          Esc
        </button>
      </div>
      <h2 id="search-dialog-title" class="visually-hidden">
        Documentation search
      </h2>
      <div class="search-results" data-search-results>
        <p class="search-empty">Preparing search index…</p>
      </div>
    </section>
  `;

  document.body.classList.add("search-open");
  const input = root.querySelector("[data-search-input]");
  input?.focus();

  await ensureSearchIndex();
  renderSearchResults(input?.value ?? "");

  input?.addEventListener("input", (event) => {
    renderSearchResults(event.target.value);
  });
}

function closeSearch() {
  document.body.classList.remove("search-open");
  const root = document.querySelector("#search-dialog-root");

  if (root) {
    root.innerHTML = "";
  }
}

function bindInteractiveControls() {
  document
    .querySelector("[data-theme-toggle]")
    ?.addEventListener("click", toggleTheme);

  document
    .querySelector("[data-mobile-menu-toggle]")
    ?.addEventListener("click", toggleMobileNavigation);

  document
    .querySelector("[data-mobile-menu-close]")
    ?.addEventListener("click", closeMobileNavigation);

  document
    .querySelector("[data-mobile-backdrop]")
    ?.addEventListener("click", closeMobileNavigation);

  document
    .querySelector("[data-search-open]")
    ?.addEventListener("click", openSearch);

  updateThemeControls();
}

const router = createRouter({
  baseUrl: BASE_URL,
  onRouteChange: (route, options) => loadRoute(route, options),
});

document.addEventListener("click", (event) => {
  const searchClose = event.target.closest("[data-search-close]");
  if (searchClose) {
    closeSearch();
    return;
  }

  const routeLink = event.target.closest("a[data-route]");
  if (!routeLink) {
    return;
  }

  const route = routeLink.dataset.route;
  if (!route) {
    return;
  }

  event.preventDefault();
  closeSearch();
  closeMobileNavigation();
  router.navigate(route);
});

document.addEventListener("keydown", (event) => {
  const target = event.target;
  const editing = target instanceof HTMLInputElement
    || target instanceof HTMLTextAreaElement
    || target?.isContentEditable;

  if (event.key === "Escape") {
    closeSearch();
    closeMobileNavigation();
    return;
  }

  if (event.key === "/" && !editing) {
    event.preventDefault();
    openSearch();
  }
});

window.addEventListener("scroll", () => {
  window.requestAnimationFrame(updateTableOfContentsHighlight);
}, { passive: true });

window.addEventListener("resize", () => {
  if (window.innerWidth >= 1024) {
    closeMobileNavigation();
  }
});

router.start();
