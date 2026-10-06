import DOMPurify from "dompurify";
import hljs from "highlight.js/lib/common";
import { Marked } from "marked";
import { markedHighlight } from "marked-highlight";

const MARKDOWN_EXTENSION = ".md";
const HEADING_SELECTOR = "h1, h2, h3, h4, h5, h6";
const OUTLINE_SELECTOR = "h2, h3";
const EXTERNAL_SCHEMES = new Set([
  "http:",
  "https:",
  "mailto:",
  "tel:",
]);

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function normalizeBaseUrl(value) {
  if (!value || value === "/") {
    return "/";
  }

  if (value === "." || value === "./") {
    return "./";
  }

  const trimmed = String(value).trim();
  const withLeadingSlash = trimmed.startsWith("/")
    ? trimmed
    : `/${trimmed}`;

  return withLeadingSlash.endsWith("/")
    ? withLeadingSlash
    : `${withLeadingSlash}/`;
}

function joinBaseUrl(baseUrl, relativePath) {
  const cleanPath = String(relativePath).replace(/^\/+/, "");
  return `${normalizeBaseUrl(baseUrl)}${cleanPath}`;
}

function toPosixPath(value) {
  return String(value).replaceAll("\\", "/");
}

function normalizeSourcePath(value) {
  const segments = toPosixPath(value).split("/");
  const normalized = [];

  for (const segment of segments) {
    if (!segment || segment === ".") {
      continue;
    }

    if (segment === "..") {
      if (normalized.length === 0) {
        return null;
      }

      normalized.pop();
      continue;
    }

    normalized.push(segment);
  }

  return normalized.join("/");
}

function sourceDirectory(source) {
  const normalized = toPosixPath(source);
  const separator = normalized.lastIndexOf("/");

  return separator < 0
    ? ""
    : normalized.slice(0, separator);
}

function splitDestination(destination) {
  const hashIndex = destination.indexOf("#");
  const queryIndex = destination.indexOf("?");
  const indexes = [hashIndex, queryIndex]
    .filter((index) => index >= 0);
  const boundary = indexes.length > 0
    ? Math.min(...indexes)
    : destination.length;
  const filePart = destination.slice(0, boundary);
  const suffix = destination.slice(boundary);

  return { filePart, suffix };
}

function splitSuffix(suffix) {
  const hashIndex = suffix.indexOf("#");

  if (hashIndex < 0) {
    return {
      query: suffix,
      hash: "",
    };
  }

  return {
    query: suffix.slice(0, hashIndex),
    hash: suffix.slice(hashIndex),
  };
}

function isExternalDestination(destination) {
  if (
    destination.startsWith("//")
    || destination.startsWith("data:")
  ) {
    return true;
  }

  try {
    const parsed = new URL(destination);
    return EXTERNAL_SCHEMES.has(parsed.protocol);
  } catch {
    return false;
  }
}

function isMarkdownPath(value) {
  return value.toLowerCase().endsWith(MARKDOWN_EXTENSION);
}

function slugify(value) {
  return String(value)
    .toLowerCase()
    .normalize("NFKD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/&/g, " and ")
    .replace(/[^\p{Letter}\p{Number}\s-]/gu, "")
    .trim()
    .replace(/\s+/g, "-")
    .replace(/-+/g, "-");
}

function addHeadingIdentifiers(document) {
  const counts = new Map();

  for (const heading of document.querySelectorAll(HEADING_SELECTOR)) {
    const explicitId = heading.getAttribute("id")?.trim();
    const base = explicitId || slugify(heading.textContent ?? "");

    if (!base) {
      continue;
    }

    const count = counts.get(base) ?? 0;
    counts.set(base, count + 1);

    const id = count === 0 ? base : `${base}-${count}`;
    heading.id = id;

    if (heading.tagName !== "H1") {
      const anchor = document.createElement("a");
      anchor.className = "heading-anchor";
      anchor.href = `#${id}`;
      anchor.setAttribute(
        "aria-label",
        `Link to ${heading.textContent?.trim() || "section"}`
      );
      anchor.textContent = "#";
      heading.append(anchor);
    }
  }
}

function rewriteRepositoryRelativeLink(anchor, destination, repositoryUrl) {
  if (!repositoryUrl) {
    anchor.href = destination;
    anchor.dataset.repositoryRelative = "true";
    return;
  }

  const cleanRepositoryUrl = repositoryUrl.replace(/\/$/, "");
  const cleanDestination = destination.replace(/^\.\.\//, "");
  anchor.href = `${cleanRepositoryUrl}/blob/main/${cleanDestination}`;
  anchor.target = "_blank";
  anchor.rel = "noreferrer";
}

function rewriteAnchor({
  anchor,
  page,
  pageBySource,
  routeToHref,
  repositoryUrl,
}) {
  const rawHref = anchor.getAttribute("href")?.trim();

  if (!rawHref) {
    return;
  }

  if (rawHref.startsWith("#")) {
    return;
  }

  if (isExternalDestination(rawHref)) {
    if (rawHref.startsWith("http://") || rawHref.startsWith("https://")) {
      anchor.target = "_blank";
      anchor.rel = "noreferrer";
    }
    return;
  }

  const { filePart, suffix } = splitDestination(rawHref);

  if (!filePart || !isMarkdownPath(filePart)) {
    return;
  }

  const combined = sourceDirectory(page.source)
    ? `${sourceDirectory(page.source)}/${filePart}`
    : filePart;
  const normalizedSource = normalizeSourcePath(combined);

  if (!normalizedSource) {
    rewriteRepositoryRelativeLink(
      anchor,
      rawHref,
      repositoryUrl
    );
    return;
  }

  const targetPage = pageBySource.get(normalizedSource);

  if (!targetPage) {
    rewriteRepositoryRelativeLink(
      anchor,
      rawHref,
      repositoryUrl
    );
    return;
  }

  const { query, hash } = splitSuffix(suffix);
  const cleanHash = hash.replace(/^#/, "");
  const routeHref = routeToHref(targetPage.route, {
    hash: cleanHash,
  });

  anchor.href = `${routeHref}${query}`;
  anchor.dataset.route = targetPage.route;
}

function rewriteImage({
  image,
  page,
  baseUrl,
  repositoryUrl,
}) {
  const rawSource = image.getAttribute("src")?.trim();

  if (!rawSource || isExternalDestination(rawSource)) {
    return;
  }

  const combined = sourceDirectory(page.source)
    ? `${sourceDirectory(page.source)}/${rawSource}`
    : rawSource;
  const normalizedSource = normalizeSourcePath(combined);

  if (normalizedSource) {
    image.src = joinBaseUrl(
      baseUrl,
      `docs/${normalizedSource}`
    );
    image.loading = "lazy";
    image.decoding = "async";
    return;
  }

  if (repositoryUrl) {
    const cleanRepositoryUrl = repositoryUrl.replace(/\/$/, "");
    const cleanSource = rawSource.replace(/^\.\.\//, "");
    image.src = `${cleanRepositoryUrl}/raw/main/${cleanSource}`;
  }
}

function decorateExternalLinks(document) {
  for (const anchor of document.querySelectorAll("a[href]")) {
    const href = anchor.getAttribute("href") ?? "";

    if (!href.startsWith("http://") && !href.startsWith("https://")) {
      continue;
    }

    anchor.classList.add("external-link");
  }
}

function wrapTables(document) {
  for (const table of document.querySelectorAll("table")) {
    if (table.parentElement?.classList.contains("table-scroll")) {
      continue;
    }

    const wrapper = document.createElement("div");
    wrapper.className = "table-scroll";
    wrapper.setAttribute("role", "region");
    wrapper.setAttribute("aria-label", "Scrollable table");
    wrapper.tabIndex = 0;

    table.before(wrapper);
    wrapper.append(table);
  }
}

function sanitizeRenderedHtml(html) {
  return DOMPurify.sanitize(html, {
    USE_PROFILES: {
      html: true,
    },
    ADD_ATTR: [
      "aria-label",
      "aria-hidden",
      "class",
      "id",
      "target",
      "rel",
      "role",
      "tabindex",
    ],
    FORBID_TAGS: [
      "base",
      "form",
      "iframe",
      "input",
      "object",
      "script",
      "style",
      "textarea",
    ],
  });
}

function createMarkedInstance() {
  const marked = new Marked(
    markedHighlight({
      emptyLangClass: "hljs language-plaintext",
      langPrefix: "hljs language-",
      highlight(code, language) {
        const normalizedLanguage = language?.trim().toLowerCase();

        if (
          normalizedLanguage
          && hljs.getLanguage(normalizedLanguage)
        ) {
          return hljs.highlight(code, {
            language: normalizedLanguage,
          }).value;
        }

        return escapeHtml(code);
      },
    })
  );

  marked.setOptions({
    gfm: true,
    breaks: false,
  });

  return marked;
}

export function createMarkdownRenderer({
  baseUrl = import.meta.env.BASE_URL || "/",
  pageBySource,
  routeToHref,
  repositoryUrl = "",
} = {}) {
  if (!(pageBySource instanceof Map)) {
    throw new TypeError(
      "createMarkdownRenderer requires pageBySource to be a Map."
    );
  }

  if (typeof routeToHref !== "function") {
    throw new TypeError(
      "createMarkdownRenderer requires routeToHref to be a function."
    );
  }

  const marked = createMarkedInstance();

  function render(markdown, page) {
    if (!page?.source) {
      throw new TypeError(
        "Markdown rendering requires a page with a source path."
      );
    }

    const rendered = marked.parse(String(markdown));

    if (rendered instanceof Promise) {
      throw new Error(
        "The configured Markdown pipeline unexpectedly became asynchronous."
      );
    }

    const sanitized = sanitizeRenderedHtml(rendered);
    const document = new DOMParser().parseFromString(
      `<main>${sanitized}</main>`,
      "text/html"
    );
    const root = document.querySelector("main");

    if (!root) {
      throw new Error("Could not create the rendered Markdown root.");
    }

    addHeadingIdentifiers(document);

    for (const anchor of root.querySelectorAll("a[href]")) {
      rewriteAnchor({
        anchor,
        page,
        pageBySource,
        routeToHref,
        repositoryUrl,
      });
    }

    for (const image of root.querySelectorAll("img[src]")) {
      rewriteImage({
        image,
        page,
        baseUrl,
        repositoryUrl,
      });
    }

    decorateExternalLinks(document);
    wrapTables(document);

    return root.innerHTML;
  }

  return Object.freeze({ render });
}

export function extractPageOutline(html) {
  const document = new DOMParser().parseFromString(
    `<main>${String(html)}</main>`,
    "text/html"
  );

  return [...document.querySelectorAll(OUTLINE_SELECTOR)]
    .filter((heading) => heading.id)
    .map((heading) => {
      const clone = heading.cloneNode(true);
      clone.querySelector(".heading-anchor")?.remove();

      return Object.freeze({
        id: heading.id,
        level: Number(heading.tagName.slice(1)),
        text: clone.textContent?.trim() ?? "",
      });
    })
    .filter((heading) => heading.text);
}
