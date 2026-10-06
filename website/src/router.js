const DEFAULT_BASE_URL = import.meta.env.BASE_URL || "/";

function normalizeBaseUrl(value = DEFAULT_BASE_URL) {
  if (!value || value === "/") {
    return "/";
  }

  if (value === "." || value === "./") {
    return "./";
  }

  const trimmed = value.trim();
  const withLeadingSlash = trimmed.startsWith("/")
    ? trimmed
    : `/${trimmed}`;

  return withLeadingSlash.endsWith("/")
    ? withLeadingSlash
    : `${withLeadingSlash}/`;
}

function stripQueryAndHash(value) {
  return value.split(/[?#]/, 1)[0];
}

function collapseSlashes(value) {
  return value.replace(/\/{2,}/g, "/");
}

export function normalizeRoute(value = "/") {
  let route = String(value).trim();

  if (!route) {
    return "/";
  }

  try {
    if (/^[a-z][a-z\d+.-]*:/i.test(route)) {
      route = new URL(route).pathname;
    }
  } catch {
    return "/";
  }

  route = stripQueryAndHash(route);
  route = collapseSlashes(route.replaceAll("\\", "/"));

  if (!route.startsWith("/")) {
    route = `/${route}`;
  }

  if (route.length > 1 && route.endsWith("/")) {
    route = route.replace(/\/+$/, "");
  }

  return route || "/";
}

function pathnameFromRoute(route, baseUrl = DEFAULT_BASE_URL) {
  const normalizedRoute = normalizeRoute(route);
  const normalizedBase = normalizeBaseUrl(baseUrl);

  if (normalizedBase === "./") {
    return normalizedRoute === "/"
      ? "./"
      : `.${normalizedRoute}`;
  }

  if (normalizedBase === "/") {
    return normalizedRoute;
  }

  if (normalizedRoute === "/") {
    return normalizedBase;
  }

  return `${normalizedBase.replace(/\/$/, "")}${normalizedRoute}`;
}

function routeFromPathname(pathname, baseUrl = DEFAULT_BASE_URL) {
  const normalizedBase = normalizeBaseUrl(baseUrl);
  let route = collapseSlashes(pathname || "/");

  if (normalizedBase !== "/" && normalizedBase !== "./") {
    const baseWithoutTrailingSlash = normalizedBase.replace(/\/$/, "");

    if (route === baseWithoutTrailingSlash) {
      return "/";
    }

    if (route.startsWith(normalizedBase)) {
      route = route.slice(normalizedBase.length - 1);
    }
  }

  return normalizeRoute(route);
}

export function routeToHref(
  route,
  {
    baseUrl = DEFAULT_BASE_URL,
    hash = "",
  } = {}
) {
  const pathname = pathnameFromRoute(route, baseUrl);
  const normalizedHash = hash
    ? `#${String(hash).replace(/^#/, "")}`
    : "";

  return `${pathname}${normalizedHash}`;
}

function isModifiedNavigation(event) {
  return event.metaKey
    || event.ctrlKey
    || event.shiftKey
    || event.altKey;
}

function isPrimaryButton(event) {
  return event.button === 0;
}

function shouldHandleAnchor(anchor, event) {
  if (
    event.defaultPrevented
    || !isPrimaryButton(event)
    || isModifiedNavigation(event)
  ) {
    return false;
  }

  if (
    anchor.target
    || anchor.hasAttribute("download")
    || anchor.getAttribute("rel")?.includes("external")
  ) {
    return false;
  }

  const href = anchor.getAttribute("href");

  if (!href || href.startsWith("mailto:") || href.startsWith("tel:")) {
    return false;
  }

  const target = new URL(anchor.href, window.location.href);

  return target.origin === window.location.origin;
}

export function createRouter({
  baseUrl = DEFAULT_BASE_URL,
  onRouteChange,
} = {}) {
  if (typeof onRouteChange !== "function") {
    throw new TypeError(
      "createRouter requires an onRouteChange callback."
    );
  }

  const normalizedBase = normalizeBaseUrl(baseUrl);
  let started = false;

  function currentRoute() {
    return routeFromPathname(
      window.location.pathname,
      normalizedBase
    );
  }

  function notify(options = {}) {
    return onRouteChange(currentRoute(), options);
  }

  function navigate(
    route,
    {
      replace = false,
      hash = "",
      state = null,
      restoreHash = false,
    } = {}
  ) {
    const normalizedRoute = normalizeRoute(route);
    const href = routeToHref(normalizedRoute, {
      baseUrl: normalizedBase,
      hash,
    });
    const method = replace ? "replaceState" : "pushState";

    window.history[method](
      {
        ...(state ?? {}),
        route: normalizedRoute,
      },
      "",
      href
    );

    return onRouteChange(normalizedRoute, {
      restoreHash,
      navigationType: replace ? "replace" : "push",
    });
  }

  function replace(route, options = {}) {
    return navigate(route, {
      ...options,
      replace: true,
    });
  }

  function handlePopState() {
    notify({
      restoreHash: true,
      navigationType: "pop",
    });
  }

  function handleDocumentClick(event) {
    const anchor = event.target.closest("a[href]");

    if (!anchor || anchor.hasAttribute("data-route")) {
      return;
    }

    if (!shouldHandleAnchor(anchor, event)) {
      return;
    }

    const target = new URL(anchor.href, window.location.href);
    const targetRoute = routeFromPathname(
      target.pathname,
      normalizedBase
    );
    const activeRoute = currentRoute();

    if (
      targetRoute === activeRoute
      && target.hash
      && target.pathname === window.location.pathname
    ) {
      return;
    }

    const basePath = normalizedBase === "./"
      ? window.location.pathname
      : normalizedBase;
    const belongsToApplication = normalizedBase === "./"
      || target.pathname === basePath.replace(/\/$/, "")
      || target.pathname.startsWith(basePath);

    if (!belongsToApplication) {
      return;
    }

    event.preventDefault();
    navigate(targetRoute, {
      hash: target.hash,
      restoreHash: Boolean(target.hash),
    });
  }

  function start() {
    if (started) {
      return;
    }

    started = true;
    window.addEventListener("popstate", handlePopState);
    document.addEventListener("click", handleDocumentClick);

    notify({
      restoreHash: true,
      navigationType: "initial",
    });
  }

  function stop() {
    if (!started) {
      return;
    }

    started = false;
    window.removeEventListener("popstate", handlePopState);
    document.removeEventListener("click", handleDocumentClick);
  }

  return Object.freeze({
    start,
    stop,
    navigate,
    replace,
    currentRoute,
    href: (route, options = {}) => routeToHref(route, {
      baseUrl: normalizedBase,
      ...options,
    }),
  });
}
