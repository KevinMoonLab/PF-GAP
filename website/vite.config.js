import { defineConfig, loadEnv } from "vite";

function normalizeBasePath(value) {
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

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  const base = normalizeBasePath(
    env.PFGAP_DOCS_BASE || "/"
  );

  return {
    base,

    build: {
      outDir: "dist",
      emptyOutDir: true,
      assetsDir: "assets",
      sourcemap: true,
      target: "baseline-widely-available",
    },

    server: {
      host: "127.0.0.1",
      port: 5173,
      strictPort: true,
      open: true,
    },

    preview: {
      host: "127.0.0.1",
      port: 4173,
      strictPort: true,
      open: true,
    },
  };
});
