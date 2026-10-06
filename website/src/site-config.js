const DEFAULT_PROJECT_REPOSITORY =
  "https://github.com/KevinMoonLab/PF-GAP";
const DEFAULT_EDIT_REPOSITORY = DEFAULT_PROJECT_REPOSITORY;
const DEFAULT_BRANCH = "main";

function normalizeRepositoryUrl(value, fallback) {
  const candidate = String(value || fallback).trim();

  if (!candidate) {
    return fallback;
  }

  return candidate.replace(/\/+$/, "");
}

function normalizeBranch(value) {
  const candidate = String(value || DEFAULT_BRANCH).trim();
  return candidate || DEFAULT_BRANCH;
}

export const siteConfig = Object.freeze({
  projectRepository: normalizeRepositoryUrl(
    import.meta.env.VITE_PROJECT_REPOSITORY,
    DEFAULT_PROJECT_REPOSITORY
  ),

  editRepository: normalizeRepositoryUrl(
    import.meta.env.VITE_EDIT_REPOSITORY,
    DEFAULT_EDIT_REPOSITORY
  ),

  defaultBranch: normalizeBranch(
    import.meta.env.VITE_REPOSITORY_BRANCH
  ),
});

export function repositoryFileUrl(
  relativePath,
  {
    repository = siteConfig.projectRepository,
    branch = siteConfig.defaultBranch,
    mode = "blob",
  } = {}
) {
  const cleanRepository = normalizeRepositoryUrl(
    repository,
    siteConfig.projectRepository
  );
  const cleanBranch = encodeURIComponent(normalizeBranch(branch));
  const cleanPath = String(relativePath)
    .replaceAll("\\", "/")
    .replace(/^\/+/, "")
    .split("/")
    .map((segment) => encodeURIComponent(segment))
    .join("/");
  const operation = mode === "edit" ? "edit" : "blob";

  return `${cleanRepository}/${operation}/${cleanBranch}/${cleanPath}`;
}
