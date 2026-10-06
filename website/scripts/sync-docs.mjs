import {
  cp,
  mkdir,
  readdir,
  readFile,
  rm,
  stat,
  writeFile,
} from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const scriptFile = fileURLToPath(import.meta.url);
const scriptDirectory = path.dirname(scriptFile);
const websiteDirectory = path.resolve(scriptDirectory, "..");
const repositoryDirectory = path.resolve(websiteDirectory, "..");
const sourceDirectory = path.join(repositoryDirectory, "docs");
const publicDirectory = path.join(websiteDirectory, "public");
const targetDirectory = path.join(publicDirectory, "docs");
const manifestPath = path.join(publicDirectory, "docs-manifest.json");

async function requireDirectory(directory, description) {
  let details;

  try {
    details = await stat(directory);
  } catch (error) {
    if (error?.code === "ENOENT") {
      throw new Error(
        `${description} does not exist: ${directory}`,
        { cause: error }
      );
    }

    throw error;
  }

  if (!details.isDirectory()) {
    throw new Error(
      `${description} is not a directory: ${directory}`
    );
  }
}

function toPosixPath(value) {
  return value.split(path.sep).join("/");
}

function titleFromMarkdown(markdown, relativePath) {
  const heading = markdown.match(/^#\s+(.+?)\s*$/m);

  if (heading) {
    return heading[1].trim();
  }

  return path
    .basename(relativePath, path.extname(relativePath))
    .replaceAll("_", " ");
}

function routeFromRelativePath(relativePath) {
  const normalized = toPosixPath(relativePath);

  if (normalized.toLowerCase() === "index.md") {
    return "/";
  }

  const withoutExtension = normalized.replace(/\.md$/i, "");
  const route = withoutExtension
    .split("/")
    .map((segment) =>
      segment
        .replaceAll("_", "-")
        .replace(/([a-z0-9])([A-Z])/g, "$1-$2")
        .toLowerCase()
    )
    .join("/");

  return `/${route}`;
}

async function collectMarkdownFiles(directory, root = directory) {
  const entries = await readdir(directory, {
    withFileTypes: true,
  });

  const files = [];

  for (const entry of entries) {
    if (entry.name.startsWith(".")) {
      continue;
    }

    const absolutePath = path.join(directory, entry.name);

    if (entry.isDirectory()) {
      files.push(
        ...(await collectMarkdownFiles(absolutePath, root))
      );
      continue;
    }

    if (entry.isFile() && entry.name.endsWith(".md")) {
      files.push(path.relative(root, absolutePath));
    }
  }

  return files.sort((first, second) =>
    toPosixPath(first).localeCompare(toPosixPath(second))
  );
}

async function createManifest(relativePaths) {
  const pages = [];

  for (const relativePath of relativePaths) {
    const absolutePath = path.join(
      sourceDirectory,
      relativePath
    );
    const markdown = await readFile(absolutePath, "utf8");

    pages.push({
      source: toPosixPath(relativePath),
      publicPath: `docs/${toPosixPath(relativePath)}`,
      route: routeFromRelativePath(relativePath),
      title: titleFromMarkdown(markdown, relativePath),
    });
  }

  return {
    generatedBy: "website/scripts/sync-docs.mjs",
    pageCount: pages.length,
    pages,
  };
}

async function syncDocumentation() {
  await requireDirectory(
    sourceDirectory,
    "Documentation source directory"
  );

  const markdownFiles = await collectMarkdownFiles(
    sourceDirectory
  );

  if (markdownFiles.length === 0) {
    throw new Error(
      `No Markdown files were found under ${sourceDirectory}`
    );
  }

  if (!markdownFiles.some((file) =>
    toPosixPath(file).toLowerCase() === "index.md"
  )) {
    throw new Error(
      `Documentation home page is missing: ${path.join(
        sourceDirectory,
        "index.md"
      )}`
    );
  }

  await mkdir(publicDirectory, { recursive: true });
  await rm(targetDirectory, {
    recursive: true,
    force: true,
  });

  await cp(sourceDirectory, targetDirectory, {
    recursive: true,
    force: true,
    filter(source) {
      return !path.basename(source).startsWith(".");
    },
  });

  const manifest = await createManifest(markdownFiles);

  await writeFile(
    manifestPath,
    `${JSON.stringify(manifest, null, 2)}\n`,
    "utf8"
  );

  console.log(
    `Synchronized ${manifest.pageCount} documentation pages.`
  );
  console.log(
    `Source: ${sourceDirectory}`
  );
  console.log(
    `Target: ${targetDirectory}`
  );
  console.log(
    `Manifest: ${manifestPath}`
  );
}

syncDocumentation().catch((error) => {
  console.error("Documentation synchronization failed.");
  console.error(error);
  process.exitCode = 1;
});
