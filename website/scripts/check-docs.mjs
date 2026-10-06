import {
  access,
  readdir,
  readFile,
  stat,
} from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const scriptFile = fileURLToPath(import.meta.url);
const scriptDirectory = path.dirname(scriptFile);
const websiteDirectory = path.resolve(scriptDirectory, "..");
const repositoryDirectory = path.resolve(websiteDirectory, "..");
const documentationDirectory = path.join(
  repositoryDirectory,
  "docs"
);
const navigationFile = path.join(
  websiteDirectory,
  "src",
  "navigation.js"
);

const MARKDOWN_EXTENSION = ".md";
const EXTERNAL_SCHEMES = new Set([
  "http:",
  "https:",
  "mailto:",
  "tel:",
]);

const errors = [];
const warnings = [];

function toPosixPath(value) {
  return value.split(path.sep).join("/");
}

function reportError(file, message) {
  errors.push({ file: toPosixPath(file), message });
}

function reportWarning(file, message) {
  warnings.push({ file: toPosixPath(file), message });
}

async function pathExists(target) {
  try {
    await access(target);
    return true;
  } catch {
    return false;
  }
}

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

    if (
      entry.isFile()
      && entry.name.toLowerCase().endsWith(MARKDOWN_EXTENSION)
    ) {
      files.push(path.relative(root, absolutePath));
    }
  }

  return files.sort((first, second) =>
    toPosixPath(first).localeCompare(toPosixPath(second))
  );
}

function stripFencedCode(markdown) {
  return markdown.replace(
    /(^|\n)(```|~~~)[^\n]*\n[\s\S]*?\n\2(?=\n|$)/g,
    "$1"
  );
}

function stripInlineCode(markdown) {
  return markdown.replace(/`[^`\n]*`/g, "");
}

function normalizeHeadingText(value) {
  return value
    .replace(/\[([^\]]+)]\([^)]*\)/g, "$1")
    .replace(/[*_~]/g, "")
    .replace(/<[^>]+>/g, "")
    .trim();
}

function slugifyHeading(value) {
  return normalizeHeadingText(value)
    .toLowerCase()
    .normalize("NFKD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/&amp;/g, " and ")
    .replace(/[^\p{Letter}\p{Number}\s-]/gu, "")
    .trim()
    .replace(/\s+/g, "-")
    .replace(/-+/g, "-");
}

function extractHeadings(markdown) {
  const withoutCode = stripFencedCode(markdown);
  const headings = [];
  const slugCounts = new Map();
  const lines = withoutCode.split(/\r?\n/);

  for (let index = 0; index < lines.length; index += 1) {
    const line = lines[index];
    const atx = line.match(/^(#{1,6})\s+(.+?)\s*#*\s*$/);

    let level;
    let text;

    if (atx) {
      level = atx[1].length;
      text = atx[2];
    } else if (
      index + 1 < lines.length
      && /^=+\s*$/.test(lines[index + 1])
      && line.trim()
    ) {
      level = 1;
      text = line.trim();
      index += 1;
    } else if (
      index + 1 < lines.length
      && /^-+\s*$/.test(lines[index + 1])
      && line.trim()
    ) {
      level = 2;
      text = line.trim();
      index += 1;
    } else {
      continue;
    }

    const baseSlug = slugifyHeading(text);
    const previousCount = slugCounts.get(baseSlug) ?? 0;
    slugCounts.set(baseSlug, previousCount + 1);

    const slug = previousCount === 0
      ? baseSlug
      : `${baseSlug}-${previousCount}`;

    headings.push({
      level,
      text: normalizeHeadingText(text),
      slug,
    });
  }

  return headings;
}

function extractMarkdownLinks(markdown) {
  const withoutCode = stripInlineCode(
    stripFencedCode(markdown)
  );
  const links = [];
  const regularLink = /!?\[[^\]]*]\(([^)]+)\)/g;
  const referenceDefinition = /^\s*\[[^\]]+]:\s*(\S+)/gm;

  for (const match of withoutCode.matchAll(regularLink)) {
    links.push(match[1].trim());
  }

  for (const match of withoutCode.matchAll(referenceDefinition)) {
    links.push(match[1].trim());
  }

  return links;
}

function unwrapDestination(destination) {
  let value = destination.trim();

  if (value.startsWith("<") && value.endsWith(">")) {
    value = value.slice(1, -1);
  }

  const titleSeparator = value.match(/\s+["'(]/);
  if (titleSeparator?.index !== undefined) {
    value = value.slice(0, titleSeparator.index);
  }

  return value;
}

function isExternalDestination(destination) {
  if (
    destination.startsWith("//")
    || destination.startsWith("data:")
  ) {
    return true;
  }

  try {
    const url = new URL(destination);
    return EXTERNAL_SCHEMES.has(url.protocol);
  } catch {
    return false;
  }
}

function splitDestination(destination) {
  const hashIndex = destination.indexOf("#");

  if (hashIndex < 0) {
    return {
      filePart: destination,
      fragment: "",
    };
  }

  return {
    filePart: destination.slice(0, hashIndex),
    fragment: destination.slice(hashIndex + 1),
  };
}

function decodeSafely(value) {
  try {
    return decodeURIComponent(value);
  } catch {
    return value;
  }
}

async function checkLocalLink(
  sourceRelativePath,
  destination,
  headingsByFile
) {
  const normalizedDestination = unwrapDestination(destination);

  if (
    !normalizedDestination
    || isExternalDestination(normalizedDestination)
  ) {
    return;
  }

  const { filePart, fragment } = splitDestination(
    normalizedDestination
  );
  const sourceDirectory = path.dirname(
    path.join(documentationDirectory, sourceRelativePath)
  );

  let targetPath;

  if (!filePart) {
    targetPath = path.join(
      documentationDirectory,
      sourceRelativePath
    );
  } else if (filePart.startsWith("/")) {
    targetPath = path.join(
      repositoryDirectory,
      decodeSafely(filePart.slice(1))
    );
  } else {
    targetPath = path.resolve(
      sourceDirectory,
      decodeSafely(filePart)
    );
  }

  const relativeTarget = path.relative(
    repositoryDirectory,
    targetPath
  );

  if (
    relativeTarget.startsWith("..")
    || path.isAbsolute(relativeTarget)
  ) {
    reportError(
      sourceRelativePath,
      `Link escapes the repository: ${destination}`
    );
    return;
  }

  if (!(await pathExists(targetPath))) {
    reportError(
      sourceRelativePath,
      `Link target does not exist: ${destination}`
    );
    return;
  }

  if (!fragment || path.extname(targetPath).toLowerCase() !== ".md") {
    return;
  }

  const targetRelativeToDocs = path.relative(
    documentationDirectory,
    targetPath
  );

  if (
    targetRelativeToDocs.startsWith("..")
    || path.isAbsolute(targetRelativeToDocs)
  ) {
    return;
  }

  const headingSlugs = headingsByFile.get(
    toPosixPath(targetRelativeToDocs)
  );

  if (!headingSlugs) {
    return;
  }

  const decodedFragment = decodeSafely(fragment).toLowerCase();

  if (!headingSlugs.has(decodedFragment)) {
    reportError(
      sourceRelativePath,
      `Heading fragment does not exist: ${destination}`
    );
  }
}

function checkDocumentStructure(relativePath, markdown, headings) {
  const levelOneHeadings = headings.filter(
    (heading) => heading.level === 1
  );

  if (levelOneHeadings.length === 0) {
    reportError(relativePath, "Document has no level-one heading.");
  } else if (levelOneHeadings.length > 1) {
    reportError(
      relativePath,
      "Document has more than one level-one heading."
    );
  }

  if (/\r(?!\n)/.test(markdown)) {
    reportWarning(
      relativePath,
      "Document contains standalone carriage returns."
    );
  }

  const trailingWhitespaceLines = markdown
    .split(/\r?\n/)
    .reduce((count, line) =>
      /[ \t]+$/.test(line) ? count + 1 : count,
    0);

  if (trailingWhitespaceLines > 0) {
    reportWarning(
      relativePath,
      `${trailingWhitespaceLines} line(s) contain trailing whitespace.`
    );
  }

  const seenSlugs = new Set();
  for (const heading of headings) {
    if (!heading.slug) {
      reportError(
        relativePath,
        `Heading cannot produce a URL fragment: ${heading.text}`
      );
      continue;
    }

    if (seenSlugs.has(heading.slug)) {
      reportWarning(
        relativePath,
        `Duplicate heading fragment generated: #${heading.slug}`
      );
    }

    seenSlugs.add(heading.slug);
  }
}

async function checkNavigationCoverage(markdownFiles) {
  if (!(await pathExists(navigationFile))) {
    reportWarning(
      "website/src/navigation.js",
      "Navigation validation was skipped because the file does not exist yet."
    );
    return;
  }

  const navigationSource = await readFile(navigationFile, "utf8");

  for (const relativePath of markdownFiles) {
    const posixPath = toPosixPath(relativePath);

    if (!navigationSource.includes(posixPath)) {
      reportError(
        "website/src/navigation.js",
        `Documentation page is missing from navigation: ${posixPath}`
      );
    }
  }
}

function printFindings(label, findings) {
  if (findings.length === 0) {
    return;
  }

  console.log(`\n${label}:`);

  for (const finding of findings) {
    console.log(
      `- ${finding.file}: ${finding.message}`
    );
  }
}

async function checkDocumentation() {
  await requireDirectory(
    documentationDirectory,
    "Documentation directory"
  );

  const markdownFiles = await collectMarkdownFiles(
    documentationDirectory
  );

  if (markdownFiles.length === 0) {
    throw new Error(
      `No Markdown files were found under ${documentationDirectory}`
    );
  }

  if (!markdownFiles.some((file) =>
    toPosixPath(file).toLowerCase() === "index.md"
  )) {
    reportError("docs/index.md", "Documentation home page is missing.");
  }

  const documents = new Map();
  const headingsByFile = new Map();

  for (const relativePath of markdownFiles) {
    const normalizedPath = toPosixPath(relativePath);
    const absolutePath = path.join(
      documentationDirectory,
      relativePath
    );
    const markdown = await readFile(absolutePath, "utf8");
    const headings = extractHeadings(markdown);

    documents.set(normalizedPath, markdown);
    headingsByFile.set(
      normalizedPath,
      new Set(headings.map((heading) => heading.slug))
    );

    checkDocumentStructure(
      normalizedPath,
      markdown,
      headings
    );
  }

  for (const [relativePath, markdown] of documents) {
    const links = extractMarkdownLinks(markdown);

    for (const destination of links) {
      await checkLocalLink(
        relativePath,
        destination,
        headingsByFile
      );
    }
  }

  await checkNavigationCoverage(markdownFiles);

  printFindings("Warnings", warnings);
  printFindings("Errors", errors);

  console.log(
    `\nChecked ${markdownFiles.length} Markdown files.`
  );
  console.log(`Warnings: ${warnings.length}`);
  console.log(`Errors: ${errors.length}`);

  if (errors.length > 0) {
    process.exitCode = 1;
    return;
  }

  console.log("Documentation checks passed.");
}

checkDocumentation().catch((error) => {
  console.error("Documentation checks failed unexpectedly.");
  console.error(error);
  process.exitCode = 1;
});
