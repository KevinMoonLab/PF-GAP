import {
  access,
  copyFile,
  writeFile,
} from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const scriptFile = fileURLToPath(import.meta.url);
const scriptDirectory = path.dirname(scriptFile);
const websiteDirectory = path.resolve(scriptDirectory, "..");
const outputDirectory = path.join(websiteDirectory, "dist");
const indexPath = path.join(outputDirectory, "index.html");
const fallbackPath = path.join(outputDirectory, "404.html");
const noJekyllPath = path.join(outputDirectory, ".nojekyll");

async function requireFile(filePath, description) {
  try {
    await access(filePath);
  } catch (error) {
    throw new Error(
      `${description} does not exist: ${filePath}`,
      { cause: error }
    );
  }
}

async function createPagesFallback() {
  await requireFile(
    indexPath,
    "Built website entry point"
  );

  await copyFile(indexPath, fallbackPath);
  await writeFile(noJekyllPath, "", "utf8");

  console.log(`Created GitHub Pages fallback: ${fallbackPath}`);
  console.log(`Created GitHub Pages marker: ${noJekyllPath}`);
}

createPagesFallback().catch((error) => {
  console.error("Could not create the GitHub Pages fallback.");
  console.error(error);
  process.exitCode = 1;
});
