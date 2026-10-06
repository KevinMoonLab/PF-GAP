const page = (title, route, source, description) =>
  Object.freeze({
    title,
    route,
    source,
    description,
  });

export const navigationGroups = Object.freeze([
  Object.freeze({
    title: "Overview",
    pages: Object.freeze([
      page(
        "Documentation Home",
        "/",
        "index.md",
        "Start here for an overview of PFGAP and its documentation."
      ),
    ]),
  }),

  Object.freeze({
    title: "Getting Started",
    pages: Object.freeze([
      page(
        "Installation",
        "/getting-started/installation",
        "getting-started/Installation.md",
        "Install PFGAP and prepare the Java and Python environments."
      ),
      page(
        "Quick Start",
        "/getting-started/quick-start",
        "getting-started/Quick_Start.md",
        "Run a minimal end-to-end PFGAP workflow."
      ),
      page(
        "Configuration",
        "/getting-started/configuration",
        "getting-started/Configuration.md",
        "Understand paths, options, defaults, and configuration precedence."
      ),
    ]),
  }),

  Object.freeze({
    title: "Data and I/O",
    pages: Object.freeze([
      page(
        "Data Formats",
        "/data/data-formats",
        "data/Data_Formats.md",
        "Review supported external file formats and physical layouts."
      ),
      page(
        "Dataset Representations",
        "/data/dataset-representations",
        "data/Dataset_Representations.md",
        "Understand PFGAP observation shapes and in-memory representations."
      ),
      page(
        "Readers",
        "/data/readers",
        "data/Readers.md",
        "Choose and configure built-in eager and deferred readers."
      ),
      page(
        "Writers",
        "/data/writers",
        "data/Writers.md",
        "Write complete materialized and imputed datasets."
      ),
    ]),
  }),

  Object.freeze({
    title: "Guides",
    pages: Object.freeze([
      page(
        "Classification",
        "/guides/classification",
        "guides/Classification.md",
        "Train and evaluate classification forests."
      ),
      page(
        "Regression",
        "/guides/regression",
        "guides/Regression.md",
        "Train and evaluate regression forests."
      ),
      page(
        "Imputation",
        "/guides/imputation",
        "guides/Imputation.md",
        "Initialize and iteratively update missing feature values."
      ),
      page(
        "Outlier Scoring",
        "/guides/outlier-scoring",
        "guides/Outlier_Scoring.md",
        "Use supervised outlier scores and unsupervised isolation scoring."
      ),
      page(
        "OOD Scoring",
        "/guides/ood-scoring",
        "guides/OOD_Scoring.md",
        "Score evaluation observations against retained training support."
      ),
      page(
        "Standardization",
        "/guides/standardization",
        "guides/Standardization.md",
        "Configure numeric transformation, saved statistics, and inversion."
      ),
      page(
        "Eager and Lazy Data",
        "/guides/eager-and-lazy-data",
        "guides/Eager_and_Lazy_Data.md",
        "Choose between materialized and deferred data access."
      ),
      page(
        "Parallelism and Reproducibility",
        "/guides/parallelism-and-reproducibility",
        "guides/Parallelism_and_Reproducibility.md",
        "Control worker budgets, random seeds, and reproducible execution."
      ),
    ]),
  }),

  Object.freeze({
    title: "Extensions",
    pages: Object.freeze([
      page(
        "Custom Distances",
        "/extensions/custom-distances",
        "extensions/Custom_Distances.md",
        "Implement Java, Python, Maple, and meta distance extensions."
      ),
      page(
        "Custom Readers",
        "/extensions/custom-readers",
        "extensions/Custom_Readers.md",
        "Implement eager and deferred custom readers in separate JARs."
      ),
    ]),
  }),

  Object.freeze({
    title: "Reference",
    pages: Object.freeze([
      page(
        "Configuration Reference",
        "/reference/configuration-reference",
        "reference/Configuration_Reference.md",
        "Look up Python arguments, Java options, defaults, and dependencies."
      ),
      page(
        "CLI Reference",
        "/reference/cli-reference",
        "reference/CLI_Reference.md",
        "Look up direct Java command-line options and syntax."
      ),
      page(
        "Distances",
        "/reference/distances",
        "reference/Distances.md",
        "Review registered distance identifiers and compatibility guidance."
      ),
      page(
        "Missing Values",
        "/reference/missing-values",
        "reference/Missing_Values.md",
        "Review missing-value representations and handling rules."
      ),
      page(
        "Outputs",
        "/reference/outputs",
        "reference/Outputs.md",
        "Review artifact names, schemas, indexing, and output availability."
      ),
      page(
        "Imputed-Only Output",
        "/reference/imputed-only-output",
        "reference/Imputed_Only_Output.md",
        "Interpret Matrix Market output for originally missing coordinates."
      ),
      page(
        "Model Persistence",
        "/reference/model-persistence",
        "reference/Model_Persistence.md",
        "Understand saved model contents, dependencies, and compatibility."
      ),
    ]),
  }),
]);

export const orderedPages = Object.freeze(
  navigationGroups.flatMap((group) => group.pages)
);

function createUniqueMap(keyName) {
  const result = new Map();

  for (const item of orderedPages) {
    const key = item[keyName];

    if (result.has(key)) {
      throw new Error(
        `Duplicate documentation ${keyName}: ${key}`
      );
    }

    result.set(key, item);
  }

  return result;
}

export const pageByRoute = createUniqueMap("route");
export const pageBySource = createUniqueMap("source");

export function findNavigationGroup(route) {
  return navigationGroups.find((group) =>
    group.pages.some((item) => item.route === route)
  ) ?? null;
}

export function getAdjacentPages(route) {
  const index = orderedPages.findIndex(
    (item) => item.route === route
  );

  if (index < 0) {
    return {
      previous: null,
      next: null,
    };
  }

  return {
    previous: index > 0
      ? orderedPages[index - 1]
      : null,
    next: index < orderedPages.length - 1
      ? orderedPages[index + 1]
      : null,
  };
}
