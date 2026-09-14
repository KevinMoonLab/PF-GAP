package preprocessing.standardization;

import core.AppContext;
import datasets.ListObjectDataset;
import datasets.readers.lazy.LazySeriesRef;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

/**
 * Coordinates standardization across PFGAP training and evaluation workflows.
 *
 * <p>This class owns workflow decisions, not file parsing. The ownership rule
 * is:</p>
 *
 * <ul>
 *     <li>Eager readers return raw, materialized data. This pipeline applies
 *     prepared statistics exactly once after reading.</li>
 *     <li>Lazy readers retain prepared statistics in their reader
 *     specifications and apply them when a series is materialized. This
 *     pipeline never attempts to transform {@code LazySeriesRef} objects.</li>
 *     <li>Statistics are fitted only from training data. Testing and
 *     validation data always reuse the training statistics.</li>
 * </ul>
 *
 * <p>The current implementation preserves the existing Phase 1 restriction
 * that lazy training with enabled standardization requires externally
 * supplied statistics.</p>
 *
 * <p>This class is stateless. Prepared statistics remain in
 * {@link AppContext#standardizationStats} because reader construction and
 * saved-model restoration currently use that shared application state.</p>
 */
public final class StandardizationPipeline {

    private StandardizationPipeline() {
        // Utility class.
    }

    /**
     * Prepares standardization state before any training or testing reader is
     * constructed.
     *
     * <p>Externally supplied statistics must be loaded here so lazy reader
     * specifications can capture them. When automatic eager fitting is
     * configured, stale statistics are cleared before the raw training data
     * is read.</p>
     *
     * @throws IOException if configured statistics cannot be read
     */
    public static void prepareSuppliedStatistics()
            throws IOException {

        StandardizationConfig config =
                AppContext.standardizationConfig;

        if (config == null || config.isDisabled()) {
            AppContext.standardizationStats = null;
            return;
        }

        config.requireImplemented();

        if (!config.shouldLoadStatistics()) {
            AppContext.standardizationStats = null;
            return;
        }

        List<String> featureNames =
                getConfiguredFeatureNames();

        StandardizationStats stats =
                StandardizationJson.read(
                        config.getStatisticsPath(),
                        config,
                        featureNames
                );

        config.validateStatistics(stats);
        AppContext.standardizationStats = stats;

        if (AppContext.verbosity > 0) {
            System.out.println(
                    "Loaded standardization statistics from: "
                            + config.getStatisticsPath()
            );
            printSummary(stats);
        }
    }

    /**
     * Ensures that training statistics are available before a testing reader
     * is constructed.
     *
     * <p>For automatic eager fitting, this method fits the raw training data
     * exactly once. Running it before the testing reader is constructed makes
     * the fitted statistics available to a lazy testing reader. Eager testing
     * readers must still return raw data and are transformed later by
     * {@link #applyPreparedStatistics(ListObjectDataset, ListObjectDataset)}.</p>
     *
     * @param trainingData raw eager training data or a lazy training dataset
     * @throws IOException if fitted statistics cannot be written
     */
    public static void prepareTrainingStatisticsBeforeTestRead(
            ListObjectDataset trainingData
    ) throws IOException {

        StandardizationConfig config =
                AppContext.standardizationConfig;

        if (config == null || config.isDisabled()) {
            AppContext.standardizationStats = null;
            return;
        }

        config.requireImplemented();

        if (config.shouldLoadStatistics()) {
            StandardizationStats supplied =
                    requirePreparedStatistics();

            config.validateStatistics(supplied);
            return;
        }

        if (trainingData == null) {
            throw new IllegalArgumentException(
                    "Training data cannot be null while fitting "
                            + "standardization statistics."
            );
        }

        if (isLazyDataset(trainingData)) {
            throw new UnsupportedOperationException(
                    "Lazy training with standardization currently requires "
                            + "precomputed statistics supplied through "
                            + "-standardization_stats. Automatic streaming "
                            + "statistics fitting is not yet implemented."
            );
        }

        List<String> featureNames =
                getConfiguredFeatureNames();

        StandardizationStats stats =
                StandardizationFitter.fit(
                        trainingData,
                        config.getMethod(),
                        config.getScope(),
                        config.getVarianceConvention(),
                        featureNames
                );

        config.validateStatistics(stats);
        AppContext.standardizationStats = stats;

        if (config.shouldSaveFittedStatistics()) {
            Path outputPath =
                    resolveStatisticsOutputPath(config);

            StandardizationJson.write(
                    outputPath,
                    stats
            );

            if (AppContext.verbosity > 0) {
                System.out.println(
                        "Saved fitted standardization statistics to: "
                                + outputPath
                );
            }
        }

        if (AppContext.verbosity > 0) {
            System.out.println(
                    "Fitted standardization statistics from the eager "
                            + "training dataset."
            );
            printSummary(stats);
        }
    }

    /**
     * Applies prepared statistics to eager training and testing datasets.
     * Lazy datasets are skipped because their materializers own the
     * transformation.
     *
     * @param trainingData training dataset, possibly lazy
     * @param testingData testing or validation dataset, possibly null or lazy
     */
    public static void applyPreparedStatistics(
            ListObjectDataset trainingData,
            ListObjectDataset testingData
    ) {

        StandardizationConfig config =
                AppContext.standardizationConfig;

        if (config == null || config.isDisabled()) {
            return;
        }

        config.requireImplemented();

        StandardizationStats stats =
                requirePreparedStatistics();

        config.validateStatistics(stats);

        List<String> featureNames =
                getConfiguredFeatureNames();

        transformEagerDataset(
                trainingData,
                stats,
                featureNames,
                "training"
        );

        transformEagerDataset(
                testingData,
                stats,
                featureNames,
                "testing"
        );
    }

    /**
     * Applies restored model statistics to an eager evaluation dataset.
     * Lazy evaluation datasets are left for their registered materializers.
     *
     * @param testingData evaluation dataset
     */
    public static void applyEvaluationStatistics(
            ListObjectDataset testingData
    ) {

        StandardizationConfig config =
                AppContext.standardizationConfig;

        if (config == null || config.isDisabled()) {
            return;
        }

        config.requireImplemented();

        StandardizationStats stats =
                requirePreparedStatistics();

        config.validateStatistics(stats);

        if (testingData == null) {
            throw new IllegalArgumentException(
                    "Evaluation data cannot be null when standardization "
                            + "is enabled."
            );
        }

        if (isLazyDataset(testingData)) {
            if (AppContext.verbosity > 0) {
                System.out.println(
                        "Lazy testing data will use saved "
                                + stats.getMethod()
                                + " standardization during materialization."
                );
            }
            return;
        }

        Standardizer.transformInPlace(
                testingData,
                stats,
                getConfiguredFeatureNames()
        );

        if (AppContext.verbosity > 0) {
            System.out.println(
                    "Applied saved "
                            + stats.getMethod()
                            + " standardization to the eager testing dataset."
            );
        }
    }

    /**
     * Returns whether a dataset is represented by lazy series references.
     *
     * <p>A dataset mixing lazy references and materialized instances is
     * rejected because no standardization ownership rule can safely handle
     * such a mixture.</p>
     *
     * @param dataset dataset to inspect, possibly null
     * @return true if the non-null instances are lazy references
     */
    public static boolean isLazyDataset(
            ListObjectDataset dataset
    ) {
        if (dataset == null) {
            return false;
        }

        boolean foundLazy = false;
        boolean foundMaterialized = false;

        for (Object value : dataset.getData()) {
            if (value == null) {
                continue;
            }

            if (value instanceof LazySeriesRef) {
                foundLazy = true;
            } else {
                foundMaterialized = true;
            }

            if (foundLazy && foundMaterialized) {
                throw new IllegalStateException(
                        "A ListObjectDataset cannot mix lazy references "
                                + "and materialized instances."
                );
            }
        }

        return foundLazy;
    }

    private static void transformEagerDataset(
            ListObjectDataset dataset,
            StandardizationStats stats,
            List<String> featureNames,
            String role
    ) {
        if (dataset == null || isLazyDataset(dataset)) {
            return;
        }

        Standardizer.transformInPlace(
                dataset,
                stats,
                featureNames
        );

        if (AppContext.verbosity > 0) {
            System.out.println(
                    "Applied "
                            + stats.getMethod()
                            + " standardization to the eager "
                            + role
                            + " dataset."
            );
        }
    }

    private static StandardizationStats requirePreparedStatistics() {
        StandardizationStats stats =
                AppContext.standardizationStats;

        if (stats == null) {
            throw new IllegalStateException(
                    "Standardization is enabled but no prepared statistics "
                            + "are available."
            );
        }

        return stats;
    }

    private static Path resolveStatisticsOutputPath(
            StandardizationConfig config
    ) {
        String configuredPath =
                config.getStatisticsOutputPath();

        if (configuredPath != null
                && !configuredPath.isBlank()) {

            return Paths.get(configuredPath);
        }

        return Paths.get(
                AppContext.output_dir,
                "standardization_stats.json"
        );
    }

    private static List<String> getConfiguredFeatureNames() {
        if (AppContext.feature_columns == null
                || AppContext.feature_columns.isEmpty()) {

            return List.of();
        }

        return new ArrayList<>(
                AppContext.feature_columns
        );
    }

    private static void printSummary(
            StandardizationStats stats
    ) {
        System.out.println(
                "Prepared "
                        + stats.getMethod()
                        + " standardization with scope "
                        + stats.getScope()
                        + " using "
                        + stats.getStatisticGroupCount()
                        + " fitted statistic group(s)."
        );

        if (AppContext.verbosity <= 1) {
            return;
        }

        System.out.println(
                "Standardization centers: "
                        + java.util.Arrays.toString(
                        stats.getCenters()
                )
        );

        System.out.println(
                "Standardization scales: "
                        + java.util.Arrays.toString(
                        stats.getScales()
                )
        );

        System.out.println(
                "Standardization counts: "
                        + java.util.Arrays.toString(
                        stats.getCounts()
                )
        );
    }
}