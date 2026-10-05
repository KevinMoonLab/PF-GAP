package preprocessing.standardization;

import core.AppContext;
import datasets.ListObjectDataset;
import datasets.readers.lazy.LazySeriesRef;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Coordinates standardization ownership across training and evaluation.
 *
 * <p>Eager readers return raw data. GLOBAL and PER_DIMENSION fit or load
 * reusable training statistics and apply them exactly once. PER_SERIES and
 * PER_SERIES_PER_DIMENSION calculate local parameters while transforming each
 * eager realized instance. Lazy readers remain responsible for transformation
 * during materialization.</p>
 */
public final class StandardizationPipeline {

    private StandardizationPipeline() {
        // Utility class.
    }

    /**
     * Immutable forward-transformation states aligned with the training and
     * testing datasets. Lists are empty when reusable statistics are used.
     *
     * <p>Callers may ignore this result during ordinary model fitting. An
     * imputed-data writer can retain it to inverse-transform per-series data
     * into original coordinates while streaming output.</p>
     */
    public record ApplicationResult(
            List<PerSeriesStandardizationState> trainingStates,
            List<PerSeriesStandardizationState> testingStates
    ) {
        public ApplicationResult {
            trainingStates = List.copyOf(
                    Objects.requireNonNull(
                            trainingStates,
                            "trainingStates cannot be null."
                    )
            );
            testingStates = List.copyOf(
                    Objects.requireNonNull(
                            testingStates,
                            "testingStates cannot be null."
                    )
            );
        }

        public static ApplicationResult empty() {
            return new ApplicationResult(List.of(), List.of());
        }

        public boolean hasPerSeriesStates() {
            return !trainingStates.isEmpty() || !testingStates.isEmpty();
        }
    }

    /**
     * Loads reusable supplied statistics before readers are constructed.
     * Per-series scopes have no reusable dataset-level statistics.
     */
    public static void prepareSuppliedStatistics()
            throws IOException {
        StandardizationConfig config = AppContext.standardizationConfig;
        if (config == null || config.isDisabled()) {
            AppContext.standardizationStats = null;
            return;
        }

        config.requireImplemented();
        if (config.usesPerSeriesStatistics()) {
            AppContext.standardizationStats = null;
            return;
        }
        if (!config.shouldLoadStatistics()) {
            AppContext.standardizationStats = null;
            return;
        }

        List<String> featureNames = getConfiguredFeatureNames();
        StandardizationStats stats = StandardizationJson.read(
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
     * Fits reusable eager training statistics before the testing reader is
     * constructed. Per-series scopes need no dataset-level fitting pass.
     */
    public static void prepareTrainingStatisticsBeforeTestRead(
            ListObjectDataset trainingData
    ) throws IOException {
        StandardizationConfig config = AppContext.standardizationConfig;
        if (config == null || config.isDisabled()) {
            AppContext.standardizationStats = null;
            return;
        }

        config.requireImplemented();
        if (config.usesPerSeriesStatistics()) {
            AppContext.standardizationStats = null;
            return;
        }
        if (config.shouldLoadStatistics()) {
            StandardizationStats supplied = requirePreparedStatistics();
            config.validateStatistics(supplied);
            return;
        }
        if (!config.shouldFitStatistics()) {
            throw new IllegalStateException(
                    "Enabled reusable standardization has neither supplied "
                            + "nor fitted statistics."
            );
        }
        if (trainingData == null) {
            throw new IllegalArgumentException(
                    "Training data cannot be null while fitting "
                            + "standardization statistics."
            );
        }
        if (isLazyDataset(trainingData)) {
            throw new UnsupportedOperationException(
                    "Lazy training with GLOBAL or PER_DIMENSION "
                            + "standardization requires precomputed "
                            + "statistics. Automatic streaming fitting is "
                            + "deferred."
            );
        }

        StandardizationStats stats = StandardizationFitter.fit(
                trainingData,
                config.getMethod(),
                config.getScope(),
                config.getVarianceConvention(),
                getConfiguredFeatureNames()
        );
        config.validateStatistics(stats);
        AppContext.standardizationStats = stats;

        if (config.shouldSaveFittedStatistics()) {
            Path outputPath = resolveStatisticsOutputPath(config);
            StandardizationJson.write(outputPath, stats);
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
     * Applies configured standardization to eager training and testing data.
     *
     * <p>For reusable scopes the returned state lists are empty. For per-series
     * scopes they are aligned with eager dataset order and can be retained for
     * inverse output. Existing callers may invoke this method as a statement
     * and ignore the result.</p>
     */
    public static ApplicationResult applyPreparedStatistics(
            ListObjectDataset trainingData,
            ListObjectDataset testingData
    ) {
        StandardizationConfig config = AppContext.standardizationConfig;
        if (config == null || config.isDisabled()) {
            return ApplicationResult.empty();
        }

        config.requireImplemented();
        if (config.usesPerSeriesStatistics()) {
            List<PerSeriesStandardizationState> trainingStates =
                    transformPerSeriesEagerDataset(
                            trainingData,
                            config,
                            "training"
                    );
            List<PerSeriesStandardizationState> testingStates =
                    transformPerSeriesEagerDataset(
                            testingData,
                            config,
                            "testing"
                    );
            return new ApplicationResult(
                    trainingStates,
                    testingStates
            );
        }

        StandardizationStats stats = requirePreparedStatistics();
        config.validateStatistics(stats);
        List<String> featureNames = getConfiguredFeatureNames();
        transformReusableEagerDataset(
                trainingData,
                stats,
                featureNames,
                "training"
        );
        transformReusableEagerDataset(
                testingData,
                stats,
                featureNames,
                "testing"
        );
        return ApplicationResult.empty();
    }

    /**
     * Applies restored model policy to eager evaluation data and returns local
     * inverse state when a per-series scope is configured.
     */
    public static List<PerSeriesStandardizationState>
    applyEvaluationStatistics(
            ListObjectDataset testingData
    ) {
        StandardizationConfig config = AppContext.standardizationConfig;
        if (config == null || config.isDisabled()) {
            return List.of();
        }
        config.requireImplemented();
        if (testingData == null) {
            throw new IllegalArgumentException(
                    "Evaluation data cannot be null when standardization is "
                            + "enabled."
            );
        }

        if (config.usesPerSeriesStatistics()) {
            return transformPerSeriesEagerDataset(
                    testingData,
                    config,
                    "evaluation"
            );
        }

        StandardizationStats stats = requirePreparedStatistics();
        config.validateStatistics(stats);
        if (isLazyDataset(testingData)) {
            if (AppContext.verbosity > 0) {
                System.out.println(
                        "Lazy evaluation data will use saved "
                                + stats.getMethod()
                                + " standardization during materialization."
                );
            }
            return List.of();
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
                            + " standardization to eager evaluation data."
            );
        }
        return List.of();
    }

    /**
     * Returns whether all non-null instances are lazy references. Mixed lazy
     * and materialized datasets are rejected.
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
                        "A ListObjectDataset cannot mix lazy references and "
                                + "materialized instances."
                );
            }
        }
        return foundLazy;
    }

    private static List<PerSeriesStandardizationState>
    transformPerSeriesEagerDataset(
            ListObjectDataset dataset,
            StandardizationConfig config,
            String role
    ) {
        if (dataset == null) {
            return List.of();
        }
        if (isLazyDataset(dataset)) {
            throw new UnsupportedOperationException(
                    "Lazy "
                            + role
                            + " data with "
                            + config.getScope()
                            + " standardization requires reader/materializer "
                            + "integration so each realization can calculate "
                            + "and retain its local inverse state."
            );
        }

        List<PerSeriesStandardizationState> states =
                Standardizer.transformPerSeriesInPlace(
                        dataset,
                        config.getMethod(),
                        config.getScope(),
                        config.getVarianceConvention()
                );
        if (AppContext.verbosity > 0) {
            System.out.println(
                    "Applied "
                            + config.getMethod()
                            + " standardization with scope "
                            + config.getScope()
                            + " to the eager "
                            + role
                            + " dataset and retained "
                            + states.size()
                            + " local inverse state(s)."
            );
        }
        return states;
    }

    private static void transformReusableEagerDataset(
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
        StandardizationStats stats = AppContext.standardizationStats;
        if (stats == null) {
            throw new IllegalStateException(
                    "Reusable standardization is enabled but no prepared "
                            + "statistics are available."
            );
        }
        return stats;
    }

    private static Path resolveStatisticsOutputPath(
            StandardizationConfig config
    ) {
        String configuredPath = config.getStatisticsOutputPath();
        if (configuredPath != null && !configuredPath.isBlank()) {
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
        return new ArrayList<>(AppContext.feature_columns);
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
                        + java.util.Arrays.toString(stats.getCenters())
        );
        System.out.println(
                "Standardization scales: "
                        + java.util.Arrays.toString(stats.getScales())
        );
        System.out.println(
                "Standardization counts: "
                        + java.util.Arrays.toString(stats.getCounts())
        );
    }
}