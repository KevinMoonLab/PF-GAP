package core.experiment;

import core.AppContext;
import core.ProximityForestResult;
import datasets.ListObjectDataset;
import output.ExperimentResultRecord;

import java.util.Map;
import java.util.Objects;

/**
 * Converts the completed state of one experiment repetition into an immutable
 * {@link ExperimentResultRecord}.
 *
 * <p>This class owns result-record assembly only. It does not train forests,
 * evaluate data, compute proximity, write artifacts, or mutate datasets.</p>
 *
 * <p>Runtime configuration is read when {@link #assemble} is called so the
 * resulting record describes the configuration that produced that repetition.
 * Repetition-specific metrics, counts, timings, and artifacts are obtained from
 * the supplied {@link ExperimentRepetitionContext}.</p>
 */
public final class ExperimentResultAssembler {

    /**
     * Builds one experiment result record.
     *
     * @param result completed forest result
     * @param datasetName logical dataset name
     * @param trainingData prepared training data, or null when unavailable
     * @param testingData prepared testing data, or null when unavailable
     * @param context repetition-specific execution and artifact state
     * @return immutable result record
     */
    public ExperimentResultRecord assemble(
            ProximityForestResult result,
            String datasetName,
            ListObjectDataset trainingData,
            ListObjectDataset testingData,
            ExperimentRepetitionContext context
    ) {
        Objects.requireNonNull(
                result,
                "ProximityForestResult cannot be null."
        );
        Objects.requireNonNull(
                context,
                "ExperimentRepetitionContext cannot be null."
        );

        String normalizedDatasetName =
                normalizeDatasetName(datasetName);

        /*
         * Preserve the existing result contract: forest-level aggregate
         * statistics are finalized immediately before the record is built.
         */
        result.collateResults();

        ExperimentResultRecord.Builder builder =
                ExperimentResultRecord.builder()
                        .setDataset(normalizedDatasetName)
                        .setRepetition(context.getDisplayRepetition())
                        .setForestId(result.forest_id)
                        .setForestMode(AppContext.forest_mode)
                        .addCount(
                                "trainingInstanceCount",
                                sizeOf(trainingData)
                        )
                        .addCount(
                                "testingInstanceCount",
                                sizeOf(testingData)
                        )
                        .addTimingNanoseconds(
                                "trainingMilliseconds",
                                Math.max(
                                        0L,
                                        result.elapsedTimeTrain
                                )
                        )
                        .addTimingNanoseconds(
                                "testingMilliseconds",
                                Math.max(
                                        0L,
                                        result.elapsedTimeTest
                                )
                        )
                        .addForestStatistic(
                                "numTrees",
                                result.total_num_trees
                        )
                        .addConfiguration(
                                "numCandidatesPerSplit",
                                AppContext.num_candidates_per_split
                        )
                        .addConfiguration(
                                "bootstrapTrees",
                                AppContext.bootstrap_trees
                        )
                        .addConfiguration(
                                "numWorkersRequested",
                                AppContext.num_workers
                        )
                        .addConfiguration(
                                "numWorkersEffective",
                                context.getWorkerCount()
                        )
                        .addForestStatistic(
                                "meanNodesPerTree",
                                result.mean_num_nodes_per_tree
                        )
                        .addForestStatistic(
                                "standardDeviationNodesPerTree",
                                result.sd_num_nodes_per_tree
                        )
                        .addForestStatistic(
                                "meanDepthPerTree",
                                result.mean_depth_per_tree
                        )
                        .addForestStatistic(
                                "standardDeviationDepthPerTree",
                                result.sd_depth_per_tree
                        )
                        .addForestStatistic(
                                "meanWeightedDepthPerTree",
                                result.mean_weighted_depth_per_tree
                        )
                        .addForestStatistic(
                                "standardDeviationWeightedDepthPerTree",
                                result.sd_weighted_depth_per_tree
                        )
                        .addConfiguration(
                                "proximityType",
                                AppContext.proximityType
                        )
                        .addConfiguration(
                                "trainingReaderType",
                                AppContext.getTrainingReaderType()
                        )
                        .addConfiguration(
                                "testingReaderType",
                                testingData == null
                                        ? null
                                        : AppContext.getTestingReaderType()
                        )
                        .addConfiguration(
                                "standardizationMethod",
                                AppContext.standardizationConfig == null
                                        ? null
                                        : AppContext.standardizationConfig
                                                .getMethod()
                        )
                        .addConfiguration(
                                "standardizationScope",
                                AppContext.standardizationConfig == null
                                        ? null
                                        : AppContext.standardizationConfig
                                                .getScope()
                        )
                        .addMetrics(context.getAdditionalMetrics())
                        .addCounts(context.getAdditionalCounts())
                        .addArtifacts(context.getArtifacts());

        addAdditionalTimings(
                builder,
                context.getAdditionalTimings()
        );

        addModeSpecificConfiguration(builder);
        addLearningMetrics(builder, result);

        return builder.build();
    }

    private static void addAdditionalTimings(
            ExperimentResultRecord.Builder builder,
            Map<String, Double> timings
    ) {
        if (timings == null || timings.isEmpty()) {
            return;
        }

        for (Map.Entry<String, Double> entry : timings.entrySet()) {
            builder.addTimingMilliseconds(
                    entry.getKey(),
                    entry.getValue()
            );
        }
    }

    private static void addModeSpecificConfiguration(
            ExperimentResultRecord.Builder builder
    ) {
        if (!AppContext.isIsolationMode()) {
            return;
        }

        builder.addConfiguration(
                "isolationNumBranches",
                AppContext.isolation_num_branches
        );

        builder.addConfiguration(
                "isolationMinLeafSize",
                AppContext.isolation_min_leaf_size
        );

        builder.addConfiguration(
                "isolationScoreMethod",
                "pathLength"
        );

        builder.addConfiguration(
                "purityMeasure",
                AppContext.purity_measure
        );
    }

    private static void addLearningMetrics(
            ExperimentResultRecord.Builder builder,
            ProximityForestResult result
    ) {
        if (AppContext.isClassificationMode()) {
            addClassificationMetrics(
                    builder,
                    result
            );
            return;
        }

        if (AppContext.isRegressionMode()) {
            addRegressionMetrics(
                    builder,
                    result
            );
        }
    }

    private static void addClassificationMetrics(
            ExperimentResultRecord.Builder builder,
            ProximityForestResult result
    ) {
        long correct =
                result.correct;

        long errors =
                result.errors;

        long total =
                correct + errors;

        if (total > 0L) {
            double accuracy =
                    (double) correct / total;

            double errorRate =
                    (double) errors / total;

            builder.addMetric(
                    "accuracy",
                    accuracy
            );

            builder.addMetric(
                    "errorRate",
                    errorRate
            );
        }

        builder.addCount(
                "correct",
                correct
        );

        builder.addCount(
                "errors",
                errors
        );
    }

    private static void addRegressionMetrics(
            ExperimentResultRecord.Builder builder,
            ProximityForestResult result
    ) {
        /*
         * Preserve the existing generic name until ProximityForestResult
         * explicitly identifies the regression statistic represented by score.
         */
        if (Double.isFinite(result.score)) {
            builder.addMetric(
                    "regressionScore",
                    result.score
            );
        }
    }

    private static long sizeOf(ListObjectDataset dataset) {
        return dataset == null
                ? 0L
                : dataset.size();
    }

    private static String normalizeDatasetName(String datasetName) {
        if (datasetName == null || datasetName.isBlank()) {
            return "dataset";
        }
        return datasetName.trim();
    }
}
