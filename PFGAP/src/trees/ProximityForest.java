package trees;

import core.AppContext;
import core.ForestPredictionResult;
import core.ProximityForestResult;
import core.parallel.ParallelRuntime;
import datasets.ListObjectDataset;
import datasets.readers.lazy.LazySeriesRef;
import distance.MEASURE;
import ood.OODScoreResult;
import ood.OODScoreType;
import ood.PathOODScorer;
import ood.RelativeSupportExceedanceOODScorer;
import util.PrintUtilities;
import util.LabelUtils;

import java.io.Serial;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Random;
import java.util.concurrent.locks.ReentrantLock;

/**
 * Proximity forest supporting classification, regression, and isolation.
 *
 * <p>All PFGAP-owned parallel work is submitted to an explicitly supplied
 * {@link ParallelRuntime}. This class does not create executors and does not
 * use the common fork-join pool.</p>
 *
 * <p>Training currently exposes whole trees as parallel work. Evaluation
 * exposes test instances when several queries are available and tree ranges
 * when only one query is available. Later training-tail and prediction-grid
 * refinements can expose finer work through the same runtime.</p>
 *
 * <p>Lazy prediction queries are materialized once at the forest boundary and
 * reused throughout every tree traversal. Node exemplars remain controlled by
 * their splitter.</p>
 */
public class ProximityForest implements Serializable {

    @Serial
    private static final long serialVersionUID =
            -1183368028217094381L;

    private static final int MINIMUM_TREE_RANGE_SIZE = 1;
    private static final int MINIMUM_PREDICTION_RANGE_SIZE = 1;
    /** Maximum training observations held in the temporary OOB grid. */
    private static final int OOB_AGGREGATION_BATCH_SIZE = 4096;

    protected ProximityForestResult result;
    protected int forest_id;
    protected ProximityTree[] trees;
    public String prefix;

    /** Retained for compatibility with older callers. */
    List<Object> predictions;

    private final ReentrantLock trainLock =
            new ReentrantLock();

    private final long forestSeed;

    public ProximityForest(
            int forestId,
            MEASURE... selectedDistances
    ) {
        this.result =
                new ProximityForestResult(this);
        this.forest_id =
                forestId;
        this.forestSeed =
                AppContext.getRand().nextLong();
        this.trees =
                new ProximityTree[AppContext.num_trees];

        MEASURE[] distances =
                selectedDistances == null
                        ? new MEASURE[0]
                        : selectedDistances.clone();

        for (int treeIndex = 0;
             treeIndex < trees.length;
             treeIndex++) {
            trees[treeIndex] =
                    new ProximityTree(
                            treeIndex,
                            this,
                            distances
                    );
        }
    }

    /**
     * Compatibility entry point that owns a runtime for this call.
     * Repetition workflows should prefer {@link #train(ListObjectDataset,
     * ParallelRuntime)} so training and evaluation share one repetition-owned
     * runtime.
     */
    public void train(
            ListObjectDataset trainData
    ) throws Exception {
        try (ParallelRuntime runtime =
                     new ParallelRuntime(AppContext.num_workers)) {
            train(trainData, runtime);
        }
    }

    /**
     * Trains every tree using the supplied bounded runtime.
     */
    public void train(
            ListObjectDataset trainData,
            ParallelRuntime runtime
    ) throws Exception {
        Objects.requireNonNull(
                trainData,
                "Training data cannot be null."
        );
        Objects.requireNonNull(
                runtime,
                "ParallelRuntime cannot be null."
        );

        if (trainData.size() == 0) {
            throw new IllegalArgumentException(
                    "Training data cannot be empty."
            );
        }

        if (!AppContext.isIsolationMode()) {
            if (AppContext.isRegressionMode()) {
                LabelUtils.validateRegressionLabels(
                        trainData._internal_class_list()
                );
            } else {
                LabelUtils.validateClassificationLabels(
                        trainData._internal_class_list()
                );
            }
        }

        trainLock.lockInterruptibly();
        try {
            result.startTimeTrain =
                    System.nanoTime();

            runtime.forRange(
                    0,
                    trees.length,
                    MINIMUM_TREE_RANGE_SIZE,
                    treeIndex -> {
                        trees[treeIndex].train(trainData, runtime);
                        reportTreeProgress(treeIndex);
                    }
            );

            if (AppContext.return_oob_predictions) {
                computeOutOfBagPredictions(trainData, runtime);
            } else {
                result.OobPredictions.clear();
                result.OobPredictionTreeCounts = new int[0];
            }

            result.endTimeTrain =
                    System.nanoTime();
            result.elapsedTimeTrain =
                    result.endTimeTrain - result.startTimeTrain;

            if (AppContext.verbosity > 0) {
                System.out.println();
                PrintUtilities.printMemoryUsage();
            }
        } finally {
            trainLock.unlock();
        }
    }

    /**
     * Aggregates predictions from OOB leaf memberships already recorded while
     * fitting. This performs no second tree traversal.
     *
     * <p>Only one bounded training-position batch is materialized at a time.
     * Temporary reference storage is therefore O(treeCount * batchSize), not
     * O(treeCount * trainingSize). Batches are processed in source order and
     * tree predictions are combined in tree-index order, preserving seeded
     * determinism.</p>
     */
    private void computeOutOfBagPredictions(
            ListObjectDataset trainData,
            ParallelRuntime runtime
    ) throws Exception {
        if (!AppContext.oob_scores) {
            throw new IllegalStateException(
                    "OOB prediction output requires fit-time OOB routing."
            );
        }
        if (AppContext.isIsolationMode()) {
            throw new IllegalArgumentException(
                    "OOB predictions are not currently supported in isolation mode."
            );
        }

        int trainingSize = trainData.size();
        Object[] aggregatePredictions = new Object[trainingSize];
        int[] contributingTreeCounts = new int[trainingSize];

        for (int batchStart = 0;
             batchStart < trainingSize;
             batchStart += OOB_AGGREGATION_BATCH_SIZE) {
            int batchEnd = Math.min(
                    trainingSize,
                    batchStart + OOB_AGGREGATION_BATCH_SIZE
            );
            aggregateOutOfBagBatch(
                    batchStart,
                    batchEnd,
                    aggregatePredictions,
                    contributingTreeCounts,
                    runtime
            );
        }

        result.OobPredictions = new ArrayList<>(
                Arrays.asList(aggregatePredictions)
        );
        result.OobPredictionTreeCounts = contributingTreeCounts;
    }

    private void aggregateOutOfBagBatch(
            int batchStart,
            int batchEnd,
            Object[] aggregatePredictions,
            int[] contributingTreeCounts,
            ParallelRuntime runtime
    ) throws Exception {
        int batchSize = batchEnd - batchStart;
        Object[][] predictionsByTree =
                new Object[trees.length][batchSize];

        runtime.forRange(
                0,
                trees.length,
                MINIMUM_TREE_RANGE_SIZE,
                treeIndex -> trees[treeIndex]
                        .populateRoutedOutOfBagPredictions(
                                batchStart,
                                batchEnd,
                                predictionsByTree[treeIndex]
                        )
        );

        runtime.forRange(
                0,
                batchSize,
                MINIMUM_PREDICTION_RANGE_SIZE,
                batchOffset -> aggregateOneOutOfBagPrediction(
                        predictionsByTree,
                        batchOffset,
                        batchStart + batchOffset,
                        aggregatePredictions,
                        contributingTreeCounts
                )
        );
    }

    private void aggregateOneOutOfBagPrediction(
            Object[][] predictionsByTree,
            int batchOffset,
            int trainingPosition,
            Object[] aggregatePredictions,
            int[] contributingTreeCounts
    ) {
        Object[] treePredictions = new Object[trees.length];
        int contributingTrees = 0;
        for (int treeIndex = 0;
             treeIndex < trees.length;
             treeIndex++) {
            Object prediction = predictionsByTree[treeIndex][batchOffset];
            if (prediction != null) {
                treePredictions[treeIndex] = prediction;
                contributingTrees++;
            }
        }
        contributingTreeCounts[trainingPosition] = contributingTrees;
        aggregatePredictions[trainingPosition] = contributingTrees == 0
                ? null
                : combineTreePredictions(treePredictions);
    }

    private void reportTreeProgress(int treeIndex) {
        if (AppContext.verbosity <= 0) {
            return;
        }

        synchronized (System.out) {
            System.out.print(treeIndex + ".");

            if (AppContext.verbosity > 1) {
                PrintUtilities.printMemoryUsage(true);
                if ((treeIndex + 1) % 20 == 0) {
                    System.out.println();
                }
            }
        }
    }

    /**
     * Compatibility entry point that owns a runtime for this call.
     */
    public ProximityForestResult test(
            ListObjectDataset testData
    ) throws Exception {
        try (ParallelRuntime runtime =
                     new ParallelRuntime(AppContext.num_workers)) {
            return test(testData, runtime);
        }
    }

    /**
     * Evaluates the forest using the supplied bounded runtime.
     *
     * <p>For multiple test instances, queries are evaluated in parallel and
     * each resolved query traverses all trees sequentially. For exactly one
     * test instance, the query is resolved once and tree traversal is divided
     * across the runtime. This preserves query materialization reuse while
     * avoiding an idle pool for single-instance evaluation.</p>
     */
    public ProximityForestResult test(
            ListObjectDataset testData,
            ParallelRuntime runtime
    ) throws Exception {
        Objects.requireNonNull(
                testData,
                "Test data cannot be null."
        );
        Objects.requireNonNull(
                runtime,
                "ParallelRuntime cannot be null."
        );

        result.startTimeTest =
                System.nanoTime();

        int testSize =
                testData.size();
        Object[] actualLabels =
                new Object[testSize];
        Object[] predictedLabels =
                new Object[testSize];
        ProximityTree.Node[][] reachedLeaves =
                new ProximityTree.Node[testSize][trees.length];

        if (testSize > 1 && runtime.isParallel()) {
            runtime.forRange(
                    0,
                    testSize,
                    MINIMUM_PREDICTION_RANGE_SIZE,
                    testIndex -> {
                        evaluateOneInstanceSequentialTrees(
                                testData,
                                testIndex,
                                actualLabels,
                                predictedLabels,
                                reachedLeaves
                        );
                        reportTestProgress(testIndex);
                    }
            );
        } else {
            for (int testIndex = 0;
                 testIndex < testSize;
                 testIndex++) {
                evaluateOneInstance(
                        testData,
                        testIndex,
                        actualLabels,
                        predictedLabels,
                        reachedLeaves,
                        runtime
                );
                reportTestProgress(testIndex);
            }
        }

        /*
         * TestIndices uses ArrayList, so merge memberships only after all
         * parallel traversal has completed.
         */
        recordReachedLeaves(reachedLeaves);

        result.clearPredictionResults();
        result.setPredictions(Arrays.asList(predictedLabels));
        predictions = result.Predictions;

        calculateEvaluationMetrics(
                actualLabels,
                predictedLabels
        );

        result.endTimeTest =
                System.nanoTime();
        result.elapsedTimeTest =
                result.endTimeTest - result.startTimeTest;

        if (AppContext.verbosity > 0) {
            System.out.println();
        }

        return result;
    }

    private void evaluateOneInstanceSequentialTrees(
            ListObjectDataset testData,
            int testIndex,
            Object[] actualLabels,
            Object[] predictedLabels,
            ProximityTree.Node[][] reachedLeaves
    ) throws Exception {
        actualLabels[testIndex] =
                testData.get_class(testIndex);

        Object resolvedQuery =
                resolvePredictionQuery(
                        testData.get_series(testIndex)
                );

        Object[] treePredictions =
                new Object[trees.length];

        evaluateTreeRange(
                resolvedQuery,
                testIndex,
                0,
                trees.length,
                treePredictions,
                reachedLeaves[testIndex]
        );

        predictedLabels[testIndex] =
                combineTreePredictions(treePredictions);
    }

    private void evaluateOneInstance(
            ListObjectDataset testData,
            int testIndex,
            Object[] actualLabels,
            Object[] predictedLabels,
            ProximityTree.Node[][] reachedLeaves,
            ParallelRuntime runtime
    ) throws Exception {
        actualLabels[testIndex] =
                testData.get_class(testIndex);

        Object resolvedQuery =
                resolvePredictionQuery(
                        testData.get_series(testIndex)
                );

        Object[] treePredictions =
                new Object[trees.length];

        runtime.forRange(
                0,
                trees.length,
                MINIMUM_TREE_RANGE_SIZE,
                treeIndex -> evaluateTree(
                        resolvedQuery,
                        testIndex,
                        treeIndex,
                        treePredictions,
                        reachedLeaves[testIndex]
                )
        );

        predictedLabels[testIndex] =
                combineTreePredictions(treePredictions);
    }

    private Object resolvePredictionQuery(Object query) {
        if (query == null) {
            throw new IllegalArgumentException(
                    "Prediction query cannot be null."
            );
        }

        if (query instanceof LazySeriesRef reference) {
            return AppContext.readLazySeries(reference);
        }

        return query;
    }

    private void evaluateTreeRange(
            Object resolvedQuery,
            int testIndex,
            int treeStart,
            int treeEnd,
            Object[] treePredictions,
            ProximityTree.Node[] reachedLeaves
    ) throws Exception {
        for (int treeIndex = treeStart;
             treeIndex < treeEnd;
             treeIndex++) {
            evaluateTree(
                    resolvedQuery,
                    testIndex,
                    treeIndex,
                    treePredictions,
                    reachedLeaves
            );
        }
    }

    private void evaluateTree(
            Object resolvedQuery,
            int testIndex,
            int treeIndex,
            Object[] treePredictions,
            ProximityTree.Node[] reachedLeaves
    ) throws Exception {
        ProximityTree.Node leaf =
                trees[treeIndex].findLeafResolved(
                        resolvedQuery,
                        predictionRandom(testIndex, treeIndex)
                );

        reachedLeaves[treeIndex] =
                leaf;
        treePredictions[treeIndex] =
                leaf.label();
    }

    /**
     * Compatibility method for predicting one instance. The query is resolved
     * once and trees are traversed sequentially.
     */
    public Object predict(
            Object query,
            int index
    ) throws Exception {
        Object resolvedQuery =
                resolvePredictionQuery(query);
        Object[] treePredictions =
                new Object[trees.length];
        ProximityTree.Node[] reachedLeaves =
                new ProximityTree.Node[trees.length];

        evaluateTreeRange(
                resolvedQuery,
                index,
                0,
                trees.length,
                treePredictions,
                reachedLeaves
        );

        recordReachedLeaves(
                new ProximityTree.Node[][]{reachedLeaves},
                index
        );

        return combineTreePredictions(treePredictions);
    }

    /**
     * Calculates OOD scores for a dataset using the selected scoring method.
     *
     * <p>This operation is independent of ordinary prediction output and does
     * not modify leaf TestIndices. Queries are materialized once at the forest
     * boundary and reused across every tree traversal.</p>
     */
    public ForestOODScore[] scoreOOD(
            ListObjectDataset data,
            OODScoreType scoreType
    ) throws Exception {
        try (ParallelRuntime runtime =
                     new ParallelRuntime(AppContext.num_workers)) {
            return scoreOOD(data, scoreType, runtime);
        }
    }

    /**
     * Calculates OOD scores for every supplied query using the supplied bounded
     * runtime.
     */
    public ForestOODScore[] scoreOOD(
            ListObjectDataset data,
            OODScoreType scoreType,
            ParallelRuntime runtime
    ) throws Exception {
        Objects.requireNonNull(data, "OOD evaluation data cannot be null.");
        Objects.requireNonNull(scoreType, "OOD score type cannot be null.");
        Objects.requireNonNull(runtime, "ParallelRuntime cannot be null.");
        requireSplitDistanceOODSupport();

        ForestOODScore[] scores = new ForestOODScore[data.size()];
        if (data.size() > 1 && runtime.isParallel()) {
            runtime.forRange(
                    0,
                    data.size(),
                    MINIMUM_PREDICTION_RANGE_SIZE,
                    testIndex -> scores[testIndex] =
                            scoreOneOODSequentialTrees(
                                    resolvePredictionQuery(
                                            data.get_series(testIndex)
                                    ),
                                    testIndex,
                                    scoreType
                            )
            );
            return scores;
        }

        for (int testIndex = 0; testIndex < data.size(); testIndex++) {
            scores[testIndex] = scoreOneOOD(
                    resolvePredictionQuery(data.get_series(testIndex)),
                    testIndex,
                    scoreType,
                    runtime
            );
        }
        return scores;
    }

    /** Calculates one forest OOD score. */
    public ForestOODScore scoreOOD(
            Object query,
            int index,
            OODScoreType scoreType
    ) throws Exception {
        requireSplitDistanceOODSupport();
        return scoreOneOODSequentialTrees(
                resolvePredictionQuery(query),
                index,
                Objects.requireNonNull(
                        scoreType,
                        "OOD score type cannot be null."
                )
        );
    }

    private ForestOODScore scoreOneOODSequentialTrees(
            Object resolvedQuery,
            int testIndex,
            OODScoreType scoreType
    ) throws Exception {
        OODScoreResult[] treeScores = new OODScoreResult[trees.length];
        for (int treeIndex = 0; treeIndex < trees.length; treeIndex++) {
            treeScores[treeIndex] = scoreTreeOOD(
                    resolvedQuery,
                    testIndex,
                    treeIndex,
                    scoreType
            );
        }
        return aggregateTreeOODScores(scoreType, treeScores);
    }

    private ForestOODScore scoreOneOOD(
            Object resolvedQuery,
            int testIndex,
            OODScoreType scoreType,
            ParallelRuntime runtime
    ) throws Exception {
        OODScoreResult[] treeScores = new OODScoreResult[trees.length];
        runtime.forRange(
                0,
                trees.length,
                MINIMUM_TREE_RANGE_SIZE,
                treeIndex -> treeScores[treeIndex] = scoreTreeOOD(
                        resolvedQuery,
                        testIndex,
                        treeIndex,
                        scoreType
                )
        );
        return aggregateTreeOODScores(scoreType, treeScores);
    }

    private OODScoreResult scoreTreeOOD(
            Object resolvedQuery,
            int testIndex,
            int treeIndex,
            OODScoreType scoreType
    ) throws Exception {
        PathOODScorer scorer = createOODScorer(scoreType);
        trees[treeIndex].findLeafResolved(
                resolvedQuery,
                predictionRandom(testIndex, treeIndex),
                scorer
        );
        return scorer.finish();
    }

    private static PathOODScorer createOODScorer(
            OODScoreType scoreType
    ) {
        return switch (Objects.requireNonNull(
                scoreType,
                "OOD score type cannot be null."
        )) {
            case RELATIVE_SUPPORT_EXCEEDANCE ->
                    new RelativeSupportExceedanceOODScorer();
        };
    }

    private static ForestOODScore aggregateTreeOODScores(
            OODScoreType scoreType,
            OODScoreResult[] treeScores
    ) {
        RunningMoments moments = new RunningMoments();
        for (OODScoreResult treeScore : treeScores) {
            if (treeScore != null && treeScore.isAvailable()) {
                if (treeScore.scoreType() != scoreType) {
                    throw new IllegalStateException(
                            "Tree OOD score type does not match forest request."
                    );
                }
                moments.add(treeScore.score());
            }
        }
        return moments.finish(scoreType, treeScores.length);
    }

    /**
     * Immutable forest-level OOD result for one query.
     *
     * <p>The standard deviation is the population standard deviation across
     * available tree scores. Mean and standard deviation are NaN when no tree
     * produced an available score.</p>
     */
    public record ForestOODScore(
            OODScoreType scoreType,
            double mean,
            double standardDeviation,
            int availableTreeCount,
            int totalTreeCount
    ) implements Serializable {

        @Serial
        private static final long serialVersionUID = 1L;

        public ForestOODScore {
            Objects.requireNonNull(scoreType, "OOD score type cannot be null.");
            if (availableTreeCount < 0 || totalTreeCount < 0
                    || availableTreeCount > totalTreeCount) {
                throw new IllegalArgumentException(
                        "Invalid available and total OOD tree counts."
                );
            }
            if (availableTreeCount == 0) {
                if (!Double.isNaN(mean) || !Double.isNaN(standardDeviation)) {
                    throw new IllegalArgumentException(
                            "Unavailable forest OOD statistics must be NaN."
                    );
                }
            } else if (!Double.isFinite(mean)
                    || !Double.isFinite(standardDeviation)
                    || mean < 0.0
                    || standardDeviation < 0.0) {
                throw new IllegalArgumentException(
                        "Available forest OOD statistics must be finite and nonnegative."
                );
            }
        }

        public boolean isAvailable() {
            return availableTreeCount > 0;
        }
    }

    private static final class RunningMoments {
        private int count;
        private double mean;
        private double m2;

        private void add(double value) {
            if (!Double.isFinite(value) || value < 0.0) {
                throw new IllegalArgumentException(
                        "Tree OOD score must be finite and nonnegative."
                );
            }
            addSigned(value);
        }

        private void addSigned(double value) {
            if (!Double.isFinite(value)) {
                throw new IllegalArgumentException(
                        "Moment value must be finite."
                );
            }
            count++;
            double delta = value - mean;
            mean += delta / count;
            double delta2 = value - mean;
            m2 += delta * delta2;
        }

        private int count() {
            return count;
        }

        private double mean() {
            return canonicalizeZero(mean);
        }

        private double standardDeviation() {
            if (count == 0) {
                return Double.NaN;
            }
            return canonicalizeZero(Math.sqrt(Math.max(0.0, m2 / count)));
        }

        private ForestOODScore finish(
                OODScoreType scoreType,
                int totalTreeCount
        ) {
            if (count == 0) {
                return new ForestOODScore(
                        scoreType,
                        Double.NaN,
                        Double.NaN,
                        0,
                        totalTreeCount
                );
            }
            double variance = Math.max(0.0, m2 / count);
            return new ForestOODScore(
                    scoreType,
                    canonicalizeZero(mean),
                    canonicalizeZero(Math.sqrt(variance)),
                    count,
                    totalTreeCount
            );
        }

        private static double canonicalizeZero(double value) {
            return value == 0.0 ? 0.0 : value;
        }
    }

    /**
     * Produces structured per-instance output with independently selectable
     * prediction and OOD components.
     *
     * <p>When both components are requested, each tree is traversed once and
     * the reached leaf supplies the tree prediction while the scorer observes
     * the same routing decisions. When only one component is requested, no
     * arrays, maps, scorers, or leaf-membership state required solely by the
     * omitted component are created.</p>
     *
     * <p>This method does not replace {@link #test(ListObjectDataset)}. Ordinary
     * callers that need only legacy predictions and evaluation metrics may keep
     * using {@code test}. Call this method only when structured enhanced output,
     * OOD output, or both are required.</p>
     */
    public ForestPredictionResult[] evaluateEnhanced(
            ListObjectDataset data,
            EnhancedEvaluationOptions options
    ) throws Exception {
        try (ParallelRuntime runtime =
                     new ParallelRuntime(AppContext.num_workers)) {
            return evaluateEnhanced(data, options, runtime);
        }
    }

    /**
     * Runtime-aware structured evaluation entry point.
     */
    public ForestPredictionResult[] evaluateEnhanced(
            ListObjectDataset data,
            EnhancedEvaluationOptions options,
            ParallelRuntime runtime
    ) throws Exception {
        Objects.requireNonNull(data, "Evaluation data cannot be null.");
        Objects.requireNonNull(options, "Enhanced evaluation options cannot be null.");
        Objects.requireNonNull(runtime, "ParallelRuntime cannot be null.");
        options.validate();
        if (options.includeOOD()) {
            requireSplitDistanceOODSupport();
        }

        result.startTimeTest = System.nanoTime();
        try {
            ForestPredictionResult[] outputs = new ForestPredictionResult[data.size()];
            Object[] actualLabels = options.includePredictions()
                    ? new Object[data.size()] : null;
            ProximityTree.Node[][] reachedLeaves = options.includePredictions()
                    ? new ProximityTree.Node[data.size()][trees.length] : null;

            if (data.size() > 1 && runtime.isParallel()) {
                runtime.forRange(0, data.size(), MINIMUM_PREDICTION_RANGE_SIZE, index -> {
                    if (actualLabels != null) actualLabels[index] = data.get_class(index);
                    outputs[index] = evaluateEnhancedInstanceSequentialTrees(
                            resolvePredictionQuery(data.get_series(index)), index, options,
                            reachedLeaves == null ? null : reachedLeaves[index]
                    );
                    reportTestProgress(index);
                });
            } else {
                for (int index = 0; index < data.size(); index++) {
                    if (actualLabels != null) actualLabels[index] = data.get_class(index);
                    outputs[index] = evaluateEnhancedInstance(
                            resolvePredictionQuery(data.get_series(index)), index, options,
                            reachedLeaves == null ? null : reachedLeaves[index], runtime
                    );
                    reportTestProgress(index);
                }
            }

            if (reachedLeaves != null) recordReachedLeaves(reachedLeaves);
            storeEnhancedResults(outputs, options.includePredictions());
            if (options.includePredictions()) {
                calculateEvaluationMetrics(actualLabels, extractPredictedLabels(outputs));
            } else {
                clearPredictionMetrics();
            }
            return outputs;
        } finally {
            result.endTimeTest = System.nanoTime();
            result.elapsedTimeTest = result.endTimeTest - result.startTimeTest;
            if (AppContext.verbosity > 0) System.out.println();
        }
    }

    private static Object[] extractPredictedLabels(ForestPredictionResult[] outputs) {
        Object[] labels = new Object[outputs.length];
        for (int index = 0; index < outputs.length; index++) {
            ForestPredictionResult output = Objects.requireNonNull(
                    outputs[index], "Enhanced result cannot be null at index " + index + "."
            );
            if (!output.hasPrediction()) {
                throw new IllegalStateException("Requested prediction missing at index " + index + ".");
            }
            labels[index] = output.prediction();
        }
        return labels;
    }

    private void clearPredictionMetrics() {
        result.correct = 0;
        result.errors = 0;
        result.score = Double.NaN;
        result.error_rate = Double.NaN;
    }

    /**
     * Stores structured output in the forest result and keeps the legacy
     * prediction list aligned whenever prediction output was requested.
     *
     * <p>OOD-only evaluation clears the legacy prediction list rather than
     * filling it with null placeholders. Structured results remain available
     * through {@link ProximityForestResult#PredictionResults}.</p>
     */
    private void storeEnhancedResults(
            ForestPredictionResult[] outputs,
            boolean predictionsIncluded
    ) {
        result.setPredictionResults(Arrays.asList(outputs));

        if (!predictionsIncluded) {
            result.setPredictions(List.of());
            predictions = result.Predictions;
            return;
        }

        ArrayList<Object> legacyPredictions =
                new ArrayList<>(outputs.length);
        for (int index = 0; index < outputs.length; index++) {
            ForestPredictionResult output = Objects.requireNonNull(
                    outputs[index],
                    "Enhanced evaluation result cannot be null at index "
                            + index + "."
            );
            if (!output.hasPrediction()) {
                throw new IllegalStateException(
                        "Enhanced result at index " + index
                                + " omits a requested prediction."
                );
            }
            legacyPredictions.add(output.prediction());
        }

        result.setPredictions(legacyPredictions);
        predictions = result.Predictions;
        result.validatePredictionAlignment();
    }

    private ForestPredictionResult evaluateEnhancedInstanceSequentialTrees(
            Object resolvedQuery,
            int instanceIndex,
            EnhancedEvaluationOptions options,
            ProximityTree.Node[] reachedLeaves
    ) throws Exception {
        Object[] treePredictions = options.includePredictions()
                ? new Object[trees.length]
                : null;
        OODScoreResult[] treeOODScores = options.includeOOD()
                ? new OODScoreResult[trees.length]
                : null;

        for (int treeIndex = 0; treeIndex < trees.length; treeIndex++) {
            evaluateEnhancedTree(
                    resolvedQuery,
                    instanceIndex,
                    treeIndex,
                    options,
                    treePredictions,
                    treeOODScores,
                    reachedLeaves
            );
        }
        return buildEnhancedResult(options, treePredictions, treeOODScores);
    }

    private ForestPredictionResult evaluateEnhancedInstance(
            Object resolvedQuery,
            int instanceIndex,
            EnhancedEvaluationOptions options,
            ProximityTree.Node[] reachedLeaves,
            ParallelRuntime runtime
    ) throws Exception {
        Object[] treePredictions = options.includePredictions()
                ? new Object[trees.length]
                : null;
        OODScoreResult[] treeOODScores = options.includeOOD()
                ? new OODScoreResult[trees.length]
                : null;

        runtime.forRange(
                0,
                trees.length,
                MINIMUM_TREE_RANGE_SIZE,
                treeIndex -> evaluateEnhancedTree(
                        resolvedQuery,
                        instanceIndex,
                        treeIndex,
                        options,
                        treePredictions,
                        treeOODScores,
                        reachedLeaves
                )
        );
        return buildEnhancedResult(options, treePredictions, treeOODScores);
    }

    private void evaluateEnhancedTree(
            Object resolvedQuery,
            int instanceIndex,
            int treeIndex,
            EnhancedEvaluationOptions options,
            Object[] treePredictions,
            OODScoreResult[] treeOODScores,
            ProximityTree.Node[] reachedLeaves
    ) throws Exception {
        Random random = predictionRandom(instanceIndex, treeIndex);
        ProximityTree.Node leaf;

        if (options.includeOOD()) {
            PathOODScorer scorer = createOODScorer(
                    options.oodScoreType()
            );
            leaf = trees[treeIndex].findLeafResolved(
                    resolvedQuery,
                    random,
                    scorer
            );
            treeOODScores[treeIndex] = scorer.finish();
        } else {
            leaf = trees[treeIndex].findLeafResolved(
                    resolvedQuery,
                    random
            );
        }

        if (options.includePredictions()) {
            reachedLeaves[treeIndex] = leaf;
            treePredictions[treeIndex] = leaf.label();
        }
    }

    private ForestPredictionResult buildEnhancedResult(
            EnhancedEvaluationOptions options,
            Object[] treePredictions,
            OODScoreResult[] treeOODScores
    ) {
        ForestOODScore ood = options.includeOOD()
                ? aggregateTreeOODScores(options.oodScoreType(), treeOODScores)
                : null;

        if (!options.includePredictions()) {
            return ForestPredictionResult.oodOnly(
                    ood.scoreType(),
                    ood.mean(),
                    ood.standardDeviation(),
                    ood.availableTreeCount(),
                    ood.totalTreeCount()
            );
        }

        Object prediction = combineTreePredictions(treePredictions);

        if (LabelUtils.isMultiOutput(prediction)) {
            throw new UnsupportedOperationException(
                    "Enhanced prediction output is not supported for "
                            + "multi-output classification or multi-target "
                            + "regression. Use ordinary prediction "
                            + "output instead."
            );
        }

        if (AppContext.isRegressionMode()) {
            NumericPredictionSummary summary =
                    summarizeNumericPredictions(treePredictions);
            return options.includeOOD()
                    ? ForestPredictionResult.regressionWithOOD(
                    prediction,
                    summary.mean(),
                    summary.standardDeviation(),
                    summary.count(),
                    ood.scoreType(),
                    ood.mean(),
                    ood.standardDeviation(),
                    ood.availableTreeCount(),
                    ood.totalTreeCount()
            )
                    : ForestPredictionResult.regression(
                    prediction,
                    summary.mean(),
                    summary.standardDeviation(),
                    summary.count()
            );
        }

        if (AppContext.isIsolationMode()) {
            return options.includeOOD()
                    ? ForestPredictionResult.isolationWithOOD(
                    prediction,
                    trees.length,
                    ood.scoreType(),
                    ood.mean(),
                    ood.standardDeviation(),
                    ood.availableTreeCount(),
                    ood.totalTreeCount()
            )
                    : ForestPredictionResult.isolation(
                    prediction,
                    trees.length
            );
        }

        Map<Object, Double> probabilities =
                calculateClassVoteProbabilities(treePredictions);
        return options.includeOOD()
                ? ForestPredictionResult.classificationWithOOD(
                prediction,
                probabilities,
                treePredictions.length,
                ood.scoreType(),
                ood.mean(),
                ood.standardDeviation(),
                ood.availableTreeCount(),
                ood.totalTreeCount()
        )
                : ForestPredictionResult.classification(
                prediction,
                probabilities,
                treePredictions.length
        );
    }

    private static NumericPredictionSummary summarizeNumericPredictions(
            Object[] treePredictions
    ) {
        RunningMoments moments = new RunningMoments();
        for (Object prediction : treePredictions) {
            if (prediction instanceof Number number) {
                double value = number.doubleValue();
                if (!Double.isFinite(value)) {
                    throw new IllegalStateException(
                            "Regression tree prediction must be finite, but was "
                                    + value + "."
                    );
                }
                moments.addSigned(value);
            }
        }
        if (moments.count() == 0) {
            throw new IllegalStateException(
                    "No numeric tree prediction was available."
            );
        }
        return new NumericPredictionSummary(
                moments.mean(),
                moments.standardDeviation(),
                moments.count()
        );
    }

    private static Map<Object, Double> calculateClassVoteProbabilities(
            Object[] treePredictions
    ) {
        LinkedHashMap<Object, Integer> counts = new LinkedHashMap<>();
        for (Object prediction : treePredictions) {
            counts.merge(prediction, 1, Integer::sum);
        }
        LinkedHashMap<Object, Double> probabilities = new LinkedHashMap<>();
        for (Map.Entry<Object, Integer> entry : counts.entrySet()) {
            probabilities.put(
                    entry.getKey(),
                    (double) entry.getValue() / treePredictions.length
            );
        }
        return probabilities;
    }

    /**
     * Explicit controls for structured forest evaluation.
     *
     * <p>Use {@link #predictionsOnly()}, {@link #oodOnly(OODScoreType)}, or
     * {@link #predictionsAndOOD(OODScoreType)} for the supported cases.</p>
     */
    public record EnhancedEvaluationOptions(
            boolean includePredictions,
            boolean includeOOD,
            OODScoreType oodScoreType
    ) implements Serializable {
        @Serial
        private static final long serialVersionUID = 2L;

        public static EnhancedEvaluationOptions predictionsOnly() {
            return new EnhancedEvaluationOptions(
                    true,
                    false,
                    null
            );
        }

        public static EnhancedEvaluationOptions oodOnly(
                OODScoreType scoreType
        ) {
            return predictionsAndOrOOD(false, true, scoreType);
        }

        public static EnhancedEvaluationOptions predictionsAndOOD(
                OODScoreType scoreType
        ) {
            return predictionsAndOrOOD(true, true, scoreType);
        }

        private static EnhancedEvaluationOptions predictionsAndOrOOD(
                boolean predictions,
                boolean ood,
                OODScoreType scoreType
        ) {
            return new EnhancedEvaluationOptions(
                    predictions,
                    ood,
                    Objects.requireNonNull(
                            scoreType,
                            "OOD score type cannot be null."
                    )
            );
        }

        private void validate() {
            if (!includePredictions && !includeOOD) {
                throw new IllegalArgumentException(
                        "Enhanced evaluation must request predictions, "
                                + "OOD scores, or both."
                );
            }
            if (!includeOOD) {
                if (oodScoreType != null) {
                    throw new IllegalArgumentException(
                            "Prediction-only evaluation must not specify "
                                    + "an OOD score type."
                    );
                }
                return;
            }
            Objects.requireNonNull(
                    oodScoreType,
                    "OOD score type cannot be null when OOD output is requested."
            );
            createOODScorer(oodScoreType);
        }
    }

    private record NumericPredictionSummary(
            double mean,
            double standardDeviation,
            int count
    ) {
    }

    private Object combineTreePredictions(
            Object[] treePredictions
    ) {
        if (AppContext.isRegressionMode()) {
            return combineRegressionPredictions(treePredictions);
        }
        return combineClassificationPredictions(treePredictions);
    }

    private Object combineRegressionPredictions(
            Object[] treePredictions
    ) {
        Object firstPrediction = firstNonNullPrediction(treePredictions);
        if (firstPrediction == null) {
            return 0.0;
        }
        if (!LabelUtils.isMultiOutput(firstPrediction)) {
            return combineScalarRegressionPredictions(treePredictions);
        }

        int outputCount = LabelUtils.outputCount(firstPrediction);
        List<Double> combined = new ArrayList<>(outputCount);
        for (int outputIndex = 0;
             outputIndex < outputCount;
             outputIndex++) {
            Object[] outputPredictions = new Object[treePredictions.length];
            for (int treeIndex = 0;
                 treeIndex < treePredictions.length;
                 treeIndex++) {
                Object prediction = treePredictions[treeIndex];
                if (prediction == null) {
                    continue;
                }
                if (LabelUtils.outputCount(prediction) != outputCount) {
                    throw new IllegalStateException(
                            "Regression tree predictions have inconsistent output counts."
                    );
                }
                outputPredictions[treeIndex] =
                        LabelUtils.valueAt(prediction, outputIndex);
            }
            combined.add(
                    combineScalarRegressionPredictions(outputPredictions)
            );
        }
        return List.copyOf(combined);
    }

    private Double combineScalarRegressionPredictions(
            Object[] predictions
    ) {
        double[] numericPredictions = new double[predictions.length];
        int count = 0;
        double sum = 0.0;

        for (Object prediction : predictions) {
            if (prediction instanceof Number number) {
                double value = number.doubleValue();
                if (!Double.isFinite(value)) {
                    throw new IllegalStateException(
                            "Regression tree prediction must be finite, but was "
                                    + value + "."
                    );
                }
                numericPredictions[count++] = value;
                sum += value;
            } else if (prediction != null) {
                throw new IllegalStateException(
                        "Regression tree prediction is not numeric: "
                                + prediction
                );
            }
        }

        if (count == 0) {
            return 0.0;
        }
        if (AppContext.voting.equalsIgnoreCase("mean")) {
            return sum / count;
        }
        if (AppContext.voting.equalsIgnoreCase("median")) {
            Arrays.sort(numericPredictions, 0, count);
            if ((count & 1) == 1) {
                return numericPredictions[count / 2];
            }
            return (numericPredictions[count / 2 - 1]
                    + numericPredictions[count / 2]) / 2.0;
        }
        throw new IllegalArgumentException(
                "Unknown voting method: " + AppContext.voting
        );
    }


    private Object combineClassificationPredictions(
            Object[] treePredictions
    ) {
        Object firstPrediction = firstNonNullPrediction(treePredictions);
        if (firstPrediction == null) {
            return null;
        }
        if (!LabelUtils.isMultiOutput(firstPrediction)) {
            return combineScalarClassificationPredictions(treePredictions);
        }

        int outputCount = LabelUtils.outputCount(firstPrediction);
        List<Object> combined = new ArrayList<>(outputCount);
        for (int outputIndex = 0;
             outputIndex < outputCount;
             outputIndex++) {
            Object[] outputPredictions = new Object[treePredictions.length];
            for (int treeIndex = 0;
                 treeIndex < treePredictions.length;
                 treeIndex++) {
                Object prediction = treePredictions[treeIndex];
                if (prediction == null) {
                    continue;
                }
                if (LabelUtils.outputCount(prediction) != outputCount) {
                    throw new IllegalStateException(
                            "Classification tree predictions have inconsistent output counts."
                    );
                }
                outputPredictions[treeIndex] =
                        LabelUtils.valueAt(prediction, outputIndex);
            }
            combined.add(
                    combineScalarClassificationPredictions(outputPredictions)
            );
        }
        return List.copyOf(combined);
    }

    private Object combineScalarClassificationPredictions(
            Object[] predictions
    ) {
        Map<Object, Integer> voteCounts = new LinkedHashMap<>();
        Object majority = null;
        int maximumCount = 0;

        for (Object prediction : predictions) {
            if (prediction == null) {
                continue;
            }
            int count = voteCounts.getOrDefault(prediction, 0) + 1;
            voteCounts.put(prediction, count);
            if (count > maximumCount) {
                maximumCount = count;
                majority = prediction;
            }
        }
        return majority;
    }

    private static Object firstNonNullPrediction(
            Object[] predictions
    ) {
        for (Object prediction : predictions) {
            if (prediction != null) {
                return prediction;
            }
        }
        return null;
    }


    private void recordReachedLeaves(
            ProximityTree.Node[][] reachedLeaves
    ) {
        for (int testIndex = 0;
             testIndex < reachedLeaves.length;
             testIndex++) {
            recordReachedLeaves(reachedLeaves, testIndex);
        }
    }

    private void recordReachedLeaves(
            ProximityTree.Node[][] reachedLeaves,
            int recordedTestIndex
    ) {
        if (reachedLeaves.length == 0) {
            return;
        }

        ProximityTree.Node[] leavesForInstance =
                reachedLeaves[
                        reachedLeaves.length == 1
                                ? 0
                                : recordedTestIndex
                        ];

        for (ProximityTree.Node leaf : leavesForInstance) {
            if (leaf != null) {
                leaf.TestIndices.add(recordedTestIndex);
            }
        }
    }

    private void calculateEvaluationMetrics(
            Object[] actualLabels,
            Object[] predictedLabels
    ) {
        int correct =
                0;
        int errors =
                0;
        int validLabelCount =
                0;

        for (int index = 0;
             index < actualLabels.length;
             index++) {
            Object actual =
                    actualLabels[index];

            if (actual == null) {
                continue;
            }

            validLabelCount++;
            if (Objects.equals(actual, predictedLabels[index])) {
                correct++;
            } else {
                errors++;
            }
        }

        result.correct =
                correct;
        result.errors =
                errors;

        if (validLabelCount == 0 || !AppContext.exists_testlabels) {
            result.score =
                    Double.NaN;
            result.error_rate =
                    Double.NaN;
            return;
        }

        if (AppContext.isRegressionMode()) {
            result.score =
                    calculateRegressionScore(
                            actualLabels,
                            predictedLabels
                    );
        } else {
            result.score =
                    (double) correct / validLabelCount;
        }

        result.error_rate =
                1.0 - result.score;
    }

    /** Calculates R-squared while preserving actual/predicted alignment. */
    private double calculateRegressionScore(
            Object[] actualLabels,
            Object[] predictedLabels
    ) {
        Object firstActual = firstNonNullPrediction(actualLabels);
        if (firstActual == null) {
            return Double.NaN;
        }
        if (!LabelUtils.isMultiOutput(firstActual)) {
            return calculateScalarRegressionScore(
                    actualLabels,
                    predictedLabels
            );
        }

        int outputCount = LabelUtils.outputCount(firstActual);
        double scoreSum = 0.0;
        int scoreCount = 0;
        for (int outputIndex = 0;
             outputIndex < outputCount;
             outputIndex++) {
            Object[] actualOutput = new Object[actualLabels.length];
            Object[] predictedOutput = new Object[predictedLabels.length];
            for (int index = 0; index < actualLabels.length; index++) {
                Object actual = actualLabels[index];
                if (actual == null) {
                    continue;
                }
                Object predicted = predictedLabels[index];
                if (predicted == null) {
                    throw new IllegalStateException(
                            "Regression prediction missing at index "
                                    + index + "."
                    );
                }
                if (LabelUtils.outputCount(actual) != outputCount
                        || LabelUtils.outputCount(predicted) != outputCount) {
                    throw new IllegalStateException(
                            "Regression labels and predictions have inconsistent output counts."
                    );
                }
                actualOutput[index] = LabelUtils.valueAt(
                        actual,
                        outputIndex
                );
                predictedOutput[index] = LabelUtils.valueAt(
                        predicted,
                        outputIndex
                );
            }
            double outputScore = calculateScalarRegressionScore(
                    actualOutput,
                    predictedOutput
            );
            if (!Double.isNaN(outputScore)) {
                scoreSum += outputScore;
                scoreCount++;
            }
        }
        return scoreCount == 0
                ? Double.NaN
                : scoreSum / scoreCount;
    }

    private double calculateScalarRegressionScore(
            Object[] actualLabels,
            Object[] predictedLabels
    ) {
        int count = 0;
        double sum = 0.0;
        for (Object actual : actualLabels) {
            if (actual instanceof Number number) {
                sum += number.doubleValue();
                count++;
            }
        }
        if (count == 0) {
            return Double.NaN;
        }

        double mean = sum / count;
        double totalSumOfSquares = 0.0;
        double residualSumOfSquares = 0.0;
        for (int index = 0; index < actualLabels.length; index++) {
            Object actual = actualLabels[index];
            if (!(actual instanceof Number actualNumber)) {
                continue;
            }
            Object predicted = predictedLabels[index];
            if (!(predicted instanceof Number predictedNumber)) {
                throw new IllegalStateException(
                        "Regression prediction at index " + index
                                + " is not numeric: " + predicted
                );
            }
            double actualValue = actualNumber.doubleValue();
            double predictedValue = predictedNumber.doubleValue();
            double centered = actualValue - mean;
            double residual = actualValue - predictedValue;
            totalSumOfSquares += centered * centered;
            residualSumOfSquares += residual * residual;
        }
        if (totalSumOfSquares == 0.0) {
            return residualSumOfSquares == 0.0 ? 1.0 : 0.0;
        }
        return 1.0 - residualSumOfSquares / totalSumOfSquares;
    }


    private void reportTestProgress(int testIndex) {
        if (AppContext.verbosity <= 0) {
            return;
        }

        int interval =
                Math.max(
                        1,
                        AppContext.print_test_progress_for_each_instances
                );

        if (testIndex % interval == 0) {
            synchronized (System.out) {
                System.out.print("*");
            }
        }
    }

    private Random predictionRandom(
            int testIndex,
            int treeIndex
    ) {
        long seed =
                forestSeed;
        seed =
                mixSeed(seed, forest_id);
        seed =
                mixSeed(seed, treeIndex);
        seed =
                mixSeed(seed, testIndex);
        return new Random(seed);
    }

    private static long mixSeed(
            long seed,
            int value
    ) {
        long mixed =
                seed
                        ^ (0x9E3779B97F4A7C15L * (value + 1L));
        mixed =
                (mixed ^ (mixed >>> 30))
                        * 0xBF58476D1CE4E5B9L;
        mixed =
                (mixed ^ (mixed >>> 27))
                        * 0x94D049BB133111EBL;
        return mixed ^ (mixed >>> 31);
    }

    /** Returns whether every trained internal splitter retains OOD summaries. */
    public boolean supportsSplitDistanceOOD() {
        if (trees == null || trees.length == 0) return false;
        for (ProximityTree tree : trees) {
            if (tree == null || tree.getRootNode() == null
                    || !nodeSupportsSplitDistanceOOD(tree.getRootNode())) return false;
        }
        return true;
    }

    /** Throws when this trained forest cannot support split-distance OOD scoring. */
    public void requireSplitDistanceOODSupport() {
        if (!supportsSplitDistanceOOD()) {
            throw new IllegalStateException(
                    "The forest lacks branch-local split-distance summaries. "
                            + "Train with collect_split_distance_summaries=true."
            );
        }
    }

    private static boolean nodeSupportsSplitDistanceOOD(ProximityTree.Node node) {
        if (node.is_leaf()) return true;
        Splitter splitter = node.getSplitter();
        if (splitter == null || splitter.getSplitDistanceSummary() == null) return false;
        ProximityTree.Node[] children = node.get_children();
        if (children == null || children.length == 0) return false;
        for (ProximityTree.Node child : children) {
            if (child == null || !nodeSupportsSplitDistanceOOD(child)) return false;
        }
        return true;
    }

    public ProximityTree[] getTrees() {
        return trees;
    }

    public ProximityTree getTree(int index) {
        return trees[index];
    }

    public ProximityForestResult getResultSet() {
        return result;
    }

    public ProximityForestResult getForestStatCollection() {
        result.collateResults();
        return result;
    }

    public int getForestID() {
        return forest_id;
    }

    public void setForestID(int forestId) {
        this.forest_id = forestId;
    }
}