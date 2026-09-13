package trees;

import core.AppContext;
import core.ProximityForestResult;
import core.parallel.ParallelRuntime;
import datasets.ListObjectDataset;
import datasets.readers.lazy.LazySeriesRef;
import distance.MEASURE;
import util.PrintUtilities;

import java.io.Serial;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
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

        result.Predictions =
                new ArrayList<>(Arrays.asList(predictedLabels));
        predictions =
                result.Predictions;

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
        double[] numericPredictions =
                new double[treePredictions.length];
        int count =
                0;
        double sum =
                0.0;

        for (Object prediction : treePredictions) {
            if (prediction instanceof Number number) {
                double value =
                        number.doubleValue();
                numericPredictions[count++] =
                        value;
                sum +=
                        value;
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
            return (
                    numericPredictions[count / 2 - 1]
                            + numericPredictions[count / 2]
            ) / 2.0;
        }

        throw new IllegalArgumentException(
                "Unknown voting method: " + AppContext.voting
        );
    }

    private Object combineClassificationPredictions(
            Object[] treePredictions
    ) {
        Map<Object, Integer> voteCounts =
                new HashMap<>();
        Object majority =
                null;
        int maximumCount =
                0;

        for (Object prediction : treePredictions) {
            int count =
                    voteCounts.getOrDefault(prediction, 0) + 1;
            voteCounts.put(prediction, count);

            if (count > maximumCount) {
                maximumCount =
                        count;
                majority =
                        prediction;
            }
        }

        return majority;
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
        int count =
                0;
        double sum =
                0.0;

        for (Object actual : actualLabels) {
            if (actual instanceof Number number) {
                sum += number.doubleValue();
                count++;
            }
        }

        if (count == 0) {
            return Double.NaN;
        }

        double mean =
                sum / count;
        double totalSumOfSquares =
                0.0;
        double residualSumOfSquares =
                0.0;

        for (int index = 0;
             index < actualLabels.length;
             index++) {
            Object actual =
                    actualLabels[index];

            if (!(actual instanceof Number actualNumber)) {
                continue;
            }

            Object predicted =
                    predictedLabels[index];

            if (!(predicted instanceof Number predictedNumber)) {
                throw new IllegalStateException(
                        "Regression prediction at index "
                                + index
                                + " is not numeric: "
                                + predicted
                );
            }

            double actualValue =
                    actualNumber.doubleValue();
            double predictedValue =
                    predictedNumber.doubleValue();
            double centered =
                    actualValue - mean;
            double residual =
                    actualValue - predictedValue;

            totalSumOfSquares +=
                    centered * centered;
            residualSumOfSquares +=
                    residual * residual;
        }

        if (totalSumOfSquares == 0.0) {
            return residualSumOfSquares == 0.0
                    ? 1.0
                    : Double.NEGATIVE_INFINITY;
        }

        return 1.0
                - residualSumOfSquares / totalSumOfSquares;
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
