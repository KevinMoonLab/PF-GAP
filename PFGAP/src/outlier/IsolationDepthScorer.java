package outlier;

import core.AppContext;
import datasets.ListObjectDataset;
import datasets.readers.lazy.LazySeriesRef;
import trees.ProximityForest;
import trees.ProximityTree;

import java.util.Random;

/**
 * Isolation-Forest-style path-length scoring for proximity forests.
 *
 * <p>Higher scores indicate stronger anomalous or outlying behavior.</p>
 */
public final class IsolationDepthScorer {

    private IsolationDepthScorer() {
        // Utility class.
    }

    /**
     * Computes anomaly scores using the scored dataset size as the
     * normalization size.
     */
    public static double[] score(
            ProximityForest forest,
            ListObjectDataset data
    ) throws Exception {

        if (data == null) {
            return new double[0];
        }

        return score(
                forest,
                data,
                data.size()
        );
    }

    /**
     * Computes anomaly scores with an explicit normalization sample size.
     *
     * <p>For test scoring, normalizationSampleSize should normally be the
     * training sample size rather than the number of test instances.</p>
     */
    public static double[] score(
            ProximityForest forest,
            ListObjectDataset data,
            int normalizationSampleSize
    ) throws Exception {

        validateForest(
                forest
        );

        if (data == null) {
            return new double[0];
        }

        double[] scores =
                new double[data.size()];

        double normalizer =
                correction(
                        normalizationSampleSize
                );

        for (int index = 0;
             index < data.size();
             index++) {

            Object storedSeries =
                    data.get_series(
                            index
                    );

            Object resolvedSeries =
                    resolveSeries(
                            storedSeries
                    );

            double averagePathLength =
                    averagePathLengthResolved(
                            forest,
                            resolvedSeries,
                            stableInstanceIdentity(
                                    data,
                                    index
                            )
                    );

            scores[index] =
                    scoreFromPathLength(
                            averagePathLength,
                            normalizer
                    );
        }

        return scores;
    }

    /**
     * Computes an anomaly score for one stored or materialized series.
     */
    public static double scoreSeries(
            ProximityForest forest,
            Object series,
            int normalizationSampleSize
    ) throws Exception {

        validateForest(
                forest
        );

        Object resolvedSeries =
                resolveSeries(
                        series
                );

        double averagePathLength =
                averagePathLengthResolved(
                        forest,
                        resolvedSeries,
                        0
                );

        return scoreFromPathLength(
                averagePathLength,
                correction(
                        normalizationSampleSize
                )
        );
    }

    private static double scoreFromPathLength(
            double averagePathLength,
            double normalizer
    ) {
        if (normalizer <= 0.0) {
            return averagePathLength <= 0.0
                    ? 1.0
                    : 0.0;
        }

        return Math.pow(
                2.0,
                -averagePathLength / normalizer
        );
    }

    /**
     * Computes average corrected path length for one stored or materialized
     * series.
     */
    public static double averagePathLength(
            ProximityForest forest,
            Object series
    ) throws Exception {

        validateForest(
                forest
        );

        return averagePathLengthResolved(
                forest,
                resolveSeries(
                        series
                ),
                0
        );
    }

    /**
     * Computes average corrected path length for an already materialized
     * series.
     *
     * <p>The series is reused across every tree and is not reread at each
     * node.</p>
     */
    public static double averagePathLengthResolved(
            ProximityForest forest,
            Object resolvedSeries,
            int instanceIdentity
    ) throws Exception {

        validateForest(
                forest
        );

        if (resolvedSeries == null) {
            throw new IllegalArgumentException(
                    "Resolved series cannot be null."
            );
        }

        if (resolvedSeries instanceof LazySeriesRef) {
            throw new IllegalArgumentException(
                    "averagePathLengthResolved() received a "
                            + "LazySeriesRef."
            );
        }

        ProximityTree[] trees =
                forest.getTrees();

        double total =
                0.0;

        int validTreeCount =
                0;

        for (ProximityTree tree : trees) {
            if (tree == null
                    || tree.getRootNode() == null) {

                continue;
            }

            total +=
                    pathLengthResolved(
                            tree,
                            resolvedSeries,
                            new Random(
                                    mixSeed(
                                            instanceIdentity,
                                            tree.getTreeID()
                                    )
                            )
                    );

            validTreeCount++;
        }

        if (validTreeCount == 0) {
            throw new IllegalStateException(
                    "Cannot compute isolation score: "
                            + "forest has no valid trees."
            );
        }

        return total / validTreeCount;
    }

    /**
     * Compatibility path-length method for stored or eager input.
     */
    public static double pathLength(
            ProximityTree tree,
            Object series
    ) throws Exception {

        return pathLengthResolved(
                tree,
                resolveSeries(
                        series
                ),
                new Random(
                        mixSeed(
                                0,
                                tree == null
                                        ? 0
                                        : tree.getTreeID()
                        )
                )
        );
    }

    /**
     * Computes corrected path length for one tree using an already
     * materialized series.
     */
    public static double pathLengthResolved(
            ProximityTree tree,
            Object resolvedSeries,
            Random random
    ) throws Exception {

        if (tree == null) {
            throw new IllegalArgumentException(
                    "tree cannot be null."
            );
        }

        if (resolvedSeries == null) {
            throw new IllegalArgumentException(
                    "Resolved series cannot be null."
            );
        }

        ProximityTree.Node node =
                tree.getRootNode();

        if (node == null) {
            return 0.0;
        }

        int depth =
                0;

        while (!node.is_leaf()) {
            if (node.getSplitter() == null
                    || node.get_children() == null) {

                break;
            }

            int branch =
                    node.getSplitter()
                            .findClosestBranchResolved(
                                    resolvedSeries,
                                    random
                            );

            ProximityTree.Node[] children =
                    node.get_children();

            if (branch < 0
                    || branch >= children.length
                    || children[branch] == null) {

                break;
            }

            node =
                    children[branch];

            depth++;
        }

        return depth
                + correction(
                leafSize(
                        node
                )
        );
    }

    private static Object resolveSeries(
            Object series
    ) {
        if (series == null) {
            throw new IllegalArgumentException(
                    "Series cannot be null."
            );
        }

        if (series instanceof LazySeriesRef reference) {
            return AppContext.readLazySeries(
                    reference
            );
        }

        return series;
    }

    private static void validateForest(
            ProximityForest forest
    ) {
        if (forest == null) {
            throw new IllegalArgumentException(
                    "forest cannot be null."
            );
        }

        ProximityTree[] trees =
                forest.getTrees();

        if (trees == null || trees.length == 0) {
            throw new IllegalStateException(
                    "Cannot compute isolation score: "
                            + "forest has no trees."
            );
        }
    }

    private static int stableInstanceIdentity(
            ListObjectDataset data,
            int localIndex
    ) {
        Integer originalIndex =
                data.get_index(
                        localIndex
                );

        return originalIndex == null
                ? localIndex
                : originalIndex;
    }

    private static long mixSeed(
            int instanceIdentity,
            int treeIdentity
    ) {
        long value =
                0x9E3779B97F4A7C15L;

        value ^=
                0xBF58476D1CE4E5B9L
                        * (
                        instanceIdentity + 1L
                );

        value ^=
                0x94D049BB133111EBL
                        * (
                        treeIdentity + 1L
                );

        value =
                (value ^ (value >>> 30))
                        * 0xBF58476D1CE4E5B9L;

        value =
                (value ^ (value >>> 27))
                        * 0x94D049BB133111EBL;

        return value
                ^ (value >>> 31);
    }

    private static int leafSize(
            ProximityTree.Node node
    ) {
        if (node == null
                || node.getInBagIndices() == null) {

            return 1;
        }

        return Math.max(
                1,
                node.getInBagIndices()
                        .size()
        );
    }

    /**
     * Standard Isolation Forest path-length correction:
     *
     * <pre>
     * c(n) = 2 H(n - 1) - 2(n - 1)/n
     * </pre>
     */
    public static double correction(
            int n
    ) {
        if (n <= 1) {
            return 0.0;
        }

        if (n == 2) {
            return 1.0;
        }

        return 2.0
                * harmonic(
                n - 1
        )
                - (
                2.0
                        * (
                        n - 1
                )
                        / n
        );
    }

    private static double harmonic(
            int n
    ) {
        double sum =
                0.0;

        for (int value = 1;
             value <= n;
             value++) {

            sum +=
                    1.0 / value;
        }

        return sum;
    }
}