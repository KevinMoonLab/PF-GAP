package outlier;

import core.AppContext;
import datasets.ListObjectDataset;
import datasets.readers.lazy.LazySeriesRef;
import trees.ProximityForest;
import trees.ProximityTree;

import java.util.HashMap;
import java.util.Map;
import java.util.Random;

/**
 * Isolation-Forest-style path-length scoring for proximity forests.
 *
 * <p>Higher scores indicate stronger anomalous or outlying behavior.</p>
 *
 * <p>Training data and inference data use deliberately separate scoring
 * paths:</p>
 *
 * <ul>
 *     <li>
 *         {@link #scoreTraining(ProximityForest, ListObjectDataset)}
 *         uses stored terminal-node memberships and does not read series,
 *         invoke splitters, or compute distances.
 *     </li>
 *     <li>
 *         {@link #scoreInference(ProximityForest, ListObjectDataset, int)}
 *         routes each supplied instance through the forest and therefore
 *         invokes the configured distance measures.
 *     </li>
 * </ul>
 */
public final class IsolationDepthScorer {

    private IsolationDepthScorer() {
        // Utility class.
    }

    /**
     * Computes isolation scores for the forest's original training instances
     * using only stored leaf memberships.
     *
     * <p>No series are materialized and no distances are computed.</p>
     *
     * <p>The returned array is aligned with the current row order of
     * {@code trainingData}. Stable instance identities are used to connect
     * the supplied dataset rows to the membership information stored in the
     * trained trees.</p>
     *
     * <p>The training dataset size is used as the normalization sample
     * size.</p>
     */
    public static double[] scoreTraining(
            ProximityForest forest,
            ListObjectDataset trainingData
    ) {
        if (trainingData == null) {
            return new double[0];
        }

        return scoreTraining(
                forest,
                trainingData,
                trainingData.size()
        );
    }

    /**
     * Computes isolation scores for the forest's original training instances
     * using only stored leaf memberships and an explicit normalization sample
     * size.
     *
     * <p>Each original training instance contributes exactly once per tree.
     * Bootstrap duplicates are therefore suppressed during assignment.
     * When bootstrap sampling was enabled, terminal OOB memberships are also
     * used so that every original training instance receives one contribution
     * from every tree.</p>
     */
    public static double[] scoreTraining(
            ProximityForest forest,
            ListObjectDataset trainingData,
            int normalizationSampleSize
    ) {
        validateForest(
                forest
        );

        if (trainingData == null || trainingData.size() == 0) {
            return new double[0];
        }

        int trainingSize =
                trainingData.size();

        IdentityLookup identityLookup =
                buildIdentityLookup(
                        trainingData
                );

        double[] correctionLookup =
                buildCorrectionLookup(
                        Math.max(
                                trainingSize,
                                largestStoredLeafSize(
                                        forest
                                )
                        )
                );

        double[] pathLengthSums =
                new double[trainingSize];

        int[] assignmentStamps =
                new int[trainingSize];

        int currentStamp =
                0;

        int validTreeCount =
                0;

        for (ProximityTree tree
                : forest.getTrees()) {

            if (tree == null
                    || tree.getRootNode() == null) {

                continue;
            }

            currentStamp =
                    nextStamp(
                            assignmentStamps,
                            currentStamp
                    );

            accumulateTrainingTree(
                    tree,
                    identityLookup,
                    pathLengthSums,
                    assignmentStamps,
                    currentStamp,
                    correctionLookup
            );

            validateCompleteTreeCoverage(
                    tree,
                    assignmentStamps,
                    currentStamp
            );

            validTreeCount++;
        }

        if (validTreeCount == 0) {
            throw new IllegalStateException(
                    "Cannot compute training isolation scores: "
                            + "forest has no valid trees."
            );
        }

        double normalizer =
                correction(
                        normalizationSampleSize
                );

        double[] scores =
                new double[trainingSize];

        for (int position = 0;
             position < trainingSize;
             position++) {

            double averagePathLength =
                    pathLengthSums[position]
                            / validTreeCount;

            scores[position] =
                    scoreFromPathLength(
                            averagePathLength,
                            normalizer
                    );
        }

        return scores;
    }

    /**
     * Computes isolation scores for inference data.
     *
     * <p>Unlike training scoring, inference scoring must route each instance
     * through every tree. Splitter routing can therefore invoke expensive
     * distance measures such as DTW.</p>
     *
     * <p>The supplied dataset size is used as the normalization sample
     * size.</p>
     */
    public static double[] scoreInference(
            ProximityForest forest,
            ListObjectDataset data
    ) throws Exception {
        if (data == null) {
            return new double[0];
        }

        return scoreInference(
                forest,
                data,
                data.size()
        );
    }

    /**
     * Computes isolation scores for inference data using an explicit
     * normalization sample size.
     *
     * <p>For test scoring, {@code normalizationSampleSize} should normally
     * be the training sample size rather than the number of test
     * instances.</p>
     */
    public static double[] scoreInference(
            ProximityForest forest,
            ListObjectDataset data,
            int normalizationSampleSize
    ) throws Exception {
        validateForest(
                forest
        );

        if (data == null || data.size() == 0) {
            return new double[0];
        }

        double[] scores =
                new double[data.size()];

        double normalizer =
                correction(
                        normalizationSampleSize
                );

        for (int position = 0;
             position < data.size();
             position++) {

            Object resolvedSeries =
                    resolveSeries(
                            data.get_series(
                                    position
                            )
                    );

            double averagePathLength =
                    averageInferencePathLengthResolved(
                            forest,
                            resolvedSeries,
                            stableInstanceIdentity(
                                    data,
                                    position
                            )
                    );

            scores[position] =
                    scoreFromPathLength(
                            averagePathLength,
                            normalizer
                    );
        }

        return scores;
    }

    /**
     * Computes an inference isolation score for one stored or materialized
     * series.
     */
    public static double scoreInferenceSeries(
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
                averageInferencePathLengthResolved(
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

    /**
     * Computes average corrected inference path length for one stored or
     * materialized series.
     */
    public static double averageInferencePathLength(
            ProximityForest forest,
            Object series
    ) throws Exception {
        validateForest(
                forest
        );

        return averageInferencePathLengthResolved(
                forest,
                resolveSeries(
                        series
                ),
                0
        );
    }

    /**
     * Computes average corrected inference path length for an already
     * materialized series.
     *
     * <p>The resolved series is reused across every tree and is not reread
     * at each node.</p>
     */
    public static double averageInferencePathLengthResolved(
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
                    "averageInferencePathLengthResolved() received "
                            + "a LazySeriesRef."
            );
        }

        double total =
                0.0;

        int validTreeCount =
                0;

        for (ProximityTree tree
                : forest.getTrees()) {

            if (tree == null
                    || tree.getRootNode() == null) {

                continue;
            }

            total +=
                    inferencePathLengthResolved(
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
                    "Cannot compute inference isolation score: "
                            + "forest has no valid trees."
            );
        }

        return total / validTreeCount;
    }

    /**
     * Computes corrected path length for one tree using an already
     * materialized inference instance.
     */
    public static double inferencePathLengthResolved(
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

        if (random == null) {
            throw new IllegalArgumentException(
                    "random cannot be null."
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
                storedLeafSampleSize(
                        node
                )
        );
    }

    /**
     * Adds one tree's stored training path lengths to the forest-level sum.
     */
    private static void accumulateTrainingTree(
            ProximityTree tree,
            IdentityLookup identityLookup,
            double[] pathLengthSums,
            int[] assignmentStamps,
            int currentStamp,
            double[] correctionLookup
    ) {
        if (tree.getLeaves() == null
                || tree.getLeaves().isEmpty()) {

            throw new IllegalStateException(
                    "Tree "
                            + tree.getTreeID()
                            + " has no stored leaves."
            );
        }

        ProximityTree.Node root =
                tree.getRootNode();

        for (ProximityTree.Node leaf
                : tree.getLeaves()) {

            if (leaf == null) {
                continue;
            }

            int leafSampleSize =
                    storedLeafSampleSize(
                            leaf
                    );

            double correctedPathLength =
                    leaf.getNodeDepth()
                            + correctionLookup[
                            Math.min(
                                    leafSampleSize,
                                    correctionLookup.length - 1
                            )
                            ];

            boolean rootLeaf =
                    leaf == root;

            /*
             * Root bootstrap metadata currently stores local positions.
             * Non-root node and leaf metadata store stable instance
             * identities propagated through dataset subsets.
             */
            if (rootLeaf) {
                accumulateRootLeafMembership(
                        leaf,
                        correctedPathLength,
                        pathLengthSums,
                        assignmentStamps,
                        currentStamp
                );
            } else {
                accumulateIdentityMembership(
                        tree,
                        leaf.getInBagIndices(),
                        correctedPathLength,
                        identityLookup,
                        pathLengthSums,
                        assignmentStamps,
                        currentStamp,
                        "in-bag"
                );

                accumulateIdentityMembership(
                        tree,
                        leaf.getOutOfBagIndices(),
                        correctedPathLength,
                        identityLookup,
                        pathLengthSums,
                        assignmentStamps,
                        currentStamp,
                        "out-of-bag"
                );
            }
        }
    }

    /**
     * Handles the special case in which the root itself is a leaf.
     *
     * <p>The current tree implementation stores local source positions in
     * root bootstrap metadata rather than stable instance identities.</p>
     */
    private static void accumulateRootLeafMembership(
            ProximityTree.Node rootLeaf,
            double correctedPathLength,
            double[] pathLengthSums,
            int[] assignmentStamps,
            int currentStamp
    ) {
        accumulatePositionMembership(
                rootLeaf.getInBagIndices(),
                correctedPathLength,
                pathLengthSums,
                assignmentStamps,
                currentStamp,
                "root in-bag"
        );

        accumulatePositionMembership(
                rootLeaf.getOutOfBagIndices(),
                correctedPathLength,
                pathLengthSums,
                assignmentStamps,
                currentStamp,
                "root out-of-bag"
        );
    }

    private static void accumulatePositionMembership(
            Iterable<Integer> positions,
            double correctedPathLength,
            double[] pathLengthSums,
            int[] assignmentStamps,
            int currentStamp,
            String membershipType
    ) {
        if (positions == null) {
            return;
        }

        for (Integer position
                : positions) {

            if (position == null) {
                throw new IllegalStateException(
                        "Null position found in "
                                + membershipType
                                + " membership."
                );
            }

            if (position < 0
                    || position >= pathLengthSums.length) {

                throw new IllegalStateException(
                        "Stored "
                                + membershipType
                                + " position "
                                + position
                                + " is outside the training dataset range."
                );
            }

            assignOnce(
                    position,
                    correctedPathLength,
                    pathLengthSums,
                    assignmentStamps,
                    currentStamp
            );
        }
    }

    private static void accumulateIdentityMembership(
            ProximityTree tree,
            Iterable<Integer> identities,
            double correctedPathLength,
            IdentityLookup identityLookup,
            double[] pathLengthSums,
            int[] assignmentStamps,
            int currentStamp,
            String membershipType
    ) {
        if (identities == null) {
            return;
        }

        for (Integer identity
                : identities) {

            if (identity == null) {
                throw new IllegalStateException(
                        "Tree "
                                + tree.getTreeID()
                                + " contains a null "
                                + membershipType
                                + " training identity."
                );
            }

            int outputPosition =
                    identityLookup.positionOf(
                            identity
                    );

            if (outputPosition < 0) {
                throw new IllegalStateException(
                        "Tree "
                                + tree.getTreeID()
                                + " contains "
                                + membershipType
                                + " training identity "
                                + identity
                                + ", but that identity is absent from "
                                + "the supplied training dataset."
                );
            }

            assignOnce(
                    outputPosition,
                    correctedPathLength,
                    pathLengthSums,
                    assignmentStamps,
                    currentStamp
            );
        }
    }

    /**
     * Adds at most one contribution for an original instance in one tree.
     *
     * <p>This suppresses repeated bootstrap occurrences without allocating
     * a set for every leaf.</p>
     */
    private static void assignOnce(
            int outputPosition,
            double correctedPathLength,
            double[] pathLengthSums,
            int[] assignmentStamps,
            int currentStamp
    ) {
        if (assignmentStamps[outputPosition]
                == currentStamp) {

            return;
        }

        pathLengthSums[outputPosition] +=
                correctedPathLength;

        assignmentStamps[outputPosition] =
                currentStamp;
    }

    private static void validateCompleteTreeCoverage(
            ProximityTree tree,
            int[] assignmentStamps,
            int currentStamp
    ) {
        for (int position = 0;
             position < assignmentStamps.length;
             position++) {

            if (assignmentStamps[position]
                    != currentStamp) {

                throw new IllegalStateException(
                        "Training instance at output position "
                                + position
                                + " has no stored terminal membership in tree "
                                + tree.getTreeID()
                                + ". The supplied dataset may not be the "
                                + "dataset used to train this forest, or the "
                                + "tree membership metadata may be incomplete."
                );
            }
        }
    }

    /**
     * Determines whether stable identities are canonical positions and builds
     * a fallback map only when required.
     */
    private static IdentityLookup buildIdentityLookup(
            ListObjectDataset trainingData
    ) {
        boolean canonical =
                true;

        for (int position = 0;
             position < trainingData.size();
             position++) {

            Integer identity =
                    trainingData.get_index(
                            position
                    );

            if (identity == null) {
                throw new IllegalArgumentException(
                        "Training instance at position "
                                + position
                                + " has a null stable identity."
                );
            }

            if (identity != position) {
                canonical =
                        false;
            }
        }

        if (canonical) {
            return new IdentityLookup(
                    trainingData.size(),
                    null
            );
        }

        Map<Integer, Integer> identityToPosition =
                new HashMap<>(
                        hashMapCapacity(
                                trainingData.size()
                        )
                );

        for (int position = 0;
             position < trainingData.size();
             position++) {

            Integer identity =
                    trainingData.get_index(
                            position
                    );

            Integer previousPosition =
                    identityToPosition.put(
                            identity,
                            position
                    );

            if (previousPosition != null) {
                throw new IllegalArgumentException(
                        "Training identity "
                                + identity
                                + " occurs at both position "
                                + previousPosition
                                + " and position "
                                + position
                                + ". Stable training identities must be "
                                + "unique."
                );
            }
        }

        return new IdentityLookup(
                trainingData.size(),
                identityToPosition
        );
    }

    private static int hashMapCapacity(
            int expectedSize
    ) {
        if (expectedSize < 3) {
            return expectedSize + 1;
        }

        if (expectedSize
                >= 1_073_741_824) {

            return Integer.MAX_VALUE;
        }

        return (int) (
                expectedSize / 0.75f
        ) + 1;
    }

    /**
     * Finds the largest stored in-bag leaf size so that every correction
     * needed by the forest can be obtained by direct array lookup.
     */
    private static int largestStoredLeafSize(
            ProximityForest forest
    ) {
        int maximum =
                1;

        for (ProximityTree tree
                : forest.getTrees()) {

            if (tree == null
                    || tree.getLeaves() == null) {

                continue;
            }

            for (ProximityTree.Node leaf
                    : tree.getLeaves()) {

                maximum =
                        Math.max(
                                maximum,
                                storedLeafSampleSize(
                                        leaf
                                )
                        );
            }
        }

        return maximum;
    }

    private static int storedLeafSampleSize(
            ProximityTree.Node leaf
    ) {
        if (leaf == null
                || leaf.getInBagIndices() == null) {

            return 1;
        }

        return Math.max(
                1,
                leaf.getInBagIndices()
                        .size()
        );
    }

    /**
     * Builds all standard Isolation Forest corrections from zero through
     * {@code maximumSize} in one linear pass.
     */
    private static double[] buildCorrectionLookup(
            int maximumSize
    ) {
        int safeMaximum =
                Math.max(
                        2,
                        maximumSize
                );

        double[] corrections =
                new double[safeMaximum + 1];

        corrections[0] =
                0.0;

        corrections[1] =
                0.0;

        corrections[2] =
                1.0;

        double harmonic =
                1.0;

        for (int size = 3;
             size <= safeMaximum;
             size++) {

            harmonic +=
                    1.0
                            / (size - 1);

            corrections[size] =
                    2.0 * harmonic
                            - 2.0
                            * (size - 1.0)
                            / size;
        }

        return corrections;
    }

    /**
     * Advances the per-tree assignment stamp without clearing the array
     * during ordinary operation.
     */
    private static int nextStamp(
            int[] assignmentStamps,
            int currentStamp
    ) {
        if (currentStamp
                == Integer.MAX_VALUE) {

            java.util.Arrays.fill(
                    assignmentStamps,
                    0
            );

            return 1;
        }

        return currentStamp + 1;
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
                    "Cannot compute isolation scores: "
                            + "forest has no trees."
            );
        }
    }

    private static int stableInstanceIdentity(
            ListObjectDataset data,
            int localPosition
    ) {
        Integer identity =
                data.get_index(
                        localPosition
                );

        return identity == null
                ? localPosition
                : identity;
    }

    private static long mixSeed(
            int instanceIdentity,
            int treeIdentity
    ) {
        long value =
                0x9E3779B97F4A7C15L;

        value ^=
                0xBF58476D1CE4E5B9L
                        * (instanceIdentity + 1L);

        value ^=
                0x94D049BB133111EBL
                        * (treeIdentity + 1L);

        value =
                (value ^ (value >>> 30))
                        * 0xBF58476D1CE4E5B9L;

        value =
                (value ^ (value >>> 27))
                        * 0x94D049BB133111EBL;

        return value
                ^ (value >>> 31);
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
                - 2.0
                * (n - 1.0)
                / n;
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

    /**
     * Converts stable training identities to output-array positions.
     *
     * <p>When identities are canonical, no map is allocated and lookup is a
     * bounds-checked direct conversion.</p>
     */
    private static final class IdentityLookup {

        private final int trainingSize;

        private final Map<Integer, Integer> identityToPosition;

        private IdentityLookup(
                int trainingSize,
                Map<Integer, Integer> identityToPosition
        ) {
            this.trainingSize =
                    trainingSize;

            this.identityToPosition =
                    identityToPosition;
        }

        private int positionOf(
                int identity
        ) {
            if (identityToPosition == null) {
                return identity >= 0
                        && identity < trainingSize
                        ? identity
                        : -1;
            }

            Integer position =
                    identityToPosition.get(
                            identity
                    );

            return position == null
                    ? -1
                    : position;
        }
    }
}