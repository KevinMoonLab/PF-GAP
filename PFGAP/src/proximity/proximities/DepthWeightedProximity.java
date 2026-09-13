package proximity.proximities;

import proximity.ProximityContext;
import proximity.ProximityDataRequirement;
import proximity.ProximityMeasure;
import proximity.ProximityProperties;

import java.util.OptionalDouble;
import java.util.Set;

/**
 * Proximity based on the depth of the lowest common ancestor of two routed
 * observations.
 *
 * <p>For one tree, let {@code a} and {@code b} be the terminal nodes reached
 * by the two observations. With root depth zero, the tree contribution is:</p>
 *
 * <pre>
 * (lowestCommonAncestorDepth(a, b) + 1)
 * -------------------------------------------------
 * max(depth(a), depth(b)) + 1
 * </pre>
 *
 * <p>This preserves the behavior of the earlier path-prefix implementation:
 * observations in the same terminal node receive contribution one, deeper
 * shared ancestry receives greater similarity, and observations sharing only
 * the root receive a small positive contribution.</p>
 *
 * <p>The train/train measure is symmetric and has constant self-proximity one,
 * provided that the context records the routed terminal node of every training
 * observation in every tree. The implementation is stateless and safe for
 * concurrent invocation. Parallel scheduling belongs to the matrix layer.</p>
 */
public final class DepthWeightedProximity
        implements ProximityMeasure {

    /** Shared stateless implementation instance. */
    public static final DepthWeightedProximity INSTANCE =
            new DepthWeightedProximity();

    private static final ProximityProperties PROPERTIES =
            new ProximityProperties(
                    true,
                    OptionalDouble.of(1.0)
            );

    private static final Set<ProximityDataRequirement> TRAIN_REQUIREMENTS =
            Set.of(
                    ProximityDataRequirement
                            .TRAIN_IN_BAG_LEAF_ASSIGNMENTS,
                    ProximityDataRequirement
                            .TRAIN_OUT_OF_BAG_LEAF_ASSIGNMENTS,
                    ProximityDataRequirement
                            .NODE_ANCESTRY
            );

    private static final Set<ProximityDataRequirement> TEST_TRAIN_REQUIREMENTS =
            Set.of(
                    ProximityDataRequirement
                            .TRAIN_IN_BAG_LEAF_ASSIGNMENTS,
                    ProximityDataRequirement
                            .TRAIN_OUT_OF_BAG_LEAF_ASSIGNMENTS,
                    ProximityDataRequirement
                            .TEST_LEAF_ASSIGNMENTS,
                    ProximityDataRequirement
                            .NODE_ANCESTRY
            );

    private DepthWeightedProximity() {
    }

    @Override
    public double computeTrainTrain(
            int sourceTrainIndex,
            int targetTrainIndex,
            ProximityContext context
    ) {
        requireTrainContext(context);
        checkTrainIndex(context, sourceTrainIndex, "sourceTrainIndex");
        checkTrainIndex(context, targetTrainIndex, "targetTrainIndex");

        if (sourceTrainIndex == targetTrainIndex) {
            return 1.0;
        }

        int treeCount =
                context.treeCount();

        if (treeCount == 0) {
            return 0.0;
        }

        double sum =
                0.0;

        for (int treeIndex = 0;
             treeIndex < treeCount;
             treeIndex++) {

            int sourceLeafNodeId =
                    trainLeafNodeId(
                            context,
                            treeIndex,
                            sourceTrainIndex
                    );

            int targetLeafNodeId =
                    trainLeafNodeId(
                            context,
                            treeIndex,
                            targetTrainIndex
                    );

            sum += treeSimilarity(
                    context,
                    treeIndex,
                    sourceLeafNodeId,
                    targetLeafNodeId
            );
        }

        return sum / treeCount;
    }

    @Override
    public double computeTestTrain(
            int testIndex,
            int trainIndex,
            ProximityContext context
    ) {
        requireTestTrainContext(context);
        checkTestIndex(context, testIndex);
        checkTrainIndex(context, trainIndex, "trainIndex");

        int treeCount =
                context.treeCount();

        if (treeCount == 0) {
            return 0.0;
        }

        double sum =
                0.0;

        for (int treeIndex = 0;
             treeIndex < treeCount;
             treeIndex++) {

            int testLeafNodeId =
                    context.testLeafNodeId(
                            treeIndex,
                            testIndex
                    );

            int trainLeafNodeId =
                    trainLeafNodeId(
                            context,
                            treeIndex,
                            trainIndex
                    );

            sum += treeSimilarity(
                    context,
                    treeIndex,
                    testLeafNodeId,
                    trainLeafNodeId
            );
        }

        return sum / treeCount;
    }

    @Override
    public ProximityProperties properties() {
        return PROPERTIES;
    }

    @Override
    public Set<ProximityDataRequirement> trainRequirements() {
        return TRAIN_REQUIREMENTS;
    }

    @Override
    public Set<ProximityDataRequirement> testTrainRequirements() {
        return TEST_TRAIN_REQUIREMENTS;
    }

    @Override
    public String id() {
        return "DEPTH_WEIGHTED";
    }

    /**
     * Computes one tree's normalized shared-ancestry contribution without
     * allocating root-to-node paths.
     */
    private static double treeSimilarity(
            ProximityContext context,
            int treeIndex,
            int firstLeafNodeId,
            int secondLeafNodeId
    ) {
        if (firstLeafNodeId == ProximityContext.NO_NODE
                || secondLeafNodeId == ProximityContext.NO_NODE) {

            return 0.0;
        }

        int firstDepth =
                context.nodeDepth(
                        treeIndex,
                        firstLeafNodeId
                );

        int secondDepth =
                context.nodeDepth(
                        treeIndex,
                        secondLeafNodeId
                );

        int commonAncestorDepth =
                context.commonAncestorDepth(
                        treeIndex,
                        firstLeafNodeId,
                        secondLeafNodeId
                );

        if (commonAncestorDepth < 0) {
            return 0.0;
        }

        int maximumPathLength =
                Math.max(
                        firstDepth,
                        secondDepth
                ) + 1;

        return ((double) commonAncestorDepth + 1.0)
                / maximumPathLength;
    }

    /**
     * Returns a training observation's routed leaf regardless of bootstrap
     * status.
     */
    private static int trainLeafNodeId(
            ProximityContext context,
            int treeIndex,
            int trainIndex
    ) {
        int inBagLeafNodeId =
                context.trainInBagLeafNodeId(
                        treeIndex,
                        trainIndex
                );

        if (inBagLeafNodeId != ProximityContext.NO_NODE) {
            return inBagLeafNodeId;
        }

        return context.trainOutOfBagLeafNodeId(
                treeIndex,
                trainIndex
        );
    }

    private static void requireTrainContext(
            ProximityContext context
    ) {
        requireContext(context);

        for (ProximityDataRequirement requirement : TRAIN_REQUIREMENTS) {
            context.require(requirement);
        }
    }

    private static void requireTestTrainContext(
            ProximityContext context
    ) {
        requireContext(context);

        for (ProximityDataRequirement requirement
                : TEST_TRAIN_REQUIREMENTS) {

            context.require(requirement);
        }
    }

    private static void requireContext(
            ProximityContext context
    ) {
        if (context == null) {
            throw new IllegalArgumentException(
                    "ProximityContext cannot be null."
            );
        }
    }

    private static void checkTrainIndex(
            ProximityContext context,
            int trainIndex,
            String argumentName
    ) {
        if (trainIndex < 0 || trainIndex >= context.trainingSize()) {
            throw new IndexOutOfBoundsException(
                    argumentName
                            + " "
                            + trainIndex
                            + " is outside [0, "
                            + context.trainingSize()
                            + ")."
            );
        }
    }

    private static void checkTestIndex(
            ProximityContext context,
            int testIndex
    ) {
        if (testIndex < 0 || testIndex >= context.testingSize()) {
            throw new IndexOutOfBoundsException(
                    "testIndex "
                            + testIndex
                            + " is outside [0, "
                            + context.testingSize()
                            + ")."
            );
        }
    }
}
