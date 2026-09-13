package proximity.proximities;

import proximity.*;

import java.util.OptionalDouble;
import java.util.Set;

/**
 * Breiman's original random-forest proximity.
 *
 * <p>The proximity between two observations is the proportion of trees in
 * which they reach the same terminal node, regardless of bootstrap status:</p>
 *
 * <pre>
 * p(i, j) = (1 / T) * sum over trees t of I(v_i(t) = v_j(t))
 * </pre>
 *
 * <p>Train/train proximity is symmetric and has constant self-proximity one,
 * provided that every training observation has a routed terminal-node
 * assignment in every tree.</p>
 *
 * <p>The scalar methods use direct prepared leaf identifiers. The row methods
 * visit only the training observations routed to the source leaf in each tree,
 * permitting exact sparse accumulation without scanning every target column.</p>
 *
 * <p>This implementation is stateless and safe for concurrent invocation.
 * Parallel row scheduling belongs to the proximity-matrix computation layer.</p>
 */
public final class BreimanProximity
        implements ProximityMeasure {

    /** Shared stateless implementation instance. */
    public static final BreimanProximity INSTANCE =
            new BreimanProximity();

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
                            .LEAF_ALL_TRAIN_MEMBERS
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
                            .LEAF_ALL_TRAIN_MEMBERS
            );

    private BreimanProximity() {
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

        int matchingTreeCount =
                0;

        for (int treeIndex = 0;
             treeIndex < treeCount;
             treeIndex++) {

            int sourceLeafNodeId =
                    context.trainLeafNodeId(
                            treeIndex,
                            sourceTrainIndex
                    );

            int targetLeafNodeId =
                    context.trainLeafNodeId(
                            treeIndex,
                            targetTrainIndex
                    );

            if (sourceLeafNodeId != ProximityContext.NO_NODE
                    && sourceLeafNodeId == targetLeafNodeId) {

                matchingTreeCount++;
            }
        }

        return ((double) matchingTreeCount)
                / treeCount;
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

        int matchingTreeCount =
                0;

        for (int treeIndex = 0;
             treeIndex < treeCount;
             treeIndex++) {

            int testLeafNodeId =
                    context.testLeafNodeId(
                            treeIndex,
                            testIndex
                    );

            int trainLeafNodeId =
                    context.trainLeafNodeId(
                            treeIndex,
                            trainIndex
                    );

            if (testLeafNodeId != ProximityContext.NO_NODE
                    && testLeafNodeId == trainLeafNodeId) {

                matchingTreeCount++;
            }
        }

        return ((double) matchingTreeCount)
                / treeCount;
    }

    @Override
    public boolean supportsTrainRowAccumulation() {
        return true;
    }

    @Override
    public void accumulateTrainRow(
            int sourceTrainIndex,
            ProximityContext context,
            ProximityRowAccumulator accumulator
    ) {
        requireTrainContext(context);
        requireAccumulator(context, accumulator);
        checkTrainIndex(context, sourceTrainIndex, "sourceTrainIndex");

        int treeCount =
                context.treeCount();

        if (treeCount == 0) {
            return;
        }

        for (int treeIndex = 0;
             treeIndex < treeCount;
             treeIndex++) {

            int sourceLeafNodeId =
                    context.trainLeafNodeId(
                            treeIndex,
                            sourceTrainIndex
                    );

            if (sourceLeafNodeId == ProximityContext.NO_NODE) {
                continue;
            }

            accumulateLeafMembers(
                    context,
                    treeIndex,
                    sourceLeafNodeId,
                    accumulator
            );
        }

        accumulator.scale(
                1.0 / treeCount
        );

        accumulator.set(
                sourceTrainIndex,
                1.0
        );
    }

    @Override
    public boolean supportsTestTrainRowAccumulation() {
        return true;
    }

    @Override
    public void accumulateTestTrainRow(
            int testIndex,
            ProximityContext context,
            ProximityRowAccumulator accumulator
    ) {
        requireTestTrainContext(context);
        requireAccumulator(context, accumulator);
        checkTestIndex(context, testIndex);

        int treeCount =
                context.treeCount();

        if (treeCount == 0) {
            return;
        }

        for (int treeIndex = 0;
             treeIndex < treeCount;
             treeIndex++) {

            int testLeafNodeId =
                    context.testLeafNodeId(
                            treeIndex,
                            testIndex
                    );

            if (testLeafNodeId == ProximityContext.NO_NODE) {
                continue;
            }

            accumulateLeafMembers(
                    context,
                    treeIndex,
                    testLeafNodeId,
                    accumulator
            );
        }

        accumulator.scale(
                1.0 / treeCount
        );
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
        return "BREIMAN";
    }

    private static void accumulateLeafMembers(
            ProximityContext context,
            int treeIndex,
            int leafNodeId,
            ProximityRowAccumulator accumulator
    ) {
        int memberCount =
                context.leafAllTrainMemberCount(
                        treeIndex,
                        leafNodeId
                );

        for (int memberOffset = 0;
             memberOffset < memberCount;
             memberOffset++) {

            int targetTrainIndex =
                    context.leafAllTrainMemberAt(
                            treeIndex,
                            leafNodeId,
                            memberOffset
                    );

            if (targetTrainIndex < 0
                    || targetTrainIndex >= context.trainingSize()) {

                throw new IllegalStateException(
                        "Leaf "
                                + leafNodeId
                                + " in tree "
                                + treeIndex
                                + " contains invalid training index "
                                + targetTrainIndex
                                + "."
                );
            }

            accumulator.add(
                    targetTrainIndex,
                    1.0
            );
        }
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

    private static void requireAccumulator(
            ProximityContext context,
            ProximityRowAccumulator accumulator
    ) {
        if (accumulator == null) {
            throw new IllegalArgumentException(
                    "ProximityRowAccumulator cannot be null."
            );
        }

        if (accumulator.targetCount() != context.trainingSize()) {
            throw new IllegalArgumentException(
                    "Accumulator target count "
                            + accumulator.targetCount()
                            + " does not match training size "
                            + context.trainingSize()
                            + "."
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
