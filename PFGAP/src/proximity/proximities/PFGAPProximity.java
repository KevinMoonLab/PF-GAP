package proximity.proximities;

import proximity.*;

import java.util.OptionalDouble;
import java.util.Set;

/**
 * Geometry- and accuracy-preserving random-forest proximity.
 *
 * <p>For a training source {@code i}, the implementation averages only over
 * trees in which {@code i} is out of bag. In each such tree, weight is
 * distributed over the distinct in-bag indices in the source's routed leaf in
 * proportion to bootstrap multiplicity:</p>
 *
 * <pre>
 * p(i, j) = (1 / |S_i|) * sum over t in S_i of
 *           c_j(t) * I(j in J_i(t)) / |M_i(t)|
 * </pre>
 *
 * <p>Here {@code c_j(t)} is the bootstrap multiplicity of target {@code j},
 * {@code J_i(t)} is the set of distinct in-bag indices in the source leaf,
 * and {@code |M_i(t)|} is that leaf's total in-bag multiplicity. The directed
 * train diagonal is zero under this prediction-weight definition.</p>
 *
 * <p>For a test source, every tree contributes because the source is external
 * to every tree's bootstrap sample. Both train/train and test/train rows can be
 * accumulated by visiting only the in-bag members of reached leaves, avoiding
 * a scan over every possible target column.</p>
 *
 * <p>This class is stateless and safe for concurrent use.</p>
 */
public final class PFGAPProximity
        implements ProximityMeasure {

    /** Shared stateless implementation instance. */
    public static final PFGAPProximity INSTANCE =
            new PFGAPProximity();

    private static final ProximityProperties PROPERTIES =
            new ProximityProperties(
                    false,
                    OptionalDouble.of(0.0)
            );

    private static final Set<ProximityDataRequirement> TRAIN_REQUIREMENTS =
            Set.of(
                    ProximityDataRequirement
                            .TRAIN_OUT_OF_BAG_LEAF_ASSIGNMENTS,
                    ProximityDataRequirement
                            .LEAF_IN_BAG_DISTINCT_MEMBERS,
                    ProximityDataRequirement
                            .LEAF_IN_BAG_MULTIPLICITY_TOTALS,
                    ProximityDataRequirement
                            .BOOTSTRAP_MULTIPLICITIES,
                    ProximityDataRequirement
                            .OUT_OF_BAG_MEMBERSHIP
            );

    private static final Set<ProximityDataRequirement> TEST_TRAIN_REQUIREMENTS =
            Set.of(
                    ProximityDataRequirement
                            .TEST_LEAF_ASSIGNMENTS,
                    ProximityDataRequirement
                            .LEAF_IN_BAG_DISTINCT_MEMBERS,
                    ProximityDataRequirement
                            .LEAF_IN_BAG_MULTIPLICITY_TOTALS,
                    ProximityDataRequirement
                            .BOOTSTRAP_MULTIPLICITIES
            );

    private PFGAPProximity() {
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
            return 0.0;
        }

        double sum =
                0.0;

        int contributingTreeCount =
                0;

        for (int treeIndex = 0;
             treeIndex < context.treeCount();
             treeIndex++) {

            if (!context.isOutOfBag(
                    treeIndex,
                    sourceTrainIndex
            )) {
                continue;
            }

            contributingTreeCount++;

            int sourceLeafNodeId =
                    context.trainOutOfBagLeafNodeId(
                            treeIndex,
                            sourceTrainIndex
                    );

            if (sourceLeafNodeId == ProximityContext.NO_NODE) {
                continue;
            }

            int targetMultiplicity =
                    context.bootstrapMultiplicity(
                            treeIndex,
                            targetTrainIndex
                    );

            if (targetMultiplicity == 0) {
                continue;
            }

            if (!leafContainsTarget(
                    context,
                    treeIndex,
                    sourceLeafNodeId,
                    targetTrainIndex
            )) {
                continue;
            }

            int leafMultiplicityTotal =
                    requirePositiveLeafMultiplicityTotal(
                            context,
                            treeIndex,
                            sourceLeafNodeId
                    );

            sum +=
                    ((double) targetMultiplicity)
                            / leafMultiplicityTotal;
        }

        return contributingTreeCount == 0
                ? 0.0
                : sum / contributingTreeCount;
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

            int sourceLeafNodeId =
                    context.testLeafNodeId(
                            treeIndex,
                            testIndex
                    );

            if (sourceLeafNodeId == ProximityContext.NO_NODE) {
                continue;
            }

            int targetMultiplicity =
                    context.bootstrapMultiplicity(
                            treeIndex,
                            trainIndex
                    );

            if (targetMultiplicity == 0) {
                continue;
            }

            if (!leafContainsTarget(
                    context,
                    treeIndex,
                    sourceLeafNodeId,
                    trainIndex
            )) {
                continue;
            }

            int leafMultiplicityTotal =
                    requirePositiveLeafMultiplicityTotal(
                            context,
                            treeIndex,
                            sourceLeafNodeId
                    );

            sum +=
                    ((double) targetMultiplicity)
                            / leafMultiplicityTotal;
        }

        return sum / treeCount;
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

        int contributingTreeCount =
                0;

        for (int treeIndex = 0;
             treeIndex < context.treeCount();
             treeIndex++) {

            if (!context.isOutOfBag(
                    treeIndex,
                    sourceTrainIndex
            )) {
                continue;
            }

            contributingTreeCount++;

            int sourceLeafNodeId =
                    context.trainOutOfBagLeafNodeId(
                            treeIndex,
                            sourceTrainIndex
                    );

            if (sourceLeafNodeId == ProximityContext.NO_NODE) {
                continue;
            }

            accumulateLeafContributions(
                    context,
                    treeIndex,
                    sourceLeafNodeId,
                    accumulator
            );
        }

        if (contributingTreeCount > 0) {
            accumulator.scale(
                    1.0 / contributingTreeCount
            );
        }

        /*
         * The prediction-weight RF-GAP definition requires p(i, i) = 0.
         * Enforce it explicitly even if inconsistent context data contained
         * the source among an OOB leaf's in-bag members.
         */
        accumulator.set(
                sourceTrainIndex,
                0.0
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

            int sourceLeafNodeId =
                    context.testLeafNodeId(
                            treeIndex,
                            testIndex
                    );

            if (sourceLeafNodeId == ProximityContext.NO_NODE) {
                continue;
            }

            accumulateLeafContributions(
                    context,
                    treeIndex,
                    sourceLeafNodeId,
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
        return "PFGAP";
    }

    private static void accumulateLeafContributions(
            ProximityContext context,
            int treeIndex,
            int leafNodeId,
            ProximityRowAccumulator accumulator
    ) {
        int leafMultiplicityTotal =
                requirePositiveLeafMultiplicityTotal(
                        context,
                        treeIndex,
                        leafNodeId
                );

        int memberCount =
                context.leafInBagDistinctMemberCount(
                        treeIndex,
                        leafNodeId
                );

        for (int memberOffset = 0;
             memberOffset < memberCount;
             memberOffset++) {

            int targetTrainIndex =
                    context.leafInBagDistinctMemberAt(
                            treeIndex,
                            leafNodeId,
                            memberOffset
                    );

            int targetMultiplicity =
                    context.bootstrapMultiplicity(
                            treeIndex,
                            targetTrainIndex
                    );

            if (targetMultiplicity <= 0) {
                throw new IllegalStateException(
                        "Leaf "
                                + leafNodeId
                                + " in tree "
                                + treeIndex
                                + " contains training index "
                                + targetTrainIndex
                                + " with non-positive bootstrap multiplicity "
                                + targetMultiplicity
                                + "."
                );
            }

            accumulator.add(
                    targetTrainIndex,
                    ((double) targetMultiplicity)
                            / leafMultiplicityTotal
            );
        }
    }

    private static boolean leafContainsTarget(
            ProximityContext context,
            int treeIndex,
            int leafNodeId,
            int targetTrainIndex
    ) {
        int memberCount =
                context.leafInBagDistinctMemberCount(
                        treeIndex,
                        leafNodeId
                );

        for (int memberOffset = 0;
             memberOffset < memberCount;
             memberOffset++) {

            if (context.leafInBagDistinctMemberAt(
                    treeIndex,
                    leafNodeId,
                    memberOffset
            ) == targetTrainIndex) {
                return true;
            }
        }

        return false;
    }

    private static int requirePositiveLeafMultiplicityTotal(
            ProximityContext context,
            int treeIndex,
            int leafNodeId
    ) {
        int total =
                context.leafInBagMultiplicityTotal(
                        treeIndex,
                        leafNodeId
                );

        if (total <= 0) {
            throw new IllegalStateException(
                    "Leaf "
                            + leafNodeId
                            + " in tree "
                            + treeIndex
                            + " has non-positive total in-bag multiplicity "
                            + total
                            + "."
            );
        }

        return total;
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
