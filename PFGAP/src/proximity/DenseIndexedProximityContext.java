package proximity;

import java.util.Collections;
import java.util.EnumSet;
import java.util.Set;

/**
 * Immutable proximity context optimized for dense scalar pair evaluation.
 *
 * <p>Training, test, and multiplicity lookups use flat instance-major arrays:
 * consecutive entries for one observation correspond to consecutive trees.
 * This layout gives pairwise formulas cache-friendly tree scans while avoiding
 * one Java array object per observation.</p>
 *
 * <p>Leaf-member lists remain tree-local CSR-like arrays because node counts
 * vary by tree and row-accumulation formulas enumerate members one reached leaf
 * at a time. Ancestry data is also tree-local.</p>
 *
 * <p>Only data requested through {@link ProximityDataRequirement} is retained.
 * This class is safe for concurrent reads after construction. Its
 * package-private constructor accepts ownership of builder-created arrays;
 * those arrays must not be mutated afterward.</p>
 */
public final class DenseIndexedProximityContext
        implements ProximityContext {

    private final int treeCount;
    private final int trainingSize;
    private final int testingSize;

    private final Set<ProximityDataRequirement> preparedRequirements;

    private final int[] trainInBagLeafNodeIds;
    private final int[] trainOutOfBagLeafNodeIds;
    private final int[] testLeafNodeIds;
    private final int[] bootstrapMultiplicities;
    private final long[] outOfBagBits;

    private final TreeData[] trees;

    /**
     * Creates a validated dense indexed context.
     *
     * <p>Flat instance-major arrays are indexed as:</p>
     *
     * <pre>
     * observationIndex * treeCount + treeIndex
     * </pre>
     */
    DenseIndexedProximityContext(
            int trainingSize,
            int testingSize,
            Set<ProximityDataRequirement> preparedRequirements,
            int[] trainInBagLeafNodeIds,
            int[] trainOutOfBagLeafNodeIds,
            int[] testLeafNodeIds,
            int[] bootstrapMultiplicities,
            long[] outOfBagBits,
            TreeData[] trees
    ) {
        if (trainingSize < 0 || testingSize < 0) {
            throw new IllegalArgumentException(
                    "Training and testing sizes cannot be negative."
            );
        }
        if (preparedRequirements == null) {
            throw new IllegalArgumentException(
                    "preparedRequirements cannot be null."
            );
        }
        /*if (preparedRequirements.contains(null)) {
            throw new IllegalArgumentException(
                    "preparedRequirements cannot contain null."
            );
        }*/
        for (ProximityDataRequirement requirement : preparedRequirements) {
            if (requirement == null) {
                throw new IllegalArgumentException(
                        "preparedRequirements cannot contain null."
                );
            }
        }
        if (trees == null) {
            throw new IllegalArgumentException("trees cannot be null.");
        }

        this.treeCount = trees.length;
        this.trainingSize = trainingSize;
        this.testingSize = testingSize;

        EnumSet<ProximityDataRequirement> copiedRequirements =
                preparedRequirements.isEmpty()
                        ? EnumSet.noneOf(ProximityDataRequirement.class)
                        : EnumSet.copyOf(preparedRequirements);

        this.preparedRequirements =
                Collections.unmodifiableSet(copiedRequirements);

        this.trainInBagLeafNodeIds = trainInBagLeafNodeIds;
        this.trainOutOfBagLeafNodeIds = trainOutOfBagLeafNodeIds;
        this.testLeafNodeIds = testLeafNodeIds;
        this.bootstrapMultiplicities = bootstrapMultiplicities;
        this.outOfBagBits = outOfBagBits;
        this.trees = trees.clone();

        validateFlatStorage(copiedRequirements);

        for (int treeIndex = 0; treeIndex < this.trees.length; treeIndex++) {
            TreeData tree = this.trees[treeIndex];
            if (tree == null) {
                throw new IllegalArgumentException(
                        "Tree data cannot be null at index " + treeIndex + "."
                );
            }
            tree.validate(
                    treeIndex,
                    trainingSize,
                    copiedRequirements
            );
        }
    }

    @Override
    public int treeCount() {
        return treeCount;
    }

    @Override
    public int trainingSize() {
        return trainingSize;
    }

    @Override
    public int testingSize() {
        return testingSize;
    }

    @Override
    public Set<ProximityDataRequirement> preparedRequirements() {
        return preparedRequirements;
    }

    @Override
    public int trainInBagLeafNodeId(
            int treeIndex,
            int trainIndex
    ) {
        require(ProximityDataRequirement.TRAIN_IN_BAG_LEAF_ASSIGNMENTS);
        return trainInBagLeafNodeIds[
                trainingOffset(trainIndex, treeIndex)
                ];
    }

    @Override
    public int trainOutOfBagLeafNodeId(
            int treeIndex,
            int trainIndex
    ) {
        require(ProximityDataRequirement.TRAIN_OUT_OF_BAG_LEAF_ASSIGNMENTS);
        return trainOutOfBagLeafNodeIds[
                trainingOffset(trainIndex, treeIndex)
                ];
    }

    @Override
    public int testLeafNodeId(
            int treeIndex,
            int testIndex
    ) {
        require(ProximityDataRequirement.TEST_LEAF_ASSIGNMENTS);
        return testLeafNodeIds[
                testingOffset(testIndex, treeIndex)
                ];
    }

    @Override
    public int bootstrapMultiplicity(
            int treeIndex,
            int trainIndex
    ) {
        require(ProximityDataRequirement.BOOTSTRAP_MULTIPLICITIES);
        return bootstrapMultiplicities[
                trainingOffset(trainIndex, treeIndex)
                ];
    }

    @Override
    public boolean isOutOfBag(
            int treeIndex,
            int trainIndex
    ) {
        int flatIndex =
                trainingOffset(trainIndex, treeIndex);

        if (hasRequirement(ProximityDataRequirement.BOOTSTRAP_MULTIPLICITIES)) {
            return bootstrapMultiplicities[flatIndex] == 0;
        }

        require(ProximityDataRequirement.OUT_OF_BAG_MEMBERSHIP);
        return bitIsSet(outOfBagBits, flatIndex);
    }

    @Override
    public int leafInBagDistinctMemberCount(
            int treeIndex,
            int leafNodeId
    ) {
        require(ProximityDataRequirement.LEAF_IN_BAG_DISTINCT_MEMBERS);
        TreeData tree = tree(treeIndex);
        checkNodeId(tree, treeIndex, leafNodeId);
        return memberCount(tree.inBagMemberOffsets, leafNodeId);
    }

    @Override
    public int leafInBagDistinctMemberAt(
            int treeIndex,
            int leafNodeId,
            int memberOffset
    ) {
        require(ProximityDataRequirement.LEAF_IN_BAG_DISTINCT_MEMBERS);
        TreeData tree = tree(treeIndex);
        checkNodeId(tree, treeIndex, leafNodeId);
        return memberAt(
                tree.inBagMemberOffsets,
                tree.inBagMembers,
                leafNodeId,
                memberOffset
        );
    }

    @Override
    public int leafAllTrainMemberCount(
            int treeIndex,
            int leafNodeId
    ) {
        require(ProximityDataRequirement.LEAF_ALL_TRAIN_MEMBERS);
        TreeData tree = tree(treeIndex);
        checkNodeId(tree, treeIndex, leafNodeId);
        return memberCount(tree.allTrainMemberOffsets, leafNodeId);
    }

    @Override
    public int leafAllTrainMemberAt(
            int treeIndex,
            int leafNodeId,
            int memberOffset
    ) {
        require(ProximityDataRequirement.LEAF_ALL_TRAIN_MEMBERS);
        TreeData tree = tree(treeIndex);
        checkNodeId(tree, treeIndex, leafNodeId);
        return memberAt(
                tree.allTrainMemberOffsets,
                tree.allTrainMembers,
                leafNodeId,
                memberOffset
        );
    }

    @Override
    public int leafInBagMultiplicityTotal(
            int treeIndex,
            int leafNodeId
    ) {
        require(ProximityDataRequirement.LEAF_IN_BAG_MULTIPLICITY_TOTALS);
        TreeData tree = tree(treeIndex);
        checkNodeId(tree, treeIndex, leafNodeId);
        return tree.leafInBagMultiplicityTotals[leafNodeId];
    }

    @Override
    public int parentNodeId(
            int treeIndex,
            int nodeId
    ) {
        require(ProximityDataRequirement.NODE_ANCESTRY);
        TreeData tree = tree(treeIndex);
        checkNodeId(tree, treeIndex, nodeId);
        return tree.parentNodeIds[nodeId];
    }

    @Override
    public int nodeDepth(
            int treeIndex,
            int nodeId
    ) {
        require(ProximityDataRequirement.NODE_ANCESTRY);
        TreeData tree = tree(treeIndex);
        checkNodeId(tree, treeIndex, nodeId);
        return tree.nodeDepths[nodeId];
    }

    /**
     * Returns the flat instance-major offset for a training observation.
     *
     * <p>Package-private for optimized built-in matrix code.</p>
     */
    int trainingOffset(
            int trainIndex
    ) {
        checkTrainIndex(trainIndex);
        return trainIndex * treeCount;
    }

    /**
     * Returns the flat instance-major offset for a test observation.
     *
     * <p>Package-private for optimized built-in matrix code.</p>
     */
    int testingOffset(
            int testIndex
    ) {
        checkTestIndex(testIndex);
        return testIndex * treeCount;
    }

    private int trainingOffset(
            int trainIndex,
            int treeIndex
    ) {
        checkTrainIndex(trainIndex);
        checkTreeIndex(treeIndex);
        return trainIndex * treeCount + treeIndex;
    }

    private int testingOffset(
            int testIndex,
            int treeIndex
    ) {
        checkTestIndex(testIndex);
        checkTreeIndex(treeIndex);
        return testIndex * treeCount + treeIndex;
    }

    private TreeData tree(int treeIndex) {
        checkTreeIndex(treeIndex);
        return trees[treeIndex];
    }

    private void checkTreeIndex(int treeIndex) {
        if (treeIndex < 0 || treeIndex >= treeCount) {
            throw new IndexOutOfBoundsException(
                    "treeIndex " + treeIndex + " is outside [0, "
                            + treeCount + ")."
            );
        }
    }

    private void checkTrainIndex(int trainIndex) {
        if (trainIndex < 0 || trainIndex >= trainingSize) {
            throw new IndexOutOfBoundsException(
                    "trainIndex " + trainIndex + " is outside [0, "
                            + trainingSize + ")."
            );
        }
    }

    private void checkTestIndex(int testIndex) {
        if (testIndex < 0 || testIndex >= testingSize) {
            throw new IndexOutOfBoundsException(
                    "testIndex " + testIndex + " is outside [0, "
                            + testingSize + ")."
            );
        }
    }

    private void validateFlatStorage(
            Set<ProximityDataRequirement> requirements
    ) {
        int expectedTrainingEntries =
                checkedEntryCount(
                        trainingSize,
                        treeCount,
                        "training"
                );

        int expectedTestingEntries =
                checkedEntryCount(
                        testingSize,
                        treeCount,
                        "testing"
                );

        requireLength(
                requirements,
                ProximityDataRequirement.TRAIN_IN_BAG_LEAF_ASSIGNMENTS,
                trainInBagLeafNodeIds,
                expectedTrainingEntries,
                "trainInBagLeafNodeIds"
        );

        requireLength(
                requirements,
                ProximityDataRequirement.TRAIN_OUT_OF_BAG_LEAF_ASSIGNMENTS,
                trainOutOfBagLeafNodeIds,
                expectedTrainingEntries,
                "trainOutOfBagLeafNodeIds"
        );

        requireLength(
                requirements,
                ProximityDataRequirement.TEST_LEAF_ASSIGNMENTS,
                testLeafNodeIds,
                expectedTestingEntries,
                "testLeafNodeIds"
        );

        requireLength(
                requirements,
                ProximityDataRequirement.BOOTSTRAP_MULTIPLICITIES,
                bootstrapMultiplicities,
                expectedTrainingEntries,
                "bootstrapMultiplicities"
        );

        if (requirements.contains(ProximityDataRequirement.OUT_OF_BAG_MEMBERSHIP)
                && !requirements.contains(
                ProximityDataRequirement.BOOTSTRAP_MULTIPLICITIES)) {

            int expectedWords =
                    (expectedTrainingEntries + 63) >>> 6;

            requireExactLength(
                    outOfBagBits,
                    expectedWords,
                    "outOfBagBits"
            );
        }
    }

    private static int checkedEntryCount(
            int instanceCount,
            int treeCount,
            String label
    ) {
        long entries =
                (long) instanceCount * treeCount;

        if (entries > Integer.MAX_VALUE) {
            throw new IllegalArgumentException(
                    "Dense " + label + " context requires " + entries
                            + " entries, exceeding the maximum Java array "
                            + "length. Use sparse context storage."
            );
        }

        return (int) entries;
    }

    private static void requireLength(
            Set<ProximityDataRequirement> requirements,
            ProximityDataRequirement requirement,
            int[] values,
            int expectedLength,
            String name
    ) {
        if (requirements.contains(requirement)) {
            requireExactLength(values, expectedLength, name);
        }
    }

    private static void requireExactLength(
            int[] values,
            int expectedLength,
            String name
    ) {
        if (values == null || values.length != expectedLength) {
            throw new IllegalArgumentException(
                    name + " must have length " + expectedLength + "."
            );
        }
    }

    private static void requireExactLength(
            long[] values,
            int expectedLength,
            String name
    ) {
        if (values == null || values.length != expectedLength) {
            throw new IllegalArgumentException(
                    name + " must have length " + expectedLength + "."
            );
        }
    }

    private static int memberCount(
            int[] offsets,
            int leafNodeId
    ) {
        return offsets[leafNodeId + 1] - offsets[leafNodeId];
    }

    private static int memberAt(
            int[] offsets,
            int[] members,
            int leafNodeId,
            int memberOffset
    ) {
        int start = offsets[leafNodeId];
        int count = offsets[leafNodeId + 1] - start;

        if (memberOffset < 0 || memberOffset >= count) {
            throw new IndexOutOfBoundsException(
                    "memberOffset " + memberOffset + " is outside [0, "
                            + count + ") for leaf node " + leafNodeId + "."
            );
        }

        return members[start + memberOffset];
    }

    private static boolean bitIsSet(
            long[] bits,
            int flatIndex
    ) {
        int word = flatIndex >>> 6;
        long mask = 1L << (flatIndex & 63);
        return (bits[word] & mask) != 0L;
    }

    private static void checkNodeId(
            TreeData tree,
            int treeIndex,
            int nodeId
    ) {
        if (nodeId < 0 || nodeId >= tree.nodeCount) {
            throw new IndexOutOfBoundsException(
                    "nodeId " + nodeId + " is outside [0, "
                            + tree.nodeCount + ") for tree " + treeIndex + "."
            );
        }
    }

    /**
     * Tree-local variable-length data retained by the dense context.
     */
    static final class TreeData {

        final int nodeCount;
        final int[] inBagMemberOffsets;
        final int[] inBagMembers;
        final int[] allTrainMemberOffsets;
        final int[] allTrainMembers;
        final int[] leafInBagMultiplicityTotals;
        final int[] parentNodeIds;
        final int[] nodeDepths;

        TreeData(
                int nodeCount,
                int[] inBagMemberOffsets,
                int[] inBagMembers,
                int[] allTrainMemberOffsets,
                int[] allTrainMembers,
                int[] leafInBagMultiplicityTotals,
                int[] parentNodeIds,
                int[] nodeDepths
        ) {
            if (nodeCount < 0) {
                throw new IllegalArgumentException(
                        "nodeCount cannot be negative."
                );
            }

            this.nodeCount = nodeCount;
            this.inBagMemberOffsets = inBagMemberOffsets;
            this.inBagMembers = inBagMembers;
            this.allTrainMemberOffsets = allTrainMemberOffsets;
            this.allTrainMembers = allTrainMembers;
            this.leafInBagMultiplicityTotals = leafInBagMultiplicityTotals;
            this.parentNodeIds = parentNodeIds;
            this.nodeDepths = nodeDepths;
        }

        private void validate(
                int treeIndex,
                int trainingSize,
                Set<ProximityDataRequirement> requirements
        ) {
            validateMembership(
                    requirements,
                    ProximityDataRequirement.LEAF_IN_BAG_DISTINCT_MEMBERS,
                    inBagMemberOffsets,
                    inBagMembers,
                    trainingSize,
                    "in-bag",
                    treeIndex
            );

            validateMembership(
                    requirements,
                    ProximityDataRequirement.LEAF_ALL_TRAIN_MEMBERS,
                    allTrainMemberOffsets,
                    allTrainMembers,
                    trainingSize,
                    "all-training",
                    treeIndex
            );

            if (requirements.contains(
                    ProximityDataRequirement.LEAF_IN_BAG_MULTIPLICITY_TOTALS)) {

                requireExactLength(
                        leafInBagMultiplicityTotals,
                        nodeCount,
                        "leafInBagMultiplicityTotals for tree " + treeIndex
                );
            }

            if (requirements.contains(ProximityDataRequirement.NODE_ANCESTRY)) {
                requireExactLength(
                        parentNodeIds,
                        nodeCount,
                        "parentNodeIds for tree " + treeIndex
                );

                requireExactLength(
                        nodeDepths,
                        nodeCount,
                        "nodeDepths for tree " + treeIndex
                );
            }
        }

        private void validateMembership(
                Set<ProximityDataRequirement> requirements,
                ProximityDataRequirement requirement,
                int[] offsets,
                int[] members,
                int trainingSize,
                String label,
                int treeIndex
        ) {
            if (!requirements.contains(requirement)) {
                return;
            }

            requireExactLength(
                    offsets,
                    nodeCount + 1,
                    label + " member offsets for tree " + treeIndex
            );

            if (members == null) {
                throw new IllegalArgumentException(
                        label + " members cannot be null for tree "
                                + treeIndex + "."
                );
            }

            if (offsets[0] != 0 || offsets[nodeCount] != members.length) {
                throw new IllegalArgumentException(
                        label + " offsets for tree " + treeIndex
                                + " must start at zero and end at member count."
                );
            }

            int prior = 0;
            for (int offset : offsets) {
                if (offset < prior || offset > members.length) {
                    throw new IllegalArgumentException(
                            label + " offsets are not monotonic for tree "
                                    + treeIndex + "."
                    );
                }
                prior = offset;
            }

            for (int member : members) {
                if (member < 0 || member >= trainingSize) {
                    throw new IllegalArgumentException(
                            label + " member index " + member
                                    + " is invalid for tree " + treeIndex + "."
                    );
                }
            }
        }
    }

    @Override
    public String toString() {
        return "DenseIndexedProximityContext{"
                + "treeCount=" + treeCount
                + ", trainingSize=" + trainingSize
                + ", testingSize=" + testingSize
                + ", requirements=" + preparedRequirements
                + '}';
    }
}
