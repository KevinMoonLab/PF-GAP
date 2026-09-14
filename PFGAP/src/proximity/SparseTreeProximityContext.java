package proximity;

import java.util.Collections;
import java.util.EnumSet;
import java.util.Set;

/**
 * Immutable tree-local proximity context optimized for sparse accumulation.
 *
 * <p>Each tree owns compact primitive arrays for only the data requested by
 * the selected proximity measure and calculation domain. Leaf memberships use
 * CSR-like node offsets, so formulas can enumerate one leaf without boxed
 * collections or an {@code N x N} structure.</p>
 *
 * <p>This context is safe for concurrent reads after construction. Its
 * package-private constructor accepts ownership of builder-produced arrays;
 * callers must not mutate those arrays after construction.</p>
 */
public final class SparseTreeProximityContext
        implements ProximityContext {

    private final int trainingSize;
    private final int testingSize;
    private final Set<ProximityDataRequirement> preparedRequirements;
    private final TreeData[] trees;

    /**
     * Creates a context from validated tree-local data.
     *
     * <p>This constructor is package-private so preparation remains centralized
     * in the forthcoming context builder.</p>
     */
    SparseTreeProximityContext(
            int trainingSize,
            int testingSize,
            Set<ProximityDataRequirement> preparedRequirements,
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

        this.trainingSize = trainingSize;
        this.testingSize = testingSize;

        EnumSet<ProximityDataRequirement> copiedRequirements =
                preparedRequirements.isEmpty()
                        ? EnumSet.noneOf(ProximityDataRequirement.class)
                        : EnumSet.copyOf(preparedRequirements);

        this.preparedRequirements =
                Collections.unmodifiableSet(copiedRequirements);

        this.trees = trees.clone();

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
                    testingSize,
                    copiedRequirements
            );
        }
    }

    @Override
    public int treeCount() {
        return trees.length;
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
    public int trainInBagLeafNodeId(int treeIndex, int trainIndex) {
        require(ProximityDataRequirement.TRAIN_IN_BAG_LEAF_ASSIGNMENTS);
        checkTrainIndex(trainIndex);
        return tree(treeIndex).trainInBagLeafNodeIds[trainIndex];
    }

    @Override
    public int trainOutOfBagLeafNodeId(int treeIndex, int trainIndex) {
        require(ProximityDataRequirement.TRAIN_OUT_OF_BAG_LEAF_ASSIGNMENTS);
        checkTrainIndex(trainIndex);
        return tree(treeIndex).trainOutOfBagLeafNodeIds[trainIndex];
    }

    @Override
    public int testLeafNodeId(int treeIndex, int testIndex) {
        require(ProximityDataRequirement.TEST_LEAF_ASSIGNMENTS);
        checkTestIndex(testIndex);
        return tree(treeIndex).testLeafNodeIds[testIndex];
    }

    @Override
    public int bootstrapMultiplicity(int treeIndex, int trainIndex) {
        require(ProximityDataRequirement.BOOTSTRAP_MULTIPLICITIES);
        checkTrainIndex(trainIndex);
        return tree(treeIndex).bootstrapMultiplicities[trainIndex];
    }

    @Override
    public boolean isOutOfBag(int treeIndex, int trainIndex) {
        checkTrainIndex(trainIndex);
        TreeData tree = tree(treeIndex);

        if (hasRequirement(ProximityDataRequirement.BOOTSTRAP_MULTIPLICITIES)) {
            return tree.bootstrapMultiplicities[trainIndex] == 0;
        }

        require(ProximityDataRequirement.OUT_OF_BAG_MEMBERSHIP);
        return bitIsSet(tree.outOfBagBits, trainIndex);
    }

    @Override
    public int leafInBagDistinctMemberCount(int treeIndex, int leafNodeId) {
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
    public int leafAllTrainMemberCount(int treeIndex, int leafNodeId) {
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
    public int leafInBagMultiplicityTotal(int treeIndex, int leafNodeId) {
        require(ProximityDataRequirement.LEAF_IN_BAG_MULTIPLICITY_TOTALS);
        TreeData tree = tree(treeIndex);
        checkNodeId(tree, treeIndex, leafNodeId);
        return tree.leafInBagMultiplicityTotals[leafNodeId];
    }

    @Override
    public int parentNodeId(int treeIndex, int nodeId) {
        require(ProximityDataRequirement.NODE_ANCESTRY);
        TreeData tree = tree(treeIndex);
        checkNodeId(tree, treeIndex, nodeId);
        return tree.parentNodeIds[nodeId];
    }

    @Override
    public int nodeDepth(int treeIndex, int nodeId) {
        require(ProximityDataRequirement.NODE_ANCESTRY);
        TreeData tree = tree(treeIndex);
        checkNodeId(tree, treeIndex, nodeId);
        return tree.nodeDepths[nodeId];
    }

    private TreeData tree(int treeIndex) {
        if (treeIndex < 0 || treeIndex >= trees.length) {
            throw new IndexOutOfBoundsException(
                    "treeIndex " + treeIndex + " is outside [0, "
                            + trees.length + ")."
            );
        }
        return trees[treeIndex];
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

    private static int memberCount(int[] offsets, int leafNodeId) {
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

    private static boolean bitIsSet(long[] bits, int index) {
        int word = index >>> 6;
        long mask = 1L << (index & 63);
        return (bits[word] & mask) != 0L;
    }

    /**
     * Immutable-by-ownership primitive data for one tree.
     *
     * <p>Arrays are package-private so the context builder can construct them
     * without copying. Arrays corresponding to unrequested requirements are
     * null. Node-indexed membership offsets have length {@code nodeCount + 1};
     * non-leaf nodes have empty ranges.</p>
     */
    static final class TreeData {

        final int nodeCount;
        final int[] trainInBagLeafNodeIds;
        final int[] trainOutOfBagLeafNodeIds;
        final int[] testLeafNodeIds;
        final int[] bootstrapMultiplicities;
        final long[] outOfBagBits;
        final int[] inBagMemberOffsets;
        final int[] inBagMembers;
        final int[] allTrainMemberOffsets;
        final int[] allTrainMembers;
        final int[] leafInBagMultiplicityTotals;
        final int[] parentNodeIds;
        final int[] nodeDepths;

        TreeData(
                int nodeCount,
                int[] trainInBagLeafNodeIds,
                int[] trainOutOfBagLeafNodeIds,
                int[] testLeafNodeIds,
                int[] bootstrapMultiplicities,
                long[] outOfBagBits,
                int[] inBagMemberOffsets,
                int[] inBagMembers,
                int[] allTrainMemberOffsets,
                int[] allTrainMembers,
                int[] leafInBagMultiplicityTotals,
                int[] parentNodeIds,
                int[] nodeDepths
        ) {
            if (nodeCount < 0) {
                throw new IllegalArgumentException("nodeCount cannot be negative.");
            }
            this.nodeCount = nodeCount;
            this.trainInBagLeafNodeIds = trainInBagLeafNodeIds;
            this.trainOutOfBagLeafNodeIds = trainOutOfBagLeafNodeIds;
            this.testLeafNodeIds = testLeafNodeIds;
            this.bootstrapMultiplicities = bootstrapMultiplicities;
            this.outOfBagBits = outOfBagBits;
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
                int testingSize,
                Set<ProximityDataRequirement> requirements
        ) {
            requireLength(
                    requirements,
                    ProximityDataRequirement.TRAIN_IN_BAG_LEAF_ASSIGNMENTS,
                    trainInBagLeafNodeIds,
                    trainingSize,
                    "trainInBagLeafNodeIds",
                    treeIndex
            );
            requireLength(
                    requirements,
                    ProximityDataRequirement.TRAIN_OUT_OF_BAG_LEAF_ASSIGNMENTS,
                    trainOutOfBagLeafNodeIds,
                    trainingSize,
                    "trainOutOfBagLeafNodeIds",
                    treeIndex
            );
            requireLength(
                    requirements,
                    ProximityDataRequirement.TEST_LEAF_ASSIGNMENTS,
                    testLeafNodeIds,
                    testingSize,
                    "testLeafNodeIds",
                    treeIndex
            );
            requireLength(
                    requirements,
                    ProximityDataRequirement.BOOTSTRAP_MULTIPLICITIES,
                    bootstrapMultiplicities,
                    trainingSize,
                    "bootstrapMultiplicities",
                    treeIndex
            );

            if (requirements.contains(ProximityDataRequirement.OUT_OF_BAG_MEMBERSHIP)
                    && !requirements.contains(
                    ProximityDataRequirement.BOOTSTRAP_MULTIPLICITIES)) {

                int expectedWords = (trainingSize + 63) >>> 6;
                requireExactLength(
                        outOfBagBits,
                        expectedWords,
                        "outOfBagBits",
                        treeIndex
                );
            }

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

            requireLength(
                    requirements,
                    ProximityDataRequirement.LEAF_IN_BAG_MULTIPLICITY_TOTALS,
                    leafInBagMultiplicityTotals,
                    nodeCount,
                    "leafInBagMultiplicityTotals",
                    treeIndex
            );

            if (requirements.contains(ProximityDataRequirement.NODE_ANCESTRY)) {
                requireExactLength(parentNodeIds, nodeCount, "parentNodeIds", treeIndex);
                requireExactLength(nodeDepths, nodeCount, "nodeDepths", treeIndex);
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
                    label + " member offsets",
                    treeIndex
            );

            if (members == null) {
                throw invalid(treeIndex, label + " members cannot be null.");
            }
            if (offsets[0] != 0 || offsets[nodeCount] != members.length) {
                throw invalid(
                        treeIndex,
                        label + " offsets must start at zero and end at member count."
                );
            }

            int prior = 0;
            for (int offset : offsets) {
                if (offset < prior || offset > members.length) {
                    throw invalid(treeIndex, label + " offsets are not monotonic.");
                }
                prior = offset;
            }

            for (int member : members) {
                if (member < 0 || member >= trainingSize) {
                    throw invalid(
                            treeIndex,
                            label + " member index " + member + " is invalid."
                    );
                }
            }
        }

        private static void requireLength(
                Set<ProximityDataRequirement> requirements,
                ProximityDataRequirement requirement,
                int[] values,
                int expectedLength,
                String name,
                int treeIndex
        ) {
            if (requirements.contains(requirement)) {
                requireExactLength(values, expectedLength, name, treeIndex);
            }
        }

        private static void requireExactLength(
                int[] values,
                int expectedLength,
                String name,
                int treeIndex
        ) {
            if (values == null || values.length != expectedLength) {
                throw invalid(
                        treeIndex,
                        name + " must have length " + expectedLength + "."
                );
            }
        }

        private static void requireExactLength(
                long[] values,
                int expectedLength,
                String name,
                int treeIndex
        ) {
            if (values == null || values.length != expectedLength) {
                throw invalid(
                        treeIndex,
                        name + " must have length " + expectedLength + "."
                );
            }
        }

        private static IllegalArgumentException invalid(
                int treeIndex,
                String message
        ) {
            return new IllegalArgumentException(
                    "Invalid sparse context data for tree "
                            + treeIndex + ": " + message
            );
        }

        @Override
        public String toString() {
            return "TreeData{"
                    + "nodeCount=" + nodeCount
                    + ", inBagAssignments=" + lengthOf(trainInBagLeafNodeIds)
                    + ", oobAssignments=" + lengthOf(trainOutOfBagLeafNodeIds)
                    + ", testAssignments=" + lengthOf(testLeafNodeIds)
                    + ", inBagMembers=" + lengthOf(inBagMembers)
                    + ", allTrainMembers=" + lengthOf(allTrainMembers)
                    + '}';
        }

        private static int lengthOf(int[] values) {
            return values == null ? 0 : values.length;
        }
    }

    @Override
    public String toString() {
        return "SparseTreeProximityContext{"
                + "treeCount=" + trees.length
                + ", trainingSize=" + trainingSize
                + ", testingSize=" + testingSize
                + ", requirements=" + preparedRequirements
                + '}';
    }
}
