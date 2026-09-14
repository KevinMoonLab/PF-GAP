package proximity;

import core.parallel.ParallelRuntime;
import trees.ProximityForest;
import trees.ProximityTree;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.EnumSet;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * Builds immutable proximity contexts from a trained and evaluated forest.
 *
 * <p>Preparation is parallelized first by tree through the caller-owned
 * {@link ParallelRuntime}. Every tree writes to an independent result slot or
 * to a distinct column of a preallocated dense index, so no locks are needed.
 * The context is published only after every tree task has completed.</p>
 *
 * <p>The builder prepares only the requirements declared for the requested
 * calculation domain. Sparse contexts retain tree-local assignment arrays and
 * CSR-like leaf memberships. Dense contexts write fixed-size assignment and
 * multiplicity data directly into flat instance-major arrays, avoiding a full
 * tree-major-to-instance-major transpose.</p>
 *
 * <p>Training and test indices stored by tree nodes are assumed to be
 * zero-based positions in the corresponding matrix domain. The builder
 * validates bounds and rejects conflicting terminal-node assignments.</p>
 */
public final class ProximityContextBuilder {

    private static final int MINIMUM_PARALLEL_TREE_COUNT = 2;

    private ProximityContextBuilder() {
    }

    /**
     * Builds a train/train context using the selected measure's train
     * requirements.
     */
    public static ProximityContext buildTrain(
            ProximityForest forest,
            int trainingSize,
            ProximityMeasure measure,
            ProximityContextLayout layout,
            ParallelRuntime runtime
    ) throws Exception {
        Objects.requireNonNull(measure, "ProximityMeasure cannot be null.");
        measure.validateContract();

        return build(
                forest,
                trainingSize,
                0,
                measure.trainRequirements(),
                layout,
                runtime
        );
    }

    /**
     * Builds a test/train context using the selected measure's test/train
     * requirements.
     */
    public static ProximityContext buildTestTrain(
            ProximityForest forest,
            int trainingSize,
            int testingSize,
            ProximityMeasure measure,
            ProximityContextLayout layout,
            ParallelRuntime runtime
    ) throws Exception {
        Objects.requireNonNull(measure, "ProximityMeasure cannot be null.");
        measure.validateContract();

        return build(
                forest,
                trainingSize,
                testingSize,
                measure.testTrainRequirements(),
                layout,
                runtime
        );
    }

    /**
     * Builds a context for an explicit requirement set.
     *
     * <p>This overload is primarily useful for tests and future composite
     * proximity workflows.</p>
     */
    public static ProximityContext build(
            ProximityForest forest,
            int trainingSize,
            int testingSize,
            Set<ProximityDataRequirement> requirements,
            ProximityContextLayout layout,
            ParallelRuntime runtime
    ) throws Exception {
        validateInputs(
                forest,
                trainingSize,
                testingSize,
                requirements,
                layout,
                runtime
        );

        ProximityTree[] trees = forest.getTrees();
        Set<ProximityDataRequirement> preparedRequirements =
                immutableRequirements(requirements);

        return switch (layout) {
            case DENSE_INDEXED -> buildDense(
                    trees,
                    trainingSize,
                    testingSize,
                    preparedRequirements,
                    runtime
            );
            case SPARSE_TREE -> buildSparse(
                    trees,
                    trainingSize,
                    testingSize,
                    preparedRequirements,
                    runtime
            );
        };
    }

    private static ProximityContext buildSparse(
            ProximityTree[] trees,
            int trainingSize,
            int testingSize,
            Set<ProximityDataRequirement> requirements,
            ParallelRuntime runtime
    ) throws Exception {
        SparseTreeProximityContext.TreeData[] preparedTrees =
                new SparseTreeProximityContext.TreeData[trees.length];

        forEachTree(
                trees.length,
                runtime,
                treeIndex -> {
                    PreparedTree prepared = prepareTree(
                            trees[treeIndex],
                            treeIndex,
                            trainingSize,
                            testingSize,
                            requirements
                    );

                    preparedTrees[treeIndex] = prepared.toSparseTreeData();
                }
        );

        return new SparseTreeProximityContext(
                trainingSize,
                testingSize,
                requirements,
                preparedTrees
        );
    }

    private static ProximityContext buildDense(
            ProximityTree[] trees,
            int trainingSize,
            int testingSize,
            Set<ProximityDataRequirement> requirements,
            ParallelRuntime runtime
    ) throws Exception {
        int treeCount = trees.length;
        int trainingEntries = checkedEntryCount(trainingSize, treeCount, "training");
        int testingEntries = checkedEntryCount(testingSize, treeCount, "testing");

        int[] denseInBag = requested(
                requirements,
                ProximityDataRequirement.TRAIN_IN_BAG_LEAF_ASSIGNMENTS
        ) ? initializedNodeArray(trainingEntries) : null;

        int[] denseOutOfBag = requested(
                requirements,
                ProximityDataRequirement.TRAIN_OUT_OF_BAG_LEAF_ASSIGNMENTS
        ) ? initializedNodeArray(trainingEntries) : null;

        int[] denseTest = requested(
                requirements,
                ProximityDataRequirement.TEST_LEAF_ASSIGNMENTS
        ) ? initializedNodeArray(testingEntries) : null;

        int[] denseMultiplicities = requested(
                requirements,
                ProximityDataRequirement.BOOTSTRAP_MULTIPLICITIES
        ) ? new int[trainingEntries] : null;

        long[] denseOobBits = needsSeparateOobBits(requirements)
                ? new long[(trainingEntries + 63) >>> 6]
                : null;

        long[][] treeOobBits = denseOobBits == null
                ? null
                : new long[treeCount][];

        DenseIndexedProximityContext.TreeData[] preparedTrees =
                new DenseIndexedProximityContext.TreeData[treeCount];

        forEachTree(
                treeCount,
                runtime,
                treeIndex -> {
                    PreparedTree prepared = prepareTree(
                            trees[treeIndex],
                            treeIndex,
                            trainingSize,
                            testingSize,
                            requirements
                    );

                    copyColumn(
                            prepared.trainInBagLeafNodeIds,
                            denseInBag,
                            treeIndex,
                            treeCount
                    );
                    copyColumn(
                            prepared.trainOutOfBagLeafNodeIds,
                            denseOutOfBag,
                            treeIndex,
                            treeCount
                    );
                    copyColumn(
                            prepared.testLeafNodeIds,
                            denseTest,
                            treeIndex,
                            treeCount
                    );
                    copyColumn(
                            prepared.bootstrapMultiplicities,
                            denseMultiplicities,
                            treeIndex,
                            treeCount
                    );

                    if (treeOobBits != null) {
                        treeOobBits[treeIndex] = prepared.outOfBagBits;
                    }

                    preparedTrees[treeIndex] = prepared.toDenseTreeData();
                }
        );

        if (denseOobBits != null) {
            for (int treeIndex = 0; treeIndex < treeCount; treeIndex++) {
                copyOobColumn(
                        treeOobBits[treeIndex],
                        denseOobBits,
                        trainingSize,
                        treeIndex,
                        treeCount
                );
            }
        }

        return new DenseIndexedProximityContext(
                trainingSize,
                testingSize,
                requirements,
                denseInBag,
                denseOutOfBag,
                denseTest,
                denseMultiplicities,
                denseOobBits,
                preparedTrees
        );
    }

    private static PreparedTree prepareTree(
            ProximityTree tree,
            int treeIndex,
            int trainingSize,
            int testingSize,
            Set<ProximityDataRequirement> requirements
    ) {
        if (tree == null || tree.getRootNode() == null) {
            throw new IllegalStateException(
                    "Cannot prepare proximity context for null or untrained tree "
                            + treeIndex + "."
            );
        }

        Topology topology = buildTopology(tree.getRootNode(), treeIndex);

        int[] multiplicities = needsMultiplicities(requirements)
                ? buildMultiplicities(tree, treeIndex, trainingSize)
                : null;

        int[] inBagAssignments = needsInBagAssignments(requirements)
                ? initializedNodeArray(trainingSize)
                : null;

        int[] outOfBagAssignments = needsOutOfBagAssignments(requirements)
                ? initializedNodeArray(trainingSize)
                : null;

        int[] testAssignments = requested(
                requirements,
                ProximityDataRequirement.TEST_LEAF_ASSIGNMENTS
        ) ? initializedNodeArray(testingSize) : null;

        populateAssignments(
                topology,
                treeIndex,
                trainingSize,
                testingSize,
                inBagAssignments,
                outOfBagAssignments,
                testAssignments
        );

        long[] outOfBagBits = needsSeparateOobBits(requirements)
                ? buildOobBits(multiplicities, trainingSize)
                : null;

        Membership inBagMembers = requested(
                requirements,
                ProximityDataRequirement.LEAF_IN_BAG_DISTINCT_MEMBERS
        ) ? buildInBagMembership(
                topology.nodeCount,
                trainingSize,
                inBagAssignments,
                multiplicities,
                treeIndex
        ) : Membership.absent();

        Membership allTrainMembers = requested(
                requirements,
                ProximityDataRequirement.LEAF_ALL_TRAIN_MEMBERS
        ) ? buildAllTrainMembership(
                topology.nodeCount,
                trainingSize,
                inBagAssignments,
                outOfBagAssignments,
                treeIndex
        ) : Membership.absent();

        int[] leafMultiplicityTotals = requested(
                requirements,
                ProximityDataRequirement.LEAF_IN_BAG_MULTIPLICITY_TOTALS
        ) ? buildLeafMultiplicityTotals(
                topology.nodeCount,
                trainingSize,
                inBagAssignments,
                multiplicities,
                treeIndex
        ) : null;

        int[] parentNodeIds = requested(
                requirements,
                ProximityDataRequirement.NODE_ANCESTRY
        ) ? topology.parentNodeIds : null;

        int[] nodeDepths = requested(
                requirements,
                ProximityDataRequirement.NODE_ANCESTRY
        ) ? topology.nodeDepths : null;

        return new PreparedTree(
                topology.nodeCount,
                requested(
                        requirements,
                        ProximityDataRequirement.TRAIN_IN_BAG_LEAF_ASSIGNMENTS
                ) ? inBagAssignments : null,
                requested(
                        requirements,
                        ProximityDataRequirement.TRAIN_OUT_OF_BAG_LEAF_ASSIGNMENTS
                ) ? outOfBagAssignments : null,
                testAssignments,
                requested(
                        requirements,
                        ProximityDataRequirement.BOOTSTRAP_MULTIPLICITIES
                ) ? multiplicities : null,
                outOfBagBits,
                inBagMembers.offsets,
                inBagMembers.members,
                allTrainMembers.offsets,
                allTrainMembers.members,
                leafMultiplicityTotals,
                parentNodeIds,
                nodeDepths
        );
    }

    private static Topology buildTopology(
            ProximityTree.Node root,
            int treeIndex
    ) {
        List<ProximityTree.Node> nodes = new ArrayList<>();
        List<Integer> parents = new ArrayList<>();
        List<Integer> depths = new ArrayList<>();
        IdentityHashMap<ProximityTree.Node, Integer> nodeIds =
                new IdentityHashMap<>();

        collectTopology(
                root,
                ProximityContext.NO_PARENT,
                0,
                nodes,
                parents,
                depths,
                nodeIds,
                treeIndex
        );

        int[] parentNodeIds = new int[parents.size()];
        int[] nodeDepths = new int[depths.size()];

        for (int nodeId = 0; nodeId < parents.size(); nodeId++) {
            parentNodeIds[nodeId] = parents.get(nodeId);
            nodeDepths[nodeId] = depths.get(nodeId);
        }

        return new Topology(
                nodes.size(),
                nodes.toArray(ProximityTree.Node[]::new),
                nodeIds,
                parentNodeIds,
                nodeDepths
        );
    }

    private static void collectTopology(
            ProximityTree.Node node,
            int parentNodeId,
            int depth,
            List<ProximityTree.Node> nodes,
            List<Integer> parents,
            List<Integer> depths,
            IdentityHashMap<ProximityTree.Node, Integer> nodeIds,
            int treeIndex
    ) {
        if (node == null) {
            throw new IllegalStateException(
                    "Tree " + treeIndex + " contains a null node."
            );
        }
        if (nodeIds.containsKey(node)) {
            throw new IllegalStateException(
                    "Tree " + treeIndex + " contains a cycle or shared child."
            );
        }

        int nodeId = nodes.size();
        nodes.add(node);
        parents.add(parentNodeId);
        depths.add(depth);
        nodeIds.put(node, nodeId);

        ProximityTree.Node[] children = node.get_children();
        if (children == null) {
            return;
        }

        for (ProximityTree.Node child : children) {
            collectTopology(
                    child,
                    nodeId,
                    depth + 1,
                    nodes,
                    parents,
                    depths,
                    nodeIds,
                    treeIndex
            );
        }
    }

    private static int[] buildMultiplicities(
            ProximityTree tree,
            int treeIndex,
            int trainingSize
    ) {
        int[] multiplicities = new int[trainingSize];
        Map<Integer, Integer> source = tree.getRootNode().getMultiplicities();

        if (source == null) {
            throw new IllegalStateException(
                    "Tree " + treeIndex + " has no bootstrap multiplicities."
            );
        }

        for (Map.Entry<Integer, Integer> entry : source.entrySet()) {
            Integer index = entry.getKey();
            Integer count = entry.getValue();

            if (index == null || index < 0 || index >= trainingSize) {
                throw new IllegalStateException(
                        "Tree " + treeIndex
                                + " contains an invalid multiplicity index: "
                                + index + "."
                );
            }
            if (count == null || count <= 0) {
                throw new IllegalStateException(
                        "Tree " + treeIndex + " contains invalid multiplicity "
                                + count + " for training index " + index + "."
                );
            }

            multiplicities[index] = count;
        }

        return multiplicities;
    }

    private static void populateAssignments(
            Topology topology,
            int treeIndex,
            int trainingSize,
            int testingSize,
            int[] inBagAssignments,
            int[] outOfBagAssignments,
            int[] testAssignments
    ) {
        for (int nodeId = 0; nodeId < topology.nodeCount; nodeId++) {
            ProximityTree.Node node = topology.nodes[nodeId];
            if (node.get_children() != null) {
                continue;
            }

            if (inBagAssignments != null) {
                assignIndices(
                        node.getInBagIndices(),
                        inBagAssignments,
                        nodeId,
                        trainingSize,
                        "in-bag training",
                        treeIndex
                );
            }

            if (outOfBagAssignments != null) {
                assignIndices(
                        node.getOutOfBagIndices(),
                        outOfBagAssignments,
                        nodeId,
                        trainingSize,
                        "out-of-bag training",
                        treeIndex
                );
            }

            if (testAssignments != null) {
                assignIndices(
                        node.TestIndices,
                        testAssignments,
                        nodeId,
                        testingSize,
                        "test",
                        treeIndex
                );
            }
        }
    }

    private static void assignIndices(
            List<Integer> source,
            int[] assignments,
            int nodeId,
            int domainSize,
            String label,
            int treeIndex
    ) {
        if (source == null) {
            return;
        }

        for (Integer boxedIndex : source) {
            if (boxedIndex == null
                    || boxedIndex < 0
                    || boxedIndex >= domainSize) {

                throw new IllegalStateException(
                        "Tree " + treeIndex + " leaf " + nodeId
                                + " contains invalid " + label + " index "
                                + boxedIndex + "."
                );
            }

            int current = assignments[boxedIndex];
            if (current != ProximityContext.NO_NODE && current != nodeId) {
                throw new IllegalStateException(
                        "Tree " + treeIndex + ' ' + label + " index "
                                + boxedIndex + " is assigned to leaves "
                                + current + " and " + nodeId + "."
                );
            }

            assignments[boxedIndex] = nodeId;
        }
    }

    private static Membership buildInBagMembership(
            int nodeCount,
            int trainingSize,
            int[] inBagAssignments,
            int[] multiplicities,
            int treeIndex
    ) {
        requireArray(inBagAssignments, "in-bag assignments", treeIndex);
        requireArray(multiplicities, "bootstrap multiplicities", treeIndex);

        int[] counts = new int[nodeCount];
        for (int index = 0; index < trainingSize; index++) {
            if (multiplicities[index] <= 0) {
                continue;
            }

            int nodeId = inBagAssignments[index];
            requireAssigned(nodeId, index, "in-bag", treeIndex);
            counts[nodeId]++;
        }

        int[] offsets = prefixOffsets(counts);
        int[] members = new int[offsets[nodeCount]];
        int[] cursors = Arrays.copyOf(offsets, nodeCount);

        for (int index = 0; index < trainingSize; index++) {
            if (multiplicities[index] > 0) {
                int nodeId = inBagAssignments[index];
                members[cursors[nodeId]++] = index;
            }
        }

        return new Membership(offsets, members);
    }

    private static Membership buildAllTrainMembership(
            int nodeCount,
            int trainingSize,
            int[] inBagAssignments,
            int[] outOfBagAssignments,
            int treeIndex
    ) {
        requireArray(inBagAssignments, "in-bag assignments", treeIndex);
        requireArray(outOfBagAssignments, "out-of-bag assignments", treeIndex);

        int[] routedLeaf = new int[trainingSize];
        int[] counts = new int[nodeCount];

        for (int index = 0; index < trainingSize; index++) {
            int inBagLeaf = inBagAssignments[index];
            int outOfBagLeaf = outOfBagAssignments[index];

            if (inBagLeaf != ProximityContext.NO_NODE
                    && outOfBagLeaf != ProximityContext.NO_NODE) {
                throw new IllegalStateException(
                        "Tree " + treeIndex + " training index " + index
                                + " is marked both in bag and out of bag."
                );
            }

            int nodeId = inBagLeaf != ProximityContext.NO_NODE
                    ? inBagLeaf
                    : outOfBagLeaf;

            requireAssigned(nodeId, index, "routed training", treeIndex);
            routedLeaf[index] = nodeId;
            counts[nodeId]++;
        }

        int[] offsets = prefixOffsets(counts);
        int[] members = new int[offsets[nodeCount]];
        int[] cursors = Arrays.copyOf(offsets, nodeCount);

        for (int index = 0; index < trainingSize; index++) {
            int nodeId = routedLeaf[index];
            members[cursors[nodeId]++] = index;
        }

        return new Membership(offsets, members);
    }

    private static int[] buildLeafMultiplicityTotals(
            int nodeCount,
            int trainingSize,
            int[] inBagAssignments,
            int[] multiplicities,
            int treeIndex
    ) {
        requireArray(inBagAssignments, "in-bag assignments", treeIndex);
        requireArray(multiplicities, "bootstrap multiplicities", treeIndex);

        int[] totals = new int[nodeCount];
        for (int index = 0; index < trainingSize; index++) {
            int multiplicity = multiplicities[index];
            if (multiplicity <= 0) {
                continue;
            }

            int nodeId = inBagAssignments[index];
            requireAssigned(nodeId, index, "in-bag", treeIndex);

            totals[nodeId] = Math.addExact(
                    totals[nodeId],
                    multiplicity
            );
        }

        return totals;
    }

    private static long[] buildOobBits(
            int[] multiplicities,
            int trainingSize
    ) {
        if (multiplicities == null) {
            throw new IllegalStateException(
                    "OOB bit preparation requires temporary multiplicities."
            );
        }

        long[] bits = new long[(trainingSize + 63) >>> 6];
        for (int index = 0; index < trainingSize; index++) {
            if (multiplicities[index] == 0) {
                bits[index >>> 6] |= 1L << (index & 63);
            }
        }
        return bits;
    }

    private static int[] prefixOffsets(int[] counts) {
        int[] offsets = new int[counts.length + 1];
        for (int index = 0; index < counts.length; index++) {
            offsets[index + 1] = Math.addExact(offsets[index], counts[index]);
        }
        return offsets;
    }

    private static void forEachTree(
            int treeCount,
            ParallelRuntime runtime,
            TreeAction action
    ) throws Exception {
        if (runtime.isParallel()
                && treeCount >= MINIMUM_PARALLEL_TREE_COUNT) {

            runtime.forRange(
                    0,
                    treeCount,
                    1,
                    action::run
            );
            return;
        }

        for (int treeIndex = 0; treeIndex < treeCount; treeIndex++) {
            action.run(treeIndex);
        }
    }

    private static void copyColumn(
            int[] source,
            int[] destination,
            int treeIndex,
            int treeCount
    ) {
        if (destination == null) {
            return;
        }
        if (source == null) {
            throw new IllegalStateException(
                    "Missing prepared source column for tree " + treeIndex + "."
            );
        }

        for (int instanceIndex = 0;
             instanceIndex < source.length;
             instanceIndex++) {

            destination[instanceIndex * treeCount + treeIndex] =
                    source[instanceIndex];
        }
    }

    private static void copyOobColumn(
            long[] sourceBits,
            long[] destinationBits,
            int trainingSize,
            int treeIndex,
            int treeCount
    ) {
        if (sourceBits == null) {
            throw new IllegalStateException(
                    "Missing OOB membership for tree " + treeIndex + "."
            );
        }

        for (int trainIndex = 0; trainIndex < trainingSize; trainIndex++) {
            if ((sourceBits[trainIndex >>> 6]
                    & (1L << (trainIndex & 63))) != 0L) {

                int destinationIndex = trainIndex * treeCount + treeIndex;
                destinationBits[destinationIndex >>> 6] |=
                        1L << (destinationIndex & 63);
            }
        }
    }

    private static void validateInputs(
            ProximityForest forest,
            int trainingSize,
            int testingSize,
            Set<ProximityDataRequirement> requirements,
            ProximityContextLayout layout,
            ParallelRuntime runtime
    ) {
        Objects.requireNonNull(forest, "ProximityForest cannot be null.");
        Objects.requireNonNull(requirements, "requirements cannot be null.");
        Objects.requireNonNull(layout, "ProximityContextLayout cannot be null.");
        Objects.requireNonNull(runtime, "ParallelRuntime cannot be null.");

        /*if (requirements.contains(null)) {
            throw new IllegalArgumentException(
                    "requirements cannot contain null."
            );
        }*/
        for (ProximityDataRequirement requirement : requirements) {
            if (requirement == null) {
                throw new IllegalArgumentException(
                        "requirements cannot contain null."
                );
            }
        }
        if (trainingSize < 0 || testingSize < 0) {
            throw new IllegalArgumentException(
                    "Training and testing sizes cannot be negative."
            );
        }
        if (forest.getTrees() == null) {
            throw new IllegalStateException(
                    "ProximityForest has no tree array."
            );
        }
    }

    private static Set<ProximityDataRequirement> immutableRequirements(
            Set<ProximityDataRequirement> requirements
    ) {
        if (requirements.isEmpty()) {
            return Set.of();
        }

        return Collections.unmodifiableSet(
                EnumSet.copyOf(requirements)
        );
    }

    private static boolean requested(
            Set<ProximityDataRequirement> requirements,
            ProximityDataRequirement requirement
    ) {
        return requirements.contains(requirement);
    }

    private static boolean needsInBagAssignments(
            Set<ProximityDataRequirement> requirements
    ) {
        return requested(
                requirements,
                ProximityDataRequirement.TRAIN_IN_BAG_LEAF_ASSIGNMENTS
        ) || requested(
                requirements,
                ProximityDataRequirement.LEAF_IN_BAG_DISTINCT_MEMBERS
        ) || requested(
                requirements,
                ProximityDataRequirement.LEAF_IN_BAG_MULTIPLICITY_TOTALS
        ) || requested(
                requirements,
                ProximityDataRequirement.LEAF_ALL_TRAIN_MEMBERS
        );
    }

    private static boolean needsOutOfBagAssignments(
            Set<ProximityDataRequirement> requirements
    ) {
        return requested(
                requirements,
                ProximityDataRequirement.TRAIN_OUT_OF_BAG_LEAF_ASSIGNMENTS
        ) || requested(
                requirements,
                ProximityDataRequirement.LEAF_ALL_TRAIN_MEMBERS
        );
    }

    private static boolean needsMultiplicities(
            Set<ProximityDataRequirement> requirements
    ) {
        return requested(
                requirements,
                ProximityDataRequirement.BOOTSTRAP_MULTIPLICITIES
        ) || requested(
                requirements,
                ProximityDataRequirement.OUT_OF_BAG_MEMBERSHIP
        ) || requested(
                requirements,
                ProximityDataRequirement.LEAF_IN_BAG_DISTINCT_MEMBERS
        ) || requested(
                requirements,
                ProximityDataRequirement.LEAF_IN_BAG_MULTIPLICITY_TOTALS
        );
    }

    private static boolean needsSeparateOobBits(
            Set<ProximityDataRequirement> requirements
    ) {
        return requested(
                requirements,
                ProximityDataRequirement.OUT_OF_BAG_MEMBERSHIP
        ) && !requested(
                requirements,
                ProximityDataRequirement.BOOTSTRAP_MULTIPLICITIES
        );
    }

    private static int[] initializedNodeArray(int size) {
        int[] values = new int[size];
        Arrays.fill(values, ProximityContext.NO_NODE);
        return values;
    }

    private static int checkedEntryCount(
            int instanceCount,
            int treeCount,
            String label
    ) {
        long count = (long) instanceCount * treeCount;
        if (count > Integer.MAX_VALUE) {
            throw new IllegalArgumentException(
                    "Dense " + label + " context requires " + count
                            + " entries. Use SPARSE_TREE layout."
            );
        }
        return (int) count;
    }

    private static void requireArray(
            int[] values,
            String label,
            int treeIndex
    ) {
        if (values == null) {
            throw new IllegalStateException(
                    "Tree " + treeIndex + " requires " + label + "."
            );
        }
    }

    private static void requireAssigned(
            int nodeId,
            int trainIndex,
            String label,
            int treeIndex
    ) {
        if (nodeId == ProximityContext.NO_NODE) {
            throw new IllegalStateException(
                    "Tree " + treeIndex + ' ' + label + " index "
                            + trainIndex + " has no terminal-node assignment."
            );
        }
    }

    @FunctionalInterface
    private interface TreeAction {
        void run(int treeIndex) throws Exception;
    }

    private record Membership(int[] offsets, int[] members) {
        private static Membership absent() {
            return new Membership(null, null);
        }
    }

    private record Topology(
            int nodeCount,
            ProximityTree.Node[] nodes,
            IdentityHashMap<ProximityTree.Node, Integer> nodeIds,
            int[] parentNodeIds,
            int[] nodeDepths
    ) {
    }

    private record PreparedTree(
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
        private SparseTreeProximityContext.TreeData toSparseTreeData() {
            return new SparseTreeProximityContext.TreeData(
                    nodeCount,
                    trainInBagLeafNodeIds,
                    trainOutOfBagLeafNodeIds,
                    testLeafNodeIds,
                    bootstrapMultiplicities,
                    outOfBagBits,
                    inBagMemberOffsets,
                    inBagMembers,
                    allTrainMemberOffsets,
                    allTrainMembers,
                    leafInBagMultiplicityTotals,
                    parentNodeIds,
                    nodeDepths
            );
        }

        private DenseIndexedProximityContext.TreeData toDenseTreeData() {
            return new DenseIndexedProximityContext.TreeData(
                    nodeCount,
                    inBagMemberOffsets,
                    inBagMembers,
                    allTrainMemberOffsets,
                    allTrainMembers,
                    leafInBagMultiplicityTotals,
                    parentNodeIds,
                    nodeDepths
            );
        }
    }
}
