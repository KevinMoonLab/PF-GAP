package proximity;

import java.util.Set;

/**
 * Read-only access to forest-derived data prepared for proximity computation.
 *
 * <p>This interface separates proximity formulas from concrete storage.
 * Implementations may use dense primitive indexes, tree-local sparse indexes,
 * packed membership bits, segmented arrays, memory-mapped storage, or
 * measure-specific views.</p>
 *
 * <p>Accessors must not allocate during normal hot-path use. Leaf membership
 * is therefore exposed through count-and-index methods rather than through
 * newly allocated collections. Implementations must be safe for concurrent
 * reads after construction.</p>
 *
 * <p>A context is prepared for a declared set of
 * {@link ProximityDataRequirement} values. Calling an accessor whose
 * requirement was not prepared must fail clearly rather than silently return
 * a default value.</p>
 */
public interface ProximityContext {

    /** Sentinel indicating that an instance has no applicable node assignment. */
    int NO_NODE = -1;

    /** Sentinel indicating that a node has no parent. */
    int NO_PARENT = -1;

    /** Returns the number of represented trees. */
    int treeCount();

    /** Returns the number of represented training instances. */
    int trainingSize();

    /** Returns the number of represented test instances, or zero if absent. */
    int testingSize();

    /** Returns the immutable set of prepared data requirements. */
    Set<ProximityDataRequirement> preparedRequirements();

    /** Returns whether the specified requirement has been prepared. */
    default boolean hasRequirement(
            ProximityDataRequirement requirement
    ) {
        if (requirement == null) {
            return false;
        }

        Set<ProximityDataRequirement> prepared =
                preparedRequirements();

        return prepared != null
                && prepared.contains(requirement);
    }

    /**
     * Verifies that a required data view is available.
     *
     * @throws IllegalStateException when the requirement was not prepared
     */
    default void require(
            ProximityDataRequirement requirement
    ) {
        if (!hasRequirement(requirement)) {
            throw new IllegalStateException(
                    "Proximity context does not contain required data: "
                            + requirement
                            + "."
            );
        }
    }

    /**
     * Returns the leaf containing a training index as an in-bag observation,
     * or {@link #NO_NODE} when the index was not in bag.
     *
     * @see ProximityDataRequirement#TRAIN_IN_BAG_LEAF_ASSIGNMENTS
     */
    int trainInBagLeafNodeId(
            int treeIndex,
            int trainIndex
    );

    /**
     * Returns the leaf reached by a training index when it was out of bag, or
     * {@link #NO_NODE} when the index was not out of bag.
     *
     * @see ProximityDataRequirement#TRAIN_OUT_OF_BAG_LEAF_ASSIGNMENTS
     */
    int trainOutOfBagLeafNodeId(
            int treeIndex,
            int trainIndex
    );

    /**
     * Returns a training observation's routed leaf regardless of bootstrap
     * status.
     *
     * <p>The default implementation first checks in-bag membership and then
     * falls back to the out-of-bag routed assignment. Context implementations
     * may override this method when they store a direct all-training routing
     * index.</p>
     *
     * @return routed terminal-node identifier or {@link #NO_NODE}
     */
    default int trainLeafNodeId(
            int treeIndex,
            int trainIndex
    ) {
        int inBagLeafNodeId =
                trainInBagLeafNodeId(
                        treeIndex,
                        trainIndex
                );

        if (inBagLeafNodeId != NO_NODE) {
            return inBagLeafNodeId;
        }

        return trainOutOfBagLeafNodeId(
                treeIndex,
                trainIndex
        );
    }

    /**
     * Returns the leaf reached by a test index, or {@link #NO_NODE} when no
     * test assignment is available.
     *
     * @see ProximityDataRequirement#TEST_LEAF_ASSIGNMENTS
     */
    int testLeafNodeId(
            int treeIndex,
            int testIndex
    );

    /**
     * Returns the bootstrap multiplicity of a training index in a tree.
     * Zero means that the index was out of bag.
     *
     * @see ProximityDataRequirement#BOOTSTRAP_MULTIPLICITIES
     */
    int bootstrapMultiplicity(
            int treeIndex,
            int trainIndex
    );

    /**
     * Returns whether a training index was out of bag in a tree.
     *
     * <p>An implementation may use packed membership or derive this from a
     * zero bootstrap multiplicity.</p>
     *
     * @see ProximityDataRequirement#OUT_OF_BAG_MEMBERSHIP
     */
    boolean isOutOfBag(
            int treeIndex,
            int trainIndex
    );

    /**
     * Returns the number of distinct in-bag training indices in a leaf.
     *
     * @see ProximityDataRequirement#LEAF_IN_BAG_DISTINCT_MEMBERS
     */
    int leafInBagDistinctMemberCount(
            int treeIndex,
            int leafNodeId
    );

    /**
     * Returns one distinct in-bag training index from a leaf.
     *
     * @see ProximityDataRequirement#LEAF_IN_BAG_DISTINCT_MEMBERS
     */
    int leafInBagDistinctMemberAt(
            int treeIndex,
            int leafNodeId,
            int memberOffset
    );

    /**
     * Returns the number of distinct training observations routed to a leaf,
     * regardless of bootstrap status.
     *
     * <p>The valid member-offset range is
     * {@code [0, leafAllTrainMemberCount(treeIndex, leafNodeId))}.</p>
     *
     * @see ProximityDataRequirement#LEAF_ALL_TRAIN_MEMBERS
     */
    int leafAllTrainMemberCount(
            int treeIndex,
            int leafNodeId
    );

    /**
     * Returns one distinct training index routed to a leaf, regardless of
     * bootstrap status.
     *
     * <p>Each training index must occur at most once in this membership view,
     * even when its bootstrap multiplicity is greater than one.</p>
     *
     * @see ProximityDataRequirement#LEAF_ALL_TRAIN_MEMBERS
     */
    int leafAllTrainMemberAt(
            int treeIndex,
            int leafNodeId,
            int memberOffset
    );

    /**
     * Returns the total in-bag multiplicity in a leaf, including bootstrap
     * repetitions.
     *
     * @see ProximityDataRequirement#LEAF_IN_BAG_MULTIPLICITY_TOTALS
     */
    int leafInBagMultiplicityTotal(
            int treeIndex,
            int leafNodeId
    );

    /**
     * Returns a node's parent identifier, or {@link #NO_PARENT} for the root.
     *
     * @see ProximityDataRequirement#NODE_ANCESTRY
     */
    int parentNodeId(
            int treeIndex,
            int nodeId
    );

    /**
     * Returns a node's zero-based depth, where the root has depth zero.
     *
     * @see ProximityDataRequirement#NODE_ANCESTRY
     */
    int nodeDepth(
            int treeIndex,
            int nodeId
    );

    /**
     * Computes the depth of the lowest common ancestor of two nodes.
     *
     * <p>The default implementation uses parent and depth accessors and does
     * not allocate paths. Implementations may override it with a faster
     * ancestry index when profiling justifies the additional memory.</p>
     *
     * @return common-ancestor depth, or {@code -1} when either node is absent
     */
    default int commonAncestorDepth(
            int treeIndex,
            int firstNodeId,
            int secondNodeId
    ) {
        require(ProximityDataRequirement.NODE_ANCESTRY);

        if (firstNodeId == NO_NODE || secondNodeId == NO_NODE) {
            return -1;
        }

        int first =
                firstNodeId;

        int second =
                secondNodeId;

        int firstDepth =
                nodeDepth(
                        treeIndex,
                        first
                );

        int secondDepth =
                nodeDepth(
                        treeIndex,
                        second
                );

        while (firstDepth > secondDepth) {
            first =
                    parentNodeId(
                            treeIndex,
                            first
                    );

            if (first == NO_PARENT) {
                return -1;
            }

            firstDepth--;
        }

        while (secondDepth > firstDepth) {
            second =
                    parentNodeId(
                            treeIndex,
                            second
                    );

            if (second == NO_PARENT) {
                return -1;
            }

            secondDepth--;
        }

        while (first != second) {
            if (first == NO_PARENT || second == NO_PARENT) {
                return -1;
            }

            first =
                    parentNodeId(
                            treeIndex,
                            first
                    );

            second =
                    parentNodeId(
                            treeIndex,
                            second
                    );

            firstDepth--;
        }

        return firstDepth;
    }
}
