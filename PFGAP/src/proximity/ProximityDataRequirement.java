package proximity;

/**
 * Declares forest-derived data required by a proximity measure.
 *
 * <p>A {@link ProximityMeasure} should declare the smallest set of
 * requirements needed by its scalar and optimized accumulation paths. The
 * context builder can then prepare only those immutable lookup structures,
 * avoiding repeated tree scans without allocating every possible
 * {@code instanceCount * treeCount} index.</p>
 *
 * <p>Requirements describe data dependencies, not mathematical properties.
 * Symmetry and constant-diagonal guarantees belong in
 * {@link ProximityProperties}.</p>
 *
 * <p>For very large datasets, implementations should prefer leaf-oriented or
 * source-row accumulation when their formula permits it. A requirement does
 * not mandate a dense rectangular context array. It may be satisfied by a
 * compact tree-local index, primitive leaf membership arrays, packed bits,
 * memory-mapped storage, or another representation selected by the context
 * implementation.</p>
 */
public enum ProximityDataRequirement {

    /**
     * Provides the terminal-node assignment of each in-bag training index in
     * each tree.
     *
     * <p>An index absent from a tree's bootstrap sample has no in-bag leaf
     * assignment for that tree. A context may satisfy this requirement through
     * direct index-to-node lookup or through compact node-to-member data when
     * the selected algorithm accumulates complete sparse rows or leaf blocks.</p>
     */
    TRAIN_IN_BAG_LEAF_ASSIGNMENTS,

    /**
     * Provides the terminal node reached by each out-of-bag training index in
     * each tree for which it is out of bag.
     *
     * <p>This is distinct from in-bag membership. Source-conditioned measures
     * such as RF-GAP use an out-of-bag source observation's routed leaf while
     * distributing weight over the in-bag observations in that leaf.</p>
     */
    TRAIN_OUT_OF_BAG_LEAF_ASSIGNMENTS,

    /**
     * Provides the terminal node reached by each test index in each tree.
     *
     * <p>This requirement is meaningful for test/train calculations and
     * normally depends on test instances having already been routed through
     * the forest.</p>
     */
    TEST_LEAF_ASSIGNMENTS,

    /**
     * Provides each terminal node's distinct in-bag training indices.
     *
     * <p>This requirement supports leaf-oriented and sparse row-accumulation
     * algorithms that visit only targets capable of receiving a nonzero
     * contribution. It preserves distinct index membership; bootstrap
     * repetitions are represented separately by multiplicities.</p>
     */
    LEAF_IN_BAG_DISTINCT_MEMBERS,

    /**
     * Provides every distinct training index routed to each terminal node,
     * regardless of whether that observation was in bag or out of bag for the
     * tree.
     *
     * <p>This requirement enables exact leaf-oriented accumulation for
     * bootstrap-independent formulas such as Breiman's original proximity.
     * It can be expensive for very large datasets and therefore must be
     * prepared only when explicitly requested by the selected measure and
     * calculation domain.</p>
     *
     * <p>Membership is set-like: each training index appears at most once in a
     * leaf even when it has bootstrap multiplicity greater than one. The
     * builder should validate that every represented training observation is
     * routed to exactly one terminal node per tree.</p>
     */
    LEAF_ALL_TRAIN_MEMBERS,

    /**
     * Provides the total in-bag multiset size of each terminal node, including
     * bootstrap repetitions.
     *
     * <p>For RF-GAP notation, this is {@code |M_i(t)|} for the terminal node
     * reached by source observation {@code i} in tree {@code t}. Storing this
     * aggregate avoids reconstructing it by summing multiplicities in every
     * source-row calculation.</p>
     */
    LEAF_IN_BAG_MULTIPLICITY_TOTALS,

    /**
     * Provides the bootstrap multiplicity of each training index in each tree.
     *
     * <p>A multiplicity of zero means that the index is out of bag. A context
     * is therefore permitted to use this structure to satisfy
     * {@link #OUT_OF_BAG_MEMBERSHIP} without allocating a separate OOB index.</p>
     */
    BOOTSTRAP_MULTIPLICITIES,

    /**
     * Provides per-tree out-of-bag membership for training indices.
     *
     * <p>This supports source-conditioned measures whose contributing tree set
     * depends on whether the source observation is out of bag. The builder may
     * derive this information from zero bootstrap multiplicity when
     * {@link #BOOTSTRAP_MULTIPLICITIES} is also requested.</p>
     */
    OUT_OF_BAG_MEMBERSHIP,

    /**
     * Provides node ancestry information required for proximity based on
     * shared paths, common ancestors, graph distance, or node depth.
     *
     * <p>The prepared representation may use parent-node identifiers and
     * depths, ancestor tables, or another immutable tree-local structure. This
     * requirement does not prescribe a full path allocation for every leaf.</p>
     */
    NODE_ANCESTRY
}
