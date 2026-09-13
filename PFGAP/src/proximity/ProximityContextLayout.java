package proximity;

/**
 * Physical storage layout used by a prepared {@link ProximityContext}.
 *
 * <p>The layout affects preparation cost and memory locality, not proximity
 * semantics. The selected {@link ProximityMeasure} and its declared
 * requirements determine which data components are prepared.</p>
 */
public enum ProximityContextLayout {

    /**
     * Flat instance-major primitive indexes optimized for dense scalar pair
     * evaluation. Consecutive values for one observation correspond to
     * consecutive trees.
     */
    DENSE_INDEXED,

    /**
     * Tree-local primitive indexes and CSR-like leaf memberships optimized for
     * sparse source-row accumulation.
     */
    SPARSE_TREE
}
