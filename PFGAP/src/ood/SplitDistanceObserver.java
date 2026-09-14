package ood;

/**
 * Receives allocation-free observations from distance-aware tree traversal.
 *
 * <p>The traversal invokes this observer once for each internal node visited by
 * a query, after the node's splitter has selected a branch and before traversal
 * continues to that child. The observer receives primitive routing values and
 * the immutable training summary directly, so traversal does not need to
 * allocate one event object per node.</p>
 *
 * <p>Implementations may accumulate path statistics, collect diagnostics, or
 * forward evidence to an OOD scoring policy. This interface deliberately does
 * not define an OOD formula, scale policy, or path aggregation rule.</p>
 *
 * <p>The {@code trainingSummary} argument may be {@code null}. A null value
 * means that split-distance summaries were not collected for that splitter,
 * were unavailable in a loaded model, or otherwise cannot be supplied. An
 * observer must distinguish unavailable evidence from a valid zero score.</p>
 *
 * <p>The {@code winningDistance} may be finite or
 * {@link Double#POSITIVE_INFINITY}. NaN and negative infinity violate the
 * distance-routing contract. An observer must not enter positive infinity into
 * finite-moment arithmetic without handling it explicitly.</p>
 *
 * <p>Observers are caller-owned and are not required to be thread-safe. A
 * parallel evaluator should use an independent observer per query, worker, or
 * other exclusive evaluation scope. Tree traversal must not invoke the same
 * mutable observer concurrently unless the implementation explicitly supports
 * that use.</p>
 */
@FunctionalInterface
public interface SplitDistanceObserver {

    /**
     * Observes one internal-node routing decision.
     *
     * @param nodeId trained tree-local node identifier
     * @param nodePathIdentity deterministic structural identity of the node
     * @param depth zero-based depth of the internal node, with the root at zero
     * @param branch selected zero-based child branch
     * @param winningDistance distance from the query to the selected exemplar;
     *                        finite or positive infinity
     * @param trainingSummary immutable winning-split training summary, or null
     *                        when summary evidence is unavailable
     */
    void observe(
            int nodeId,
            long nodePathIdentity,
            int depth,
            int branch,
            double winningDistance,
            SplitDistanceSummary trainingSummary
    );
}
