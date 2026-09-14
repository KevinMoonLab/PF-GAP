package ood;

/**
 * Mutable allocation-free scorer for one root-to-leaf tree traversal.
 *
 * <p>A path scorer consumes split-distance observations directly while the tree
 * is traversed. It must update only constant-size running state and must not
 * require the traversal to retain per-node event objects. After traversal,
 * {@link #finish()} returns the immutable tree-level OOD result.</p>
 *
 * <p>One scorer instance represents one query-tree traversal at a time. Call
 * {@link #reset()} before reusing an instance for another traversal. Scorers are
 * mutable and are not required to be thread-safe. Parallel forest evaluation
 * must give each concurrent traversal its own scorer instance.</p>
 *
 * <p>The interface intentionally leaves node weighting, path-length use,
 * handling of unavailable summaries, infinity policy, and score scaling to the
 * concrete scorer. This permits future OOD methods to reuse the same
 * distance-observing tree traversal without modifying the tree.</p>
 */
public interface PathOODScorer extends SplitDistanceObserver {

    /**
     * Returns the scoring method implemented by this scorer.
     *
     * @return stable scorer type used by configuration and result metadata
     */
    OODScoreType scoreType();

    /**
     * Clears all traversal-specific state before the scorer is reused.
     *
     * <p>Implementations must restore the same state produced by a newly
     * constructed scorer while retaining immutable configuration such as scale
     * floors or infinity-proxy parameters.</p>
     */
    void reset();

    /**
     * Finalizes the observations accumulated for the current tree path.
     *
     * <p>This method must not reset the scorer implicitly. The caller may inspect
     * the returned result and then invoke {@link #reset()} explicitly before the
     * next traversal.</p>
     *
     * <p>The returned result is tree-level. Forest-level code is responsible for
     * combining tree scores and calculating across-tree statistics such as mean
     * and standard deviation.</p>
     *
     * @return immutable result for the current root-to-leaf traversal
     */
    OODScoreResult finish();
}
