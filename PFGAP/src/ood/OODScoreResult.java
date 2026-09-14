package ood;

import java.io.Serial;
import java.io.Serializable;
import java.util.Objects;

/**
 * Immutable tree-level result produced by one {@link PathOODScorer} traversal.
 *
 * <p>The score summarizes one root-to-leaf path. Forest-level evaluation is
 * responsible for combining available tree scores and calculating across-tree
 * statistics such as the mean and standard deviation.</p>
 *
 * <p>An unavailable result stores {@link Double#NaN} as its score. Unavailable
 * is distinct from a valid score of zero: zero means that the scorer observed
 * usable training summaries but found no positive OOD penalty, while
 * unavailable means that no usable scoring summary was encountered.</p>
 *
 * <p>The result intentionally contains no infinity-specific diagnostics. Such
 * details may be collected separately through {@link PathDistanceEvidence}
 * when diagnostic output is requested.</p>
 */
public final class OODScoreResult implements Serializable {

    @Serial
    private static final long serialVersionUID = 1L;

    private final OODScoreType scoreType;
    private final double score;
    private final int visitedNodeCount;
    private final int summarizedNodeCount;

    /**
     * Creates an available tree-level OOD result.
     *
     * @param scoreType scoring method that produced the result
     * @param score finite nonnegative tree score
     * @param visitedNodeCount number of internal nodes on the traversed path
     * @param summarizedNodeCount number of visited nodes having a usable
     *                            branch-local training summary
     * @return available immutable result
     */
    public static OODScoreResult available(
            OODScoreType scoreType,
            double score,
            int visitedNodeCount,
            int summarizedNodeCount
    ) {
        return new OODScoreResult(
                scoreType,
                score,
                visitedNodeCount,
                summarizedNodeCount,
                true
        );
    }

    /**
     * Creates an unavailable tree-level OOD result.
     *
     * <p>This is appropriate when traversal visited no internal nodes or when
     * none of the visited splitters supplied a usable branch-local training
     * summary.</p>
     *
     * @param scoreType scoring method that attempted the calculation
     * @param visitedNodeCount number of internal nodes on the traversed path
     * @param summarizedNodeCount number of visited nodes having a usable
     *                            branch-local training summary, normally zero
     * @return unavailable immutable result with a NaN score
     */
    public static OODScoreResult unavailable(
            OODScoreType scoreType,
            int visitedNodeCount,
            int summarizedNodeCount
    ) {
        return new OODScoreResult(
                scoreType,
                Double.NaN,
                visitedNodeCount,
                summarizedNodeCount,
                false
        );
    }

    private OODScoreResult(
            OODScoreType scoreType,
            double score,
            int visitedNodeCount,
            int summarizedNodeCount,
            boolean available
    ) {
        this.scoreType = Objects.requireNonNull(
                scoreType,
                "OOD score type cannot be null."
        );

        validateCounts(visitedNodeCount, summarizedNodeCount);
        validateScore(score, available);

        this.score = available ? canonicalizeZero(score) : Double.NaN;
        this.visitedNodeCount = visitedNodeCount;
        this.summarizedNodeCount = summarizedNodeCount;
    }

    /** Returns the scoring method that produced this result. */
    public OODScoreType scoreType() {
        return scoreType;
    }

    /**
     * Returns whether a tree-level score is available.
     *
     * <p>Availability is represented canonically by a non-NaN score.</p>
     */
    public boolean isAvailable() {
        return !Double.isNaN(score);
    }

    /**
     * Returns the finite nonnegative tree score, or NaN when unavailable.
     */
    public double score() {
        return score;
    }

    /** Returns the number of visited internal nodes. */
    public int visitedNodeCount() {
        return visitedNodeCount;
    }

    /**
     * Returns the number of visited nodes having a usable branch-local training
     * summary for this scorer.
     */
    public int summarizedNodeCount() {
        return summarizedNodeCount;
    }

    /**
     * Returns the fraction of visited nodes having a usable scoring summary.
     * Returns NaN when the path contains no internal nodes.
     */
    public double summarizedNodeFraction() {
        return visitedNodeCount == 0
                ? Double.NaN
                : (double) summarizedNodeCount / visitedNodeCount;
    }

    private static void validateCounts(
            int visitedNodeCount,
            int summarizedNodeCount
    ) {
        if (visitedNodeCount < 0) {
            throw new IllegalArgumentException(
                    "Visited node count cannot be negative: "
                            + visitedNodeCount + "."
            );
        }
        if (summarizedNodeCount < 0) {
            throw new IllegalArgumentException(
                    "Summarized node count cannot be negative: "
                            + summarizedNodeCount + "."
            );
        }
        if (summarizedNodeCount > visitedNodeCount) {
            throw new IllegalArgumentException(
                    "Summarized node count " + summarizedNodeCount
                            + " cannot exceed visited node count "
                            + visitedNodeCount + "."
            );
        }
    }

    private static void validateScore(double score, boolean available) {
        if (!available) {
            if (!Double.isNaN(score)) {
                throw new IllegalArgumentException(
                        "An unavailable OOD result must use a NaN score."
                );
            }
            return;
        }

        if (!Double.isFinite(score)) {
            throw new IllegalArgumentException(
                    "An available OOD score must be finite, but received "
                            + score + "."
            );
        }
        if (score < 0.0) {
            throw new IllegalArgumentException(
                    "An available OOD score cannot be negative: "
                            + score + "."
            );
        }
    }

    private static double canonicalizeZero(double value) {
        return value == 0.0 ? 0.0 : value;
    }

    @Override
    public boolean equals(Object other) {
        if (this == other) {
            return true;
        }
        if (!(other instanceof OODScoreResult result)) {
            return false;
        }
        return scoreType == result.scoreType
                && Double.compare(score, result.score) == 0
                && visitedNodeCount == result.visitedNodeCount
                && summarizedNodeCount == result.summarizedNodeCount;
    }

    @Override
    public int hashCode() {
        int result = scoreType.hashCode();
        result = 31 * result + Double.hashCode(score);
        result = 31 * result + visitedNodeCount;
        result = 31 * result + summarizedNodeCount;
        return result;
    }

    @Override
    public String toString() {
        return "OODScoreResult{"
                + "scoreType=" + scoreType
                + ", score=" + score
                + ", visitedNodeCount=" + visitedNodeCount
                + ", summarizedNodeCount=" + summarizedNodeCount
                + '}';
    }
}
