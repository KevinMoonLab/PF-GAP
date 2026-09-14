package ood;

/**
 * Mutable allocation-free diagnostic summary of one root-to-leaf
 * distance-observing traversal.
 *
 * <p>This class implements {@link SplitDistanceObserver} and records only
 * branch-local information encountered along one tree path. It deliberately
 * does not calculate an OOD score, choose a relative scale, or aggregate
 * node-level penalties. Those decisions belong to {@link PathOODScorer}
 * implementations.</p>
 *
 * <p>One instance represents one traversal at a time. Call {@link #reset()}
 * before reusing it for another query-tree pair. Instances are mutable and are
 * not thread-safe. Parallel evaluation must give each concurrent traversal its
 * own instance.</p>
 *
 * <p>No per-node objects, arrays, or collections are retained. Memory usage is
 * constant with respect to tree depth.</p>
 */
public final class PathDistanceEvidence implements SplitDistanceObserver {

    private int visitedNodeCount;
    private int deepestObservedDepth;

    private int summaryAvailableCount;
    private int summaryUnavailableCount;

    private int finiteWinningDistanceCount;
    private int positiveInfinityWinningDistanceCount;

    private int selectedBranchSummaryAvailableCount;
    private int selectedBranchSummaryEmptyCount;
    private int selectedBranchFiniteBoundaryCount;
    private int selectedBranchInfiniteBoundaryCount;

    private double minimumFiniteWinningDistance;
    private double maximumFiniteWinningDistance;

    private int lastNodeId;
    private long lastNodePathIdentity;
    private int lastDepth;
    private int lastBranch;
    private double lastWinningDistance;
    private SplitDistanceSummary lastTrainingSummary;

    /** Creates empty path evidence. */
    public PathDistanceEvidence() {
        reset();
    }

    /** Clears all state so this instance can be reused for another traversal. */
    public void reset() {
        visitedNodeCount = 0;
        deepestObservedDepth = -1;

        summaryAvailableCount = 0;
        summaryUnavailableCount = 0;

        finiteWinningDistanceCount = 0;
        positiveInfinityWinningDistanceCount = 0;

        selectedBranchSummaryAvailableCount = 0;
        selectedBranchSummaryEmptyCount = 0;
        selectedBranchFiniteBoundaryCount = 0;
        selectedBranchInfiniteBoundaryCount = 0;

        minimumFiniteWinningDistance = Double.NaN;
        maximumFiniteWinningDistance = Double.NaN;

        lastNodeId = -1;
        lastNodePathIdentity = 0L;
        lastDepth = -1;
        lastBranch = -1;
        lastWinningDistance = Double.NaN;
        lastTrainingSummary = null;
    }

    /**
     * Records one internal-node routing decision.
     *
     * <p>Positive infinity is a valid winning distance and is counted
     * separately. NaN, negative infinity, negative structural values, and a
     * branch outside the supplied split summary are rejected as traversal
     * contract violations.</p>
     */
    @Override
    public void observe(
            int nodeId,
            long nodePathIdentity,
            int depth,
            int branch,
            double winningDistance,
            SplitDistanceSummary trainingSummary
    ) {
        validateObservation(nodeId, depth, branch, winningDistance);

        visitedNodeCount++;
        deepestObservedDepth = Math.max(deepestObservedDepth, depth);

        lastNodeId = nodeId;
        lastNodePathIdentity = nodePathIdentity;
        lastDepth = depth;
        lastBranch = branch;
        lastWinningDistance = canonicalizeZero(winningDistance);
        lastTrainingSummary = trainingSummary;

        if (winningDistance == Double.POSITIVE_INFINITY) {
            positiveInfinityWinningDistanceCount++;
        } else {
            finiteWinningDistanceCount++;
            updateFiniteDistanceExtrema(winningDistance);
        }

        if (trainingSummary == null) {
            summaryUnavailableCount++;
            return;
        }

        summaryAvailableCount++;
        if (branch >= trainingSummary.branchCount()) {
            throw new IllegalArgumentException(
                    "Observed branch " + branch
                            + " exceeds training summary branch count "
                            + trainingSummary.branchCount()
                            + " at node " + nodeId + "."
            );
        }

        DistanceDistributionSummary branchSummary =
                trainingSummary.branch(branch);
        if (branchSummary.isEmpty()) {
            selectedBranchSummaryEmptyCount++;
            return;
        }

        selectedBranchSummaryAvailableCount++;
        if (branchSummary.hasFiniteUpperSupportBoundary()) {
            selectedBranchFiniteBoundaryCount++;
        } else {
            selectedBranchInfiniteBoundaryCount++;
        }
    }

    /** Returns whether no internal-node observation has been recorded. */
    public boolean isEmpty() {
        return visitedNodeCount == 0;
    }

    public int visitedNodeCount() {
        return visitedNodeCount;
    }

    /**
     * Returns the greatest observed zero-based internal-node depth, or -1 when
     * no observation has been recorded.
     */
    public int deepestObservedDepth() {
        return deepestObservedDepth;
    }

    public int summaryAvailableCount() {
        return summaryAvailableCount;
    }

    public int summaryUnavailableCount() {
        return summaryUnavailableCount;
    }

    public int finiteWinningDistanceCount() {
        return finiteWinningDistanceCount;
    }

    public int positiveInfinityWinningDistanceCount() {
        return positiveInfinityWinningDistanceCount;
    }

    public int selectedBranchSummaryAvailableCount() {
        return selectedBranchSummaryAvailableCount;
    }

    public int selectedBranchSummaryEmptyCount() {
        return selectedBranchSummaryEmptyCount;
    }

    public int selectedBranchFiniteBoundaryCount() {
        return selectedBranchFiniteBoundaryCount;
    }

    public int selectedBranchInfiniteBoundaryCount() {
        return selectedBranchInfiniteBoundaryCount;
    }

    /** Returns the minimum finite winning distance, or NaN when none was finite. */
    public double minimumFiniteWinningDistance() {
        return minimumFiniteWinningDistance;
    }

    /** Returns the maximum finite winning distance, or NaN when none was finite. */
    public double maximumFiniteWinningDistance() {
        return maximumFiniteWinningDistance;
    }

    public int lastNodeId() {
        return lastNodeId;
    }

    public long lastNodePathIdentity() {
        return lastNodePathIdentity;
    }

    public int lastDepth() {
        return lastDepth;
    }

    public int lastBranch() {
        return lastBranch;
    }

    public double lastWinningDistance() {
        return lastWinningDistance;
    }

    /**
     * Returns the immutable branch-local training summary from the most recent
     * observation, or null when no observation exists or summary collection was
     * unavailable.
     */
    public SplitDistanceSummary lastTrainingSummary() {
        return lastTrainingSummary;
    }

    /**
     * Returns the fraction of visited nodes having a split summary.
     * Returns NaN for an empty path.
     */
    public double summaryAvailabilityFraction() {
        return fraction(summaryAvailableCount, visitedNodeCount);
    }

    /**
     * Returns the fraction of visited nodes whose selected branch establishes a
     * finite empirical upper support boundary. Returns NaN for an empty path.
     */
    public double selectedBranchFiniteBoundaryFraction() {
        return fraction(selectedBranchFiniteBoundaryCount, visitedNodeCount);
    }

    /**
     * Returns the fraction of visited nodes whose winning query distance was
     * positive infinity. Returns NaN for an empty path.
     */
    public double positiveInfinityWinningDistanceFraction() {
        return fraction(positiveInfinityWinningDistanceCount, visitedNodeCount);
    }

    private void updateFiniteDistanceExtrema(double distance) {
        double canonicalDistance = canonicalizeZero(distance);
        if (finiteWinningDistanceCount == 1) {
            minimumFiniteWinningDistance = canonicalDistance;
            maximumFiniteWinningDistance = canonicalDistance;
            return;
        }

        minimumFiniteWinningDistance = Math.min(
                minimumFiniteWinningDistance,
                canonicalDistance
        );
        maximumFiniteWinningDistance = Math.max(
                maximumFiniteWinningDistance,
                canonicalDistance
        );
    }

    private static void validateObservation(
            int nodeId,
            int depth,
            int branch,
            double winningDistance
    ) {
        if (nodeId < 0) {
            throw new IllegalArgumentException(
                    "Observed nodeId cannot be negative: " + nodeId + "."
            );
        }
        if (depth < 0) {
            throw new IllegalArgumentException(
                    "Observed depth cannot be negative: " + depth + "."
            );
        }
        if (branch < 0) {
            throw new IllegalArgumentException(
                    "Observed branch cannot be negative: " + branch + "."
            );
        }
        if (Double.isNaN(winningDistance)) {
            throw new IllegalArgumentException(
                    "Observed winning distance cannot be NaN."
            );
        }
        if (winningDistance == Double.NEGATIVE_INFINITY) {
            throw new IllegalArgumentException(
                    "Observed winning distance cannot be negative infinity."
            );
        }
    }

    private static double fraction(int numerator, int denominator) {
        return denominator == 0
                ? Double.NaN
                : (double) numerator / denominator;
    }

    private static double canonicalizeZero(double value) {
        return value == 0.0 ? 0.0 : value;
    }

    @Override
    public String toString() {
        return "PathDistanceEvidence{"
                + "visitedNodeCount=" + visitedNodeCount
                + ", summaryAvailableCount=" + summaryAvailableCount
                + ", summaryUnavailableCount=" + summaryUnavailableCount
                + ", finiteWinningDistanceCount=" + finiteWinningDistanceCount
                + ", positiveInfinityWinningDistanceCount="
                + positiveInfinityWinningDistanceCount
                + ", selectedBranchSummaryAvailableCount="
                + selectedBranchSummaryAvailableCount
                + ", selectedBranchSummaryEmptyCount="
                + selectedBranchSummaryEmptyCount
                + ", selectedBranchFiniteBoundaryCount="
                + selectedBranchFiniteBoundaryCount
                + ", selectedBranchInfiniteBoundaryCount="
                + selectedBranchInfiniteBoundaryCount
                + '}';
    }
}
