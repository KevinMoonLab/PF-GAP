package ood;

/**
 * Computes a bounded relative support exceedance score along one tree path.
 *
 * <p>For a finite, nonnegative query winning distance {@code d} and the
 * selected branch's finite, nonnegative training maximum {@code m}, the node
 * contribution is:</p>
 *
 * <pre>
 * 0,            if d <= m
 * 1 - (m / d),  if d > m
 * </pre>
 *
 * <p>Each node contribution lies in {@code [0, 1]}. A contribution of
 * {@code 0.5} means that the query distance is twice the observed training
 * maximum. A positive query distance against a zero training maximum
 * contributes {@code 1.0}. The formulation is invariant under positive
 * rescaling of the distance measure and requires no arbitrary scale floor.</p>
 *
 * <p>Distance-based OOD scoring requires finite winning distances and finite
 * branch training maxima. A distance measure that produces a non-finite
 * winning distance is not compatible with this score. Disabling early
 * abandoning can help determine whether a non-finite result comes from the
 * distance calculation itself.</p>
 *
 * <p>The tree score is the arithmetic mean of node contributions over every
 * visited internal node. Nodes without usable summary metadata contribute zero
 * to the numerator and remain in the denominator, making the score independent
 * of path length while preventing sparse metadata from amplifying a small
 * number of contributions. The resulting tree score also lies in
 * {@code [0, 1]}.</p>
 *
 * <p>Instances are mutable and not thread-safe. One instance represents one
 * query-tree traversal at a time.</p>
 */
public final class RelativeSupportExceedanceOODScorer
        implements PathOODScorer {

    private int visitedNodeCount;
    private int summarizedNodeCount;
    private double contributionSum;
    private double contributionCompensation;

    /** Creates an empty scorer for one query-tree traversal. */
    public RelativeSupportExceedanceOODScorer() {
        reset();
    }

    @Override
    public OODScoreType scoreType() {
        return OODScoreType.RELATIVE_SUPPORT_EXCEEDANCE;
    }

    @Override
    public void reset() {
        visitedNodeCount = 0;
        summarizedNodeCount = 0;
        contributionSum = 0.0;
        contributionCompensation = 0.0;
    }

    /** Adds one selected-branch contribution to the current path score. */
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

        if (trainingSummary == null) {
            return;
        }
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
            return;
        }

        double trainingMaximum = branchSummary.maximum();
        validateTrainingMaximum(nodeId, branch, trainingMaximum);

        summarizedNodeCount++;
        addContribution(nodeContribution(winningDistance, trainingMaximum));
    }

    /** Finalizes the tree score without resetting this scorer. */
    @Override
    public OODScoreResult finish() {
        if (visitedNodeCount == 0 || summarizedNodeCount == 0) {
            return OODScoreResult.unavailable(
                    scoreType(),
                    visitedNodeCount,
                    summarizedNodeCount
            );
        }

        double score = canonicalizeZero(
                contributionSum / visitedNodeCount
        );
        if (!Double.isFinite(score) || score < 0.0 || score > 1.0) {
            throw new IllegalStateException(
                    "Relative support exceedance produced an invalid tree score: "
                            + score + "."
            );
        }

        return OODScoreResult.available(
                scoreType(),
                score,
                visitedNodeCount,
                summarizedNodeCount
        );
    }

    public int visitedNodeCount() {
        return visitedNodeCount;
    }

    public int summarizedNodeCount() {
        return summarizedNodeCount;
    }

    private static double nodeContribution(
            double winningDistance,
            double trainingMaximum
    ) {
        if (winningDistance <= trainingMaximum) {
            return 0.0;
        }

        return checkedContribution(
                1.0 - trainingMaximum / winningDistance
        );
    }

    /** Neumaier compensated addition limits path-order roundoff. */
    private void addContribution(double value) {
        double next = contributionSum + value;
        if (Math.abs(contributionSum) >= Math.abs(value)) {
            contributionCompensation +=
                    (contributionSum - next) + value;
        } else {
            contributionCompensation +=
                    (value - next) + contributionSum;
        }
        contributionSum = next;

        double corrected = contributionSum + contributionCompensation;
        if (Double.isFinite(corrected)) {
            contributionSum = corrected;
            contributionCompensation = 0.0;
        }
    }

    private static double checkedContribution(double contribution) {
        contribution = canonicalizeZero(contribution);
        if (!Double.isFinite(contribution)
                || contribution < 0.0
                || contribution > 1.0) {
            throw new IllegalStateException(
                    "Relative support exceedance produced an invalid node contribution: "
                            + contribution + "."
            );
        }
        return contribution;
    }

    private static void validateTrainingMaximum(
            int nodeId,
            int branch,
            double trainingMaximum
    ) {
        if (!Double.isFinite(trainingMaximum) || trainingMaximum < 0.0) {
            throw new IllegalStateException(
                    "Relative support exceedance requires a finite, nonnegative "
                            + "branch training maximum, but received "
                            + trainingMaximum
                            + " for branch " + branch
                            + " at node " + nodeId + "."
            );
        }
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
        if (!Double.isFinite(winningDistance) || winningDistance < 0.0) {
            throw new IllegalArgumentException(
                    "Relative support exceedance requires a finite, nonnegative "
                            + "winning distance, but received "
                            + winningDistance
                            + " at node " + nodeId
                            + " and depth " + depth + "."
            );
        }
    }

    private static double canonicalizeZero(double value) {
        return value == 0.0 ? 0.0 : value;
    }
}
