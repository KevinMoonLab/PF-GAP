package ood;

/**
 * Computes a conservative relative exceedance score along one tree path.
 *
 * <p>For a finite query winning distance {@code d} and the selected branch's
 * finite training maximum {@code m}, the node contribution is:</p>
 *
 * <pre>
 * max(0, d - m) / max(m, relativeScaleFloor)
 * </pre>
 *
 * <p>A contribution of {@code 1.0}, for example, means that the query distance
 * exceeded the training maximum by an amount equal to that maximum, so the
 * query distance was twice the observed branch boundary.</p>
 *
 * <p>If the training maximum is positive infinity, the node contributes zero:
 * that branch does not establish a finite upper support boundary. If the query
 * distance is positive infinity while the training maximum is finite, its
 * unresolved excess is represented conservatively by {@code q} effective
 * standard deviations beyond the maximum. The effective standard deviation is
 * the greatest of the selected branch standard deviation, the splitter's
 * minimum positive finite branch standard deviation, and the hard scale floor.</p>
 *
 * <p>The tree score is the arithmetic mean of node contributions over every
 * visited internal node. Nodes without usable summary metadata contribute zero
 * to the numerator and remain in the denominator, making this scorer blind to
 * path length while preventing sparse metadata from amplifying a small number
 * of contributions.</p>
 *
 * <p>Instances are mutable and not thread-safe. One instance represents one
 * query-tree traversal at a time.</p>
 */
public final class RelativeSupportExceedanceOODScorer
        implements PathOODScorer {

    /** Conservative default for an unresolved positive-infinite query distance. */
    public static final double DEFAULT_INFINITY_SIGMA_MULTIPLIER = 1.0;

    /** Absolute lower bound used when the observed branch scale is zero. */
    public static final double DEFAULT_RELATIVE_SCALE_FLOOR = 1.0e-12;

    private final double relativeScaleFloor;
    private final double infinitySigmaMultiplier;

    private int visitedNodeCount;
    private int summarizedNodeCount;
    private double contributionSum;
    private double contributionCompensation;

    /** Creates a scorer with the documented default parameters. */
    public RelativeSupportExceedanceOODScorer() {
        this(
                DEFAULT_RELATIVE_SCALE_FLOOR,
                DEFAULT_INFINITY_SIGMA_MULTIPLIER
        );
    }

    /**
     * Creates a configured scorer.
     *
     * @param relativeScaleFloor positive finite denominator and fallback scale
     * @param infinitySigmaMultiplier nonnegative finite number of effective
     *                                standard deviations used for an infinite
     *                                query distance against a finite boundary
     */
    public RelativeSupportExceedanceOODScorer(
            double relativeScaleFloor,
            double infinitySigmaMultiplier
    ) {
        if (!Double.isFinite(relativeScaleFloor)
                || relativeScaleFloor <= 0.0) {
            throw new IllegalArgumentException(
                    "Relative OOD scale floor must be positive and finite, but received "
                            + relativeScaleFloor + "."
            );
        }
        if (!Double.isFinite(infinitySigmaMultiplier)
                || infinitySigmaMultiplier < 0.0) {
            throw new IllegalArgumentException(
                    "Infinity sigma multiplier must be nonnegative and finite, but received "
                            + infinitySigmaMultiplier + "."
            );
        }

        this.relativeScaleFloor = relativeScaleFloor;
        this.infinitySigmaMultiplier = infinitySigmaMultiplier;
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

    /**
     * Adds one selected-branch contribution to the current path score.
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

        summarizedNodeCount++;
        double contribution = nodeContribution(
                winningDistance,
                branchSummary,
                trainingSummary
        );
        addContribution(contribution);
    }

    /**
     * Finalizes the tree score without resetting this scorer.
     */
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
        if (!Double.isFinite(score) || score < 0.0) {
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

    public double relativeScaleFloor() {
        return relativeScaleFloor;
    }

    public double infinitySigmaMultiplier() {
        return infinitySigmaMultiplier;
    }

    public int visitedNodeCount() {
        return visitedNodeCount;
    }

    public int summarizedNodeCount() {
        return summarizedNodeCount;
    }

    private double nodeContribution(
            double winningDistance,
            DistanceDistributionSummary branchSummary,
            SplitDistanceSummary splitSummary
    ) {
        double trainingMaximum = branchSummary.maximum();
        validateTrainingMaximum(trainingMaximum);

        if (trainingMaximum == Double.POSITIVE_INFINITY) {
            return 0.0;
        }

        double denominator = Math.max(
                canonicalizeZero(trainingMaximum),
                relativeScaleFloor
        );

        if (winningDistance == Double.POSITIVE_INFINITY) {
            double effectiveStandardDeviation =
                    resolveEffectiveStandardDeviation(
                            branchSummary,
                            splitSummary
                    );
            return checkedContribution(
                    infinitySigmaMultiplier
                            * effectiveStandardDeviation
                            / denominator
            );
        }

        double exceedance = winningDistance - trainingMaximum;
        if (exceedance <= 0.0) {
            return 0.0;
        }
        return checkedContribution(exceedance / denominator);
    }

    private double resolveEffectiveStandardDeviation(
            DistanceDistributionSummary branchSummary,
            SplitDistanceSummary splitSummary
    ) {
        double effective = relativeScaleFloor;

        double branchStandardDeviation = branchSummary.standardDeviation();
        if (Double.isFinite(branchStandardDeviation)
                && branchStandardDeviation > effective) {
            effective = branchStandardDeviation;
        }

        double splitMinimum =
                splitSummary.minimumPositiveFiniteStandardDeviation();
        if (Double.isFinite(splitMinimum) && splitMinimum > effective) {
            effective = splitMinimum;
        }

        return effective;
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
        if (!Double.isFinite(contribution) || contribution < 0.0) {
            throw new IllegalStateException(
                    "Relative support exceedance produced an invalid node contribution: "
                            + contribution + "."
            );
        }
        return contribution;
    }

    private static void validateTrainingMaximum(double trainingMaximum) {
        if (Double.isNaN(trainingMaximum)
                || trainingMaximum == Double.NEGATIVE_INFINITY
                || trainingMaximum < 0.0) {
            throw new IllegalStateException(
                    "Branch training maximum must be nonnegative or positive infinity, but was "
                            + trainingMaximum + "."
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
        if (Double.isNaN(winningDistance)
                || winningDistance == Double.NEGATIVE_INFINITY
                || winningDistance < 0.0) {
            throw new IllegalArgumentException(
                    "Winning distance must be nonnegative or positive infinity, but was "
                            + winningDistance + "."
            );
        }
    }

    private static double canonicalizeZero(double value) {
        return value == 0.0 ? 0.0 : value;
    }
}
