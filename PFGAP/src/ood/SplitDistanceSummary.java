package ood;

import java.io.Serial;
import java.io.Serializable;
import java.util.Arrays;
import java.util.Objects;

/**
 * Immutable compact branch-local distance summaries retained by one winning
 * tree splitter.
 *
 * <p>Each array position corresponds to one splitter exemplar and child branch.
 * The summary at branch {@code b} describes the bootstrap-weighted winning
 * distances of the distinct training observations routed to branch {@code b}.
 * No pooled splitter-wide distribution, raw distances, or observation-level
 * state is retained.</p>
 *
 * <p>The minimum positive finite standard deviation across nonempty branches is
 * calculated once during construction and cached. A relative-support scorer
 * does not need this value for ordinary finite distances, but may use it when a
 * test winning distance is positive infinity and the selected branch has zero
 * finite variation.</p>
 */
public final class SplitDistanceSummary implements Serializable {

    @Serial
    private static final long serialVersionUID = 3L;

    private final DistanceDistributionSummary[] branches;
    private final double minimumPositiveFiniteStandardDeviation;

    /**
     * Creates a branch-local winning-split summary.
     *
     * @param branches summary for each branch in splitter branch order; empty
     *                 branches must use {@link DistanceDistributionSummary#empty()}
     */
    public SplitDistanceSummary(
            DistanceDistributionSummary[] branches
    ) {
        Objects.requireNonNull(
                branches,
                "Branch distance summaries cannot be null."
        );
        if (branches.length == 0) {
            throw new IllegalArgumentException(
                    "At least one branch distance summary is required."
            );
        }

        this.branches = branches.clone();
        validateBranchEntries(this.branches);
        this.minimumPositiveFiniteStandardDeviation =
                findMinimumPositiveFiniteStandardDeviation(this.branches);
    }

    /** Returns the number of splitter branches represented by this summary. */
    public int branchCount() {
        return branches.length;
    }

    /**
     * Returns the immutable summary for one branch.
     *
     * @param branch zero-based splitter branch index
     * @return branch summary, which may be empty
     */
    public DistanceDistributionSummary branch(int branch) {
        checkBranchIndex(branch);
        return branches[branch];
    }

    /** Returns a defensive copy of the branch-summary reference array. */
    public DistanceDistributionSummary[] branches() {
        return branches.clone();
    }

    /** Returns whether the branch contains at least one summarized distance. */
    public boolean hasBranchSummary(int branch) {
        return !branch(branch).isEmpty();
    }

    /**
     * Returns whether the branch has at least one finite training distance.
     */
    public boolean hasFiniteBranchDistances(int branch) {
        return branch(branch).hasFiniteDistances();
    }

    /**
     * Returns whether the branch establishes a finite empirical upper support
     * boundary for relative-support exceedance scoring.
     */
    public boolean hasFiniteBranchUpperSupportBoundary(int branch) {
        return branch(branch).hasFiniteUpperSupportBoundary();
    }

    /**
     * Returns whether at least one branch has a positive finite standard
     * deviation.
     */
    public boolean hasPositiveFiniteStandardDeviation() {
        return Double.isFinite(minimumPositiveFiniteStandardDeviation);
    }

    /**
     * Returns the minimum strictly positive finite standard deviation across
     * all nonempty branches.
     *
     * <p>Returns {@link Double#NaN} when no branch has positive finite
     * variation. Scorers may then apply their documented hard scale floor.</p>
     */
    public double minimumPositiveFiniteStandardDeviation() {
        return minimumPositiveFiniteStandardDeviation;
    }

    private void checkBranchIndex(int branch) {
        if (branch < 0 || branch >= branches.length) {
            throw new IndexOutOfBoundsException(
                    "Branch index " + branch + " is outside [0, "
                            + branches.length + ")."
            );
        }
    }

    private static void validateBranchEntries(
            DistanceDistributionSummary[] branches
    ) {
        for (int branch = 0; branch < branches.length; branch++) {
            if (branches[branch] == null) {
                throw new IllegalArgumentException(
                        "Branch distance summary cannot be null at index "
                                + branch + ". Use "
                                + "DistanceDistributionSummary.empty() for an "
                                + "empty branch."
                );
            }
        }
    }

    private static double findMinimumPositiveFiniteStandardDeviation(
            DistanceDistributionSummary[] branches
    ) {
        double minimum = Double.POSITIVE_INFINITY;

        for (DistanceDistributionSummary branch : branches) {
            if (branch.isEmpty() || !branch.hasFiniteDistances()) {
                continue;
            }

            double standardDeviation = branch.standardDeviation();
            if (Double.isFinite(standardDeviation)
                    && standardDeviation > 0.0
                    && standardDeviation < minimum) {
                minimum = standardDeviation;
            }
        }

        return minimum == Double.POSITIVE_INFINITY
                ? Double.NaN
                : canonicalizeZero(minimum);
    }

    private static double canonicalizeZero(double value) {
        return value == 0.0 ? 0.0 : value;
    }

    @Override
    public boolean equals(Object other) {
        if (this == other) {
            return true;
        }
        if (!(other instanceof SplitDistanceSummary summary)) {
            return false;
        }
        return Arrays.equals(branches, summary.branches)
                && Double.compare(
                        minimumPositiveFiniteStandardDeviation,
                        summary.minimumPositiveFiniteStandardDeviation
                ) == 0;
    }

    @Override
    public int hashCode() {
        int result = Arrays.hashCode(branches);
        result = 31 * result
                + Double.hashCode(minimumPositiveFiniteStandardDeviation);
        return result;
    }

    @Override
    public String toString() {
        return "SplitDistanceSummary{"
                + "branches=" + Arrays.toString(branches)
                + ", minimumPositiveFiniteStandardDeviation="
                + minimumPositiveFiniteStandardDeviation
                + '}';
    }
}
