package distance;

/**
 * Reusable allocation-free output carrier for nearest-exemplar routing.
 *
 * <p>A caller creates one instance and reuses it across observations. Sequential
 * candidate evaluation should own one carrier. Parallel evaluation should own
 * one carrier per worker range. Instances are mutable and are not thread-safe.</p>
 *
 * <p>The carrier is intentionally not serializable. It represents temporary
 * evaluation state, not trained model state.</p>
 *
 * <p>A result is valid only after {@link DistanceMeasure} has populated it. A
 * finite distance or {@link Double#POSITIVE_INFINITY} is permitted. NaN and
 * negative infinity are invalid.</p>
 */
public final class ClosestBranchResult {

    private int branch = -1;
    private double distance = Double.NaN;

    /**
     * Creates an unset reusable result carrier.
     */
    public ClosestBranchResult() {
    }

    /**
     * Returns whether the carrier currently contains a populated result.
     */
    public boolean isSet() {
        return branch >= 0;
    }

    /**
     * Returns the selected zero-based branch index.
     *
     * @throws IllegalStateException if no result has been populated
     */
    public int branch() {
        requireSet();
        return branch;
    }

    /**
     * Returns the winning distance.
     *
     * <p>Positive infinity is a permitted result. Callers that accumulate
     * statistics must count it explicitly rather than entering it into finite
     * moment calculations.</p>
     *
     * @throws IllegalStateException if no result has been populated
     */
    public double distance() {
        requireSet();
        return distance;
    }

    /**
     * Clears this carrier before optional reuse or defensive error handling.
     *
     * <p>Normal hot loops do not need to call this between successful writes;
     * {@link DistanceMeasure} overwrites both fields atomically with respect to
     * the owning thread.</p>
     */
    public void clear() {
        branch = -1;
        distance = Double.NaN;
    }

    /**
     * Populates this carrier.
     *
     * <p>Package-private ownership prevents callers outside the distance package
     * from fabricating routing results while allowing {@link DistanceMeasure}
     * to reuse the carrier without allocation.</p>
     */
    void set(int branch, double distance) {
        if (branch < 0) {
            throw new IllegalArgumentException(
                    "Closest branch cannot be negative, but received "
                            + branch + "."
            );
        }
        if (Double.isNaN(distance)) {
            throw new IllegalArgumentException(
                    "Closest-branch distance cannot be NaN."
            );
        }
        if (distance == Double.NEGATIVE_INFINITY) {
            throw new IllegalArgumentException(
                    "Closest-branch distance cannot be negative infinity."
            );
        }

        this.branch = branch;
        this.distance = distance == 0.0 ? 0.0 : distance;
    }

    private void requireSet() {
        if (!isSet()) {
            throw new IllegalStateException(
                    "ClosestBranchResult has not been populated."
            );
        }
    }

    @Override
    public String toString() {
        if (!isSet()) {
            return "ClosestBranchResult{unset}";
        }
        return "ClosestBranchResult{"
                + "branch=" + branch
                + ", distance=" + distance
                + '}';
    }
}
