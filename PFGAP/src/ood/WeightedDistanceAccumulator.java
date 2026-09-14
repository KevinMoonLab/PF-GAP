package ood;

/**
 * Mutable constant-memory accumulator for bootstrap-weighted winning-distance
 * summaries.
 *
 * <p>This class is public because training components in other packages, such
 * as tree splitters, construct and update it. It is temporary training
 * machinery rather than serialized model state.</p>
 *
 * <p>A sequential candidate evaluation should own one accumulator for the
 * complete split and one per branch. A parallel assignment range should own an
 * equivalent worker-local set and merge it later in a fixed logical range
 * order. Instances are mutable and are not thread-safe; they must not be shared
 * by concurrent workers.</p>
 *
 * <p>Finite distances are accumulated with the weighted Welford algorithm.
 * Positive infinity is counted explicitly but is never entered into the finite
 * mean or second-moment calculation. NaN and negative infinity are invalid.</p>
 *
 * <p>The running {@code finiteM2} value is temporary. It is used only to
 * calculate the finite-distance population standard deviation when
 * {@link #finish()} is called. It is neither exposed nor persisted by
 * {@link DistanceDistributionSummary}.</p>
 */
public final class WeightedDistanceAccumulator {

    private long distinctCount;
    private long weightedCount;

    private long positiveInfinityDistinctCount;
    private long positiveInfinityWeightedCount;

    private long finiteDistinctCount;
    private long finiteWeightedCount;
    private double finiteMean;
    private double finiteM2;
    private double finiteMinimum = Double.POSITIVE_INFINITY;
    private double finiteMaximum = Double.NEGATIVE_INFINITY;

    /**
     * Adds one distinct winning distance with its bootstrap multiplicity.
     *
     * @param distance finite or positive-infinite winning distance
     * @param weight positive bootstrap multiplicity represented by the distance
     */
    public void add(double distance, int weight) {
        if (Double.isNaN(distance)) {
            throw new IllegalArgumentException(
                    "Winning distance cannot be NaN."
            );
        }
        if (distance == Double.NEGATIVE_INFINITY) {
            throw new IllegalArgumentException(
                    "Winning distance cannot be negative infinity."
            );
        }
        if (weight <= 0) {
            throw new IllegalArgumentException(
                    "Distance weight must be positive, but received "
                            + weight + "."
            );
        }

        long nextDistinctCount = Math.addExact(distinctCount, 1L);
        long nextWeightedCount = Math.addExact(weightedCount, (long) weight);

        if (distance == Double.POSITIVE_INFINITY) {
            positiveInfinityDistinctCount = Math.addExact(
                    positiveInfinityDistinctCount,
                    1L
            );
            positiveInfinityWeightedCount = Math.addExact(
                    positiveInfinityWeightedCount,
                    (long) weight
            );
            distinctCount = nextDistinctCount;
            weightedCount = nextWeightedCount;
            return;
        }

        addFinite(distance, weight);
        distinctCount = nextDistinctCount;
        weightedCount = nextWeightedCount;
    }

    /**
     * Merges another accumulator without replaying individual distances.
     *
     * <p>The finite-distance state is combined using the parallel weighted
     * variance identity. Positive-infinity counts are combined separately.
     * For reproducible floating-point results, callers should merge worker-local
     * accumulators in a deterministic logical range order.</p>
     *
     * @param other accumulator to merge; it is not modified
     */
    public void merge(WeightedDistanceAccumulator other) {
        if (other == null || other.isEmpty()) {
            return;
        }
        if (other == this) {
            throw new IllegalArgumentException(
                    "A distance accumulator cannot be merged with itself."
            );
        }
        if (isEmpty()) {
            copyFrom(other);
            return;
        }

        long mergedDistinctCount = Math.addExact(
                distinctCount,
                other.distinctCount
        );
        long mergedWeightedCount = Math.addExact(
                weightedCount,
                other.weightedCount
        );
        long mergedInfinityDistinctCount = Math.addExact(
                positiveInfinityDistinctCount,
                other.positiveInfinityDistinctCount
        );
        long mergedInfinityWeightedCount = Math.addExact(
                positiveInfinityWeightedCount,
                other.positiveInfinityWeightedCount
        );

        mergeFiniteState(other);

        distinctCount = mergedDistinctCount;
        weightedCount = mergedWeightedCount;
        positiveInfinityDistinctCount = mergedInfinityDistinctCount;
        positiveInfinityWeightedCount = mergedInfinityWeightedCount;
    }

    /**
     * Finalizes this accumulator into immutable persisted state.
     *
     * <p>The standard deviation is the bootstrap-weighted population standard
     * deviation of finite distances only. If no finite distance was observed,
     * the finite mean, standard deviation, and minimum are NaN. The maximum is
     * positive infinity whenever at least one positive-infinite distance was
     * observed.</p>
     *
     * @return immutable distance-distribution summary
     */
    public DistanceDistributionSummary finish() {
        if (isEmpty()) {
            return DistanceDistributionSummary.empty();
        }

        double mean;
        double standardDeviation;
        double minimum;
        double maximum;

        if (finiteWeightedCount == 0L) {
            mean = Double.NaN;
            standardDeviation = Double.NaN;
            minimum = Double.NaN;
            maximum = Double.POSITIVE_INFINITY;
        } else {
            double nonnegativeM2 = normalizeFiniteM2(finiteM2);
            mean = canonicalizeZero(finiteMean);
            standardDeviation = canonicalizeZero(
                    Math.sqrt(nonnegativeM2 / finiteWeightedCount)
            );
            minimum = canonicalizeZero(finiteMinimum);
            maximum = hasPositiveInfinity()
                    ? Double.POSITIVE_INFINITY
                    : canonicalizeZero(finiteMaximum);
        }

        return new DistanceDistributionSummary(
                distinctCount,
                weightedCount,
                positiveInfinityDistinctCount,
                positiveInfinityWeightedCount,
                mean,
                standardDeviation,
                minimum,
                maximum
        );
    }

    public boolean isEmpty() {
        return distinctCount == 0L;
    }

    public boolean hasFiniteDistances() {
        return finiteDistinctCount > 0L;
    }

    public boolean hasPositiveInfinity() {
        return positiveInfinityDistinctCount > 0L;
    }

    public long distinctCount() {
        return distinctCount;
    }

    public long weightedCount() {
        return weightedCount;
    }

    public long finiteDistinctCount() {
        return finiteDistinctCount;
    }

    public long finiteWeightedCount() {
        return finiteWeightedCount;
    }

    public long positiveInfinityDistinctCount() {
        return positiveInfinityDistinctCount;
    }

    public long positiveInfinityWeightedCount() {
        return positiveInfinityWeightedCount;
    }

    private void addFinite(double distance, int weight) {
        long nextFiniteDistinctCount = Math.addExact(finiteDistinctCount, 1L);
        long nextFiniteWeightedCount = Math.addExact(
                finiteWeightedCount,
                (long) weight
        );

        if (finiteWeightedCount == 0L) {
            finiteDistinctCount = nextFiniteDistinctCount;
            finiteWeightedCount = nextFiniteWeightedCount;
            finiteMean = canonicalizeZero(distance);
            finiteM2 = 0.0;
            finiteMinimum = distance;
            finiteMaximum = distance;
            return;
        }

        double previousWeight = finiteWeightedCount;
        double incomingWeight = weight;
        double combinedWeight = nextFiniteWeightedCount;
        double delta = distance - finiteMean;

        finiteMean += delta * incomingWeight / combinedWeight;
        finiteM2 += delta * delta
                * previousWeight
                * incomingWeight
                / combinedWeight;

        finiteDistinctCount = nextFiniteDistinctCount;
        finiteWeightedCount = nextFiniteWeightedCount;
        finiteMinimum = Math.min(finiteMinimum, distance);
        finiteMaximum = Math.max(finiteMaximum, distance);
    }

    private void mergeFiniteState(WeightedDistanceAccumulator other) {
        if (!other.hasFiniteDistances()) {
            return;
        }
        if (!hasFiniteDistances()) {
            finiteDistinctCount = other.finiteDistinctCount;
            finiteWeightedCount = other.finiteWeightedCount;
            finiteMean = other.finiteMean;
            finiteM2 = other.finiteM2;
            finiteMinimum = other.finiteMinimum;
            finiteMaximum = other.finiteMaximum;
            return;
        }

        long mergedFiniteDistinctCount = Math.addExact(
                finiteDistinctCount,
                other.finiteDistinctCount
        );
        long mergedFiniteWeightedCount = Math.addExact(
                finiteWeightedCount,
                other.finiteWeightedCount
        );

        double firstWeight = finiteWeightedCount;
        double secondWeight = other.finiteWeightedCount;
        double combinedWeight = mergedFiniteWeightedCount;
        double delta = other.finiteMean - finiteMean;

        finiteMean += delta * secondWeight / combinedWeight;
        finiteM2 += other.finiteM2
                + delta * delta
                * firstWeight
                * secondWeight
                / combinedWeight;

        finiteDistinctCount = mergedFiniteDistinctCount;
        finiteWeightedCount = mergedFiniteWeightedCount;
        finiteMinimum = Math.min(finiteMinimum, other.finiteMinimum);
        finiteMaximum = Math.max(finiteMaximum, other.finiteMaximum);
    }

    private void copyFrom(WeightedDistanceAccumulator other) {
        distinctCount = other.distinctCount;
        weightedCount = other.weightedCount;
        positiveInfinityDistinctCount = other.positiveInfinityDistinctCount;
        positiveInfinityWeightedCount = other.positiveInfinityWeightedCount;
        finiteDistinctCount = other.finiteDistinctCount;
        finiteWeightedCount = other.finiteWeightedCount;
        finiteMean = other.finiteMean;
        finiteM2 = other.finiteM2;
        finiteMinimum = other.finiteMinimum;
        finiteMaximum = other.finiteMaximum;
    }

    /**
     * Corrects only a tiny negative value attributable to floating-point
     * roundoff. A materially negative value indicates corrupted accumulator
     * state and is not silently hidden.
     */
    private static double normalizeFiniteM2(double value) {
        if (value >= 0.0) {
            return value;
        }

        double tolerance = 32.0 * Math.ulp(Math.max(1.0, Math.abs(value)));
        if (value >= -tolerance) {
            return 0.0;
        }

        throw new IllegalStateException(
                "Finite weighted-distance M2 became negative: " + value + "."
        );
    }

    private static double canonicalizeZero(double value) {
        return value == 0.0 ? 0.0 : value;
    }
}
