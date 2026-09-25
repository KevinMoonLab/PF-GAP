package distance.missing;

import core.contracts.ObjectDataset;

import java.io.Serial;
import java.io.Serializable;
import java.util.Objects;
import java.util.Random;

/**
 * Independent multivariate DTW-AROW for primitive dimension-major time series
 * containing missing values represented by NaN.
 *
 * <p>Each selected dimension is compared independently with the faithful
 * univariate {@link DTWAROW} implementation, and the resulting ordinary
 * DTW-AROW distances are summed. Dimension selection applies only to the outer
 * matrix axis. Selected dimension indices are never interpreted as temporal
 * indices.</p>
 *
 * <p>Both matching {@code double[][]} and matching {@code float[][]} inputs are
 * supported without slicing or whole-matrix conversion. A null selected-
 * dimension array evaluates every dimension. Unequal time lengths are allowed
 * within corresponding dimension pairs, subject to the univariate DTW-AROW
 * window rules.</p>
 *
 * <p>A finite {@code bestSoFar} is reduced by each completed component
 * distance. The remaining ordinary-distance budget is passed to the next
 * univariate DTW-AROW calculation, where it is converted to that component's
 * exact cumulative-cost cutoff. If any selected component is unavailable, the
 * independent distance is unavailable.</p>
 *
 * <p>The class contains no mutable calculation state and is safe for concurrent
 * use.</p>
 */
public final class DTWAROW_I implements Serializable {

    @Serial
    private static final long serialVersionUID = 1L;

    private final DTWAROW dtwArow;

    public DTWAROW_I() {
        dtwArow = new DTWAROW();
    }

    /** Computes unconstrained independent DTW-AROW over all dimensions. */
    public double distance(
            Object first,
            Object second
    ) {
        return distance(
                first,
                second,
                Double.POSITIVE_INFINITY,
                -1,
                null
        );
    }

    /**
     * Computes unconstrained independent DTW-AROW over all dimensions with a
     * finite aggregate bound.
     */
    public double distance(
            Object first,
            Object second,
            double bestSoFar
    ) {
        return distance(first, second, bestSoFar, -1, null);
    }

    /** Computes independent DTW-AROW over all dimensions. */
    public double distance(
            Object first,
            Object second,
            double bestSoFar,
            int windowSize
    ) {
        return distance(first, second, bestSoFar, windowSize, null);
    }

    /**
     * Computes the sum of independent DTW-AROW distances over selected
     * dimensions.
     *
     * @param first first matching {@code double[][]} or {@code float[][]} input
     * @param second second input with the same primitive matrix type
     * @param bestSoFar nonnegative aggregate ordinary-distance threshold
     * @param windowSize {@code -1} for unconstrained DTW-AROW, otherwise an
     *                   explicit nonnegative Sakoe-Chiba radius
     * @param selectedDimensions selected outer matrix rows, or null for all
     * @return summed component distance, or positive infinity if unavailable or
     *         unable to beat {@code bestSoFar}
     */
    public double distance(
            Object first,
            Object second,
            double bestSoFar,
            int windowSize,
            int[] selectedDimensions
    ) {
        validateBestSoFar(bestSoFar);
        validateWindow(windowSize);

        if (first instanceof double[][] firstValues
                && second instanceof double[][] secondValues) {
            return distance(
                    firstValues,
                    secondValues,
                    bestSoFar,
                    windowSize,
                    selectedDimensions
            );
        }

        if (first instanceof float[][] firstValues
                && second instanceof float[][] secondValues) {
            return distance(
                    firstValues,
                    secondValues,
                    bestSoFar,
                    windowSize,
                    selectedDimensions
            );
        }

        throw unsupportedPair(first, second);
    }

    /**
     * Returns whether every selected component has an available DTW-AROW
     * alignment under the requested window.
     */
    public boolean isComputable(
            Object first,
            Object second,
            int windowSize,
            int[] selectedDimensions
    ) {
        validateWindow(windowSize);

        if (first instanceof double[][] firstValues
                && second instanceof double[][] secondValues) {
            validateDimensions(firstValues, secondValues);
            int count = selectedCount(
                    firstValues.length,
                    selectedDimensions
            );
            if (count == 0) {
                return false;
            }

            for (int position = 0; position < count; position++) {
                int dimension = selectedDimension(
                        position,
                        selectedDimensions
                );
                if (Double.isInfinite(dtwArow.distance(
                        firstValues[dimension],
                        secondValues[dimension],
                        Double.POSITIVE_INFINITY,
                        windowSize
                ))) {
                    return false;
                }
            }
            return true;
        }

        if (first instanceof float[][] firstValues
                && second instanceof float[][] secondValues) {
            validateDimensions(firstValues, secondValues);
            int count = selectedCount(
                    firstValues.length,
                    selectedDimensions
            );
            if (count == 0) {
                return false;
            }

            for (int position = 0; position < count; position++) {
                int dimension = selectedDimension(
                        position,
                        selectedDimensions
                );
                if (Double.isInfinite(dtwArow.distance(
                        firstValues[dimension],
                        secondValues[dimension],
                        Double.POSITIVE_INFINITY,
                        windowSize
                ))) {
                    return false;
                }
            }
            return true;
        }

        throw unsupportedPair(first, second);
    }

    /** Checks all dimensions using the requested window. */
    public boolean isComputable(
            Object first,
            Object second,
            int windowSize
    ) {
        return isComputable(first, second, windowSize, null);
    }

    /** Checks all dimensions using unconstrained DTW-AROW. */
    public boolean isComputable(
            Object first,
            Object second
    ) {
        return isComputable(first, second, -1, null);
    }

    private double distance(
            double[][] first,
            double[][] second,
            double bestSoFar,
            int windowSize,
            int[] selectedDimensions
    ) {
        validateDimensions(first, second);
        int count = selectedCount(first.length, selectedDimensions);
        if (count == 0) {
            return Double.POSITIVE_INFINITY;
        }

        double total = 0.0;
        for (int position = 0; position < count; position++) {
            int dimension = selectedDimension(
                    position,
                    selectedDimensions
            );
            double componentDistance = dtwArow.distance(
                    first[dimension],
                    second[dimension],
                    remainingBudget(bestSoFar, total),
                    windowSize
            );

            if (Double.isInfinite(componentDistance)) {
                return Double.POSITIVE_INFINITY;
            }

            total += componentDistance;
            if (total > bestSoFar) {
                return Double.POSITIVE_INFINITY;
            }
        }

        return total;
    }

    private double distance(
            float[][] first,
            float[][] second,
            double bestSoFar,
            int windowSize,
            int[] selectedDimensions
    ) {
        validateDimensions(first, second);
        int count = selectedCount(first.length, selectedDimensions);
        if (count == 0) {
            return Double.POSITIVE_INFINITY;
        }

        double total = 0.0;
        for (int position = 0; position < count; position++) {
            int dimension = selectedDimension(
                    position,
                    selectedDimensions
            );
            double componentDistance = dtwArow.distance(
                    first[dimension],
                    second[dimension],
                    remainingBudget(bestSoFar, total),
                    windowSize
            );

            if (Double.isInfinite(componentDistance)) {
                return Double.POSITIVE_INFINITY;
            }

            total += componentDistance;
            if (total > bestSoFar) {
                return Double.POSITIVE_INFINITY;
            }
        }

        return total;
    }

    private static double remainingBudget(
            double bestSoFar,
            double accumulated
    ) {
        if (bestSoFar == Double.POSITIVE_INFINITY) {
            return Double.POSITIVE_INFINITY;
        }
        double remaining = bestSoFar - accumulated;
        return remaining < 0.0 ? 0.0 : remaining;
    }

    private static int selectedCount(
            int dimensionCount,
            int[] selectedDimensions
    ) {
        return selectedDimensions == null
                ? dimensionCount
                : selectedDimensions.length;
    }

    private static int selectedDimension(
            int position,
            int[] selectedDimensions
    ) {
        return selectedDimensions == null
                ? position
                : selectedDimensions[position];
    }

    private static void validateDimensions(
            double[][] first,
            double[][] second
    ) {
        MissingDistanceTools.validateSameRows(first, second);
        for (int dimension = 0; dimension < first.length; dimension++) {
            Objects.requireNonNull(
                    first[dimension],
                    nullDimensionMessage("first", dimension)
            );
            Objects.requireNonNull(
                    second[dimension],
                    nullDimensionMessage("second", dimension)
            );
        }
    }

    private static void validateDimensions(
            float[][] first,
            float[][] second
    ) {
        MissingDistanceTools.validateSameRows(first, second);
        for (int dimension = 0; dimension < first.length; dimension++) {
            Objects.requireNonNull(
                    first[dimension],
                    nullDimensionMessage("first", dimension)
            );
            Objects.requireNonNull(
                    second[dimension],
                    nullDimensionMessage("second", dimension)
            );
        }
    }

    public int get_random_window(
            ObjectDataset dataset,
            Random random
    ) {
        return dtwArow.get_random_window(
                Objects.requireNonNull(
                        dataset,
                        "Dataset cannot be null."
                ),
                Objects.requireNonNull(
                        random,
                        "Random cannot be null."
                )
        );
    }

    private static void validateBestSoFar(double bestSoFar) {
        if (Double.isNaN(bestSoFar) || bestSoFar < 0.0) {
            throw new IllegalArgumentException(
                    "DTWAROW_I bestSoFar must be nonnegative and not NaN. "
                            + "Received: " + bestSoFar + "."
            );
        }
    }

    private static void validateWindow(int windowSize) {
        if (windowSize < -1) {
            throw new IllegalArgumentException(
                    "windowSize must be -1 or a nonnegative integer."
            );
        }
    }

    private static String nullDimensionMessage(
            String seriesName,
            int dimension
    ) {
        return "The " + seriesName
                + " series contains a null row at dimension "
                + dimension + ".";
    }

    private static IllegalArgumentException unsupportedPair(
            Object first,
            Object second
    ) {
        return new IllegalArgumentException(
                "DTWAROW_I requires matching double[][] or float[][] inputs. "
                        + "Received " + typeName(first) + " and "
                        + typeName(second) + "."
        );
    }

    private static String typeName(Object value) {
        return value == null ? "null" : value.getClass().getTypeName();
    }
}
