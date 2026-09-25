package distance.elastic;

import core.contracts.ObjectDataset;

import java.io.Serial;
import java.io.Serializable;
import java.util.Arrays;
import java.util.Random;

/**
 * Move-Split-Merge (MSM) distance for primitive univariate series.
 *
 * <p>The implementation follows the standard MSM dynamic-programming
 * recurrence with absolute-difference move cost and the context-sensitive
 * split/merge cost. Both {@code double[]} and {@code float[]} inputs are
 * supported without conversion or boxing.</p>
 *
 * <p>Distance evaluation uses two rows, finite-bound cell pruning, and safe
 * row-level early abandonment. MSM returns its ordinary metric cost; unlike
 * squared-Euclidean DTW kernels, its value must not be squared.</p>
 *
 * <p>The recurrence contains a left-cell dependency and the split/merge cost
 * contains data-dependent branches, so no separate Vector API kernel is used.</p>
 */
public final class MSM implements Serializable {

    @Serial
    private static final long serialVersionUID = 1L;

    private static final double[] MSM_PARAMETERS = {
            0.01, 0.01375, 0.0175, 0.02125, 0.025, 0.02875,
            0.0325, 0.03625, 0.04, 0.04375, 0.0475, 0.05125,
            0.055, 0.05875, 0.0625, 0.06625, 0.07, 0.07375,
            0.0775, 0.08125, 0.085, 0.08875, 0.0925, 0.09625,
            0.1, 0.136, 0.172, 0.208, 0.244, 0.28, 0.316,
            0.352, 0.388, 0.424, 0.46, 0.496, 0.532, 0.568,
            0.604, 0.64, 0.676, 0.712, 0.748, 0.784, 0.82,
            0.856, 0.892, 0.928, 0.964, 1.0, 1.36, 1.72,
            2.08, 2.44, 2.8, 3.16, 3.52, 3.88, 4.24, 4.6,
            4.96, 5.32, 5.68, 6.04, 6.4, 6.76, 7.12, 7.48,
            7.84, 8.2, 8.56, 8.92, 9.28, 9.64, 10.0, 13.6,
            17.2, 20.8, 24.4, 28.0, 31.6, 35.2, 38.8, 42.4,
            46.0, 49.6, 53.2, 56.8, 60.4, 64.0, 67.6, 71.2,
            74.8, 78.4, 82.0, 85.6, 89.2, 92.8, 96.4, 100.0
    };

    public MSM() {
    }

    public double distance(
            Object first,
            Object second,
            double bestSoFar,
            double cost
    ) {
        validateCost(cost);

        if (first instanceof double[] firstValues
                && second instanceof double[] secondValues) {
            return distance(firstValues, secondValues, bestSoFar, cost);
        }
        if (first instanceof float[] firstValues
                && second instanceof float[] secondValues) {
            return distance(firstValues, secondValues, bestSoFar, cost);
        }
        throw unsupportedPair(first, second);
    }

    private static double distance(
            double[] first,
            double[] second,
            double bestSoFar,
            double cost
    ) {
        requireNonempty(first.length, second.length);
        double cutoff = normalizedCutoff(bestSoFar);

        if (first.length >= second.length) {
            return distanceKernel(first, second, cutoff, cost);
        }
        return distanceKernel(second, first, cutoff, cost);
    }

    private static double distance(
            float[] first,
            float[] second,
            double bestSoFar,
            double cost
    ) {
        requireNonempty(first.length, second.length);
        double cutoff = normalizedCutoff(bestSoFar);

        if (first.length >= second.length) {
            return distanceKernel(first, second, cutoff, cost);
        }
        return distanceKernel(second, first, cutoff, cost);
    }

    private static double distanceKernel(
            double[] rowSeries,
            double[] columnSeries,
            double cutoff,
            double operationCost
    ) {
        int rowCount = rowSeries.length;
        int columnCount = columnSeries.length;
        double[] previous = new double[columnCount];
        double[] current = new double[columnCount];

        previous[0] = Math.abs(rowSeries[0] - columnSeries[0]);
        for (int column = 1; column < columnCount; column++) {
            previous[column] = previous[column - 1]
                    + operationCost(
                            columnSeries[column],
                            rowSeries[0],
                            columnSeries[column - 1],
                            operationCost
                    );
            if (previous[column] > cutoff) {
                previous[column] = Double.POSITIVE_INFINITY;
            }
        }

        if (minimum(previous) == Double.POSITIVE_INFINITY) {
            return Double.POSITIVE_INFINITY;
        }

        for (int row = 1; row < rowCount; row++) {
            current[0] = previous[0]
                    + operationCost(
                            rowSeries[row],
                            rowSeries[row - 1],
                            columnSeries[0],
                            operationCost
                    );
            if (current[0] > cutoff) {
                current[0] = Double.POSITIVE_INFINITY;
            }

            double rowMinimum = current[0];
            for (int column = 1; column < columnCount; column++) {
                double move = previous[column - 1]
                        + Math.abs(rowSeries[row] - columnSeries[column]);
                double merge = previous[column]
                        + operationCost(
                                rowSeries[row],
                                rowSeries[row - 1],
                                columnSeries[column],
                                operationCost
                        );
                double split = current[column - 1]
                        + operationCost(
                                columnSeries[column],
                                rowSeries[row],
                                columnSeries[column - 1],
                                operationCost
                        );
                double value = minimum(move, merge, split);
                if (value > cutoff) {
                    value = Double.POSITIVE_INFINITY;
                }
                current[column] = value;
                if (value < rowMinimum) {
                    rowMinimum = value;
                }
            }

            if (rowMinimum == Double.POSITIVE_INFINITY) {
                return Double.POSITIVE_INFINITY;
            }

            double[] temporary = previous;
            previous = current;
            current = temporary;
        }

        return previous[columnCount - 1];
    }

    private static double distanceKernel(
            float[] rowSeries,
            float[] columnSeries,
            double cutoff,
            double operationCost
    ) {
        int rowCount = rowSeries.length;
        int columnCount = columnSeries.length;
        double[] previous = new double[columnCount];
        double[] current = new double[columnCount];

        previous[0] = Math.abs(
                (double) rowSeries[0] - (double) columnSeries[0]
        );
        for (int column = 1; column < columnCount; column++) {
            previous[column] = previous[column - 1]
                    + operationCost(
                            columnSeries[column],
                            rowSeries[0],
                            columnSeries[column - 1],
                            operationCost
                    );
            if (previous[column] > cutoff) {
                previous[column] = Double.POSITIVE_INFINITY;
            }
        }

        if (minimum(previous) == Double.POSITIVE_INFINITY) {
            return Double.POSITIVE_INFINITY;
        }

        for (int row = 1; row < rowCount; row++) {
            current[0] = previous[0]
                    + operationCost(
                            rowSeries[row],
                            rowSeries[row - 1],
                            columnSeries[0],
                            operationCost
                    );
            if (current[0] > cutoff) {
                current[0] = Double.POSITIVE_INFINITY;
            }

            double rowMinimum = current[0];
            for (int column = 1; column < columnCount; column++) {
                double move = previous[column - 1]
                        + Math.abs(
                                (double) rowSeries[row]
                                        - (double) columnSeries[column]
                        );
                double merge = previous[column]
                        + operationCost(
                                rowSeries[row],
                                rowSeries[row - 1],
                                columnSeries[column],
                                operationCost
                        );
                double split = current[column - 1]
                        + operationCost(
                                columnSeries[column],
                                rowSeries[row],
                                columnSeries[column - 1],
                                operationCost
                        );
                double value = minimum(move, merge, split);
                if (value > cutoff) {
                    value = Double.POSITIVE_INFINITY;
                }
                current[column] = value;
                if (value < rowMinimum) {
                    rowMinimum = value;
                }
            }

            if (rowMinimum == Double.POSITIVE_INFINITY) {
                return Double.POSITIVE_INFINITY;
            }

            double[] temporary = previous;
            previous = current;
            current = temporary;
        }

        return previous[columnCount - 1];
    }

    private static double operationCost(
            double newPoint,
            double firstContext,
            double secondContext,
            double cost
    ) {
        if ((firstContext <= newPoint && newPoint <= secondContext)
                || (secondContext <= newPoint
                && newPoint <= firstContext)) {
            return cost;
        }
        return cost + Math.min(
                Math.abs(newPoint - firstContext),
                Math.abs(newPoint - secondContext)
        );
    }

    private static double minimum(
            double first,
            double second,
            double third
    ) {
        double minimum = first < second ? first : second;
        return minimum < third ? minimum : third;
    }

    private static double minimum(double[] values) {
        double minimum = Double.POSITIVE_INFINITY;
        for (double value : values) {
            if (value < minimum) {
                minimum = value;
            }
        }
        return minimum;
    }

    private static double normalizedCutoff(double bestSoFar) {
        if (Double.isNaN(bestSoFar)) {
            throw new IllegalArgumentException(
                    "MSM bestSoFar cannot be NaN."
            );
        }
        return bestSoFar < 0.0 ? 0.0 : bestSoFar;
    }

    private static void validateCost(double cost) {
        if (!Double.isFinite(cost) || cost <= 0.0) {
            throw new IllegalArgumentException(
                    "MSM operation cost must be finite and positive."
            );
        }
    }

    private static void requireNonempty(
            int firstLength,
            int secondLength
    ) {
        if (firstLength == 0 || secondLength == 0) {
            throw new IllegalArgumentException(
                    "MSM requires two nonempty time series."
            );
        }
    }

    private static IllegalArgumentException unsupportedPair(
            Object first,
            Object second
    ) {
        return new IllegalArgumentException(
                "MSM requires matching double[] or float[] inputs. Received "
                        + typeName(first) + " and " + typeName(second) + "."
        );
    }

    private static String typeName(Object value) {
        return value == null ? "null" : value.getClass().getTypeName();
    }

    public double get_random_cost(ObjectDataset dataset, Random random) {
        return MSM_PARAMETERS[random.nextInt(MSM_PARAMETERS.length)];
    }
}
