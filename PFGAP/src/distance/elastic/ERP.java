package distance.elastic;

import core.contracts.ObjectDataset;
import distance.DistanceTools;

import java.io.Serial;
import java.io.Serializable;
import java.util.Arrays;
import java.util.Random;

/**
 * Edit Distance with Real Penalty using squared local value costs.
 *
 * <p>ERP aligns two real-valued series while allowing gaps represented by the
 * constant value {@code gValue}. This implementation returns the accumulated
 * squared ERP cost. A finite {@code bestSoFar} is interpreted in the same
 * squared-cost units.</p>
 *
 * <p>Both {@code double[]} and {@code float[]} inputs are supported without
 * conversion or boxing. Distance evaluation uses a Sakoe-Chiba band, two
 * dynamic-programming rows, finite-bound cell pruning, and safe row-level
 * early abandonment. A negative window represents an unconstrained alignment;
 * a finite window is widened when necessary to retain an endpoint-to-endpoint
 * path for unequal-length inputs.</p>
 *
 * <p>The recurrence has a left-cell dependency, so no separate Vector API
 * kernel is used.</p>
 */
public final class ERP implements Serializable {

    @Serial
    private static final long serialVersionUID = 1L;

    public ERP() {
    }

    public double distance(
            Object first,
            Object second,
            double bestSoFar,
            int windowSize,
            double gapValue
    ) {
        validateGapValue(gapValue);

        if (first instanceof double[] firstValues
                && second instanceof double[] secondValues) {
            return distance(
                    firstValues,
                    secondValues,
                    bestSoFar,
                    windowSize,
                    gapValue
            );
        }
        if (first instanceof float[] firstValues
                && second instanceof float[] secondValues) {
            return distance(
                    firstValues,
                    secondValues,
                    bestSoFar,
                    windowSize,
                    gapValue
            );
        }
        throw unsupportedPair(first, second);
    }

    private static double distance(
            double[] first,
            double[] second,
            double bestSoFar,
            int windowSize,
            double gapValue
    ) {
        requireNonempty(first.length, second.length);
        double cutoff = normalizedCutoff(bestSoFar);

        if (first.length >= second.length) {
            return distanceKernel(
                    first,
                    second,
                    cutoff,
                    windowSize,
                    gapValue
            );
        }
        return distanceKernel(
                second,
                first,
                cutoff,
                windowSize,
                gapValue
        );
    }

    private static double distance(
            float[] first,
            float[] second,
            double bestSoFar,
            int windowSize,
            double gapValue
    ) {
        requireNonempty(first.length, second.length);
        double cutoff = normalizedCutoff(bestSoFar);

        if (first.length >= second.length) {
            return distanceKernel(
                    first,
                    second,
                    cutoff,
                    windowSize,
                    gapValue
            );
        }
        return distanceKernel(
                second,
                first,
                cutoff,
                windowSize,
                gapValue
        );
    }

    private static double distanceKernel(
            double[] rowSeries,
            double[] columnSeries,
            double cutoff,
            int windowSize,
            double gapValue
    ) {
        int rowCount = rowSeries.length;
        int columnCount = columnSeries.length;
        int window = resolveWindow(windowSize, rowCount, columnCount);
        double[] previous = new double[columnCount + 1];
        double[] current = new double[columnCount + 1];
        Arrays.fill(previous, Double.POSITIVE_INFINITY);
        previous[0] = 0.0;

        int initialEnd = Math.min(columnCount, window);
        for (int column = 1; column <= initialEnd; column++) {
            previous[column] = previous[column - 1]
                    + square(columnSeries[column - 1] - gapValue);
            if (previous[column] > cutoff) {
                previous[column] = Double.POSITIVE_INFINITY;
            }
        }

        for (int row = 1; row <= rowCount; row++) {
            Arrays.fill(current, Double.POSITIVE_INFINITY);
            if (row <= window && previous[0] != Double.POSITIVE_INFINITY) {
                current[0] = previous[0]
                        + square(rowSeries[row - 1] - gapValue);
                if (current[0] > cutoff) {
                    current[0] = Double.POSITIVE_INFINITY;
                }
            }

            int start = Math.max(1, row - window);
            int end = Math.min(columnCount, row + window);
            double rowMinimum = current[0];
            double rowValue = rowSeries[row - 1];
            double deleteRowCost = square(rowValue - gapValue);

            for (int column = start; column <= end; column++) {
                double columnValue = columnSeries[column - 1];
                double match = previous[column - 1]
                        + square(rowValue - columnValue);
                double deleteRow = previous[column] + deleteRowCost;
                double deleteColumn = current[column - 1]
                        + square(columnValue - gapValue);
                double value = minimum(match, deleteRow, deleteColumn);
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

        return previous[columnCount];
    }

    private static double distanceKernel(
            float[] rowSeries,
            float[] columnSeries,
            double cutoff,
            int windowSize,
            double gapValue
    ) {
        int rowCount = rowSeries.length;
        int columnCount = columnSeries.length;
        int window = resolveWindow(windowSize, rowCount, columnCount);
        double[] previous = new double[columnCount + 1];
        double[] current = new double[columnCount + 1];
        Arrays.fill(previous, Double.POSITIVE_INFINITY);
        previous[0] = 0.0;

        int initialEnd = Math.min(columnCount, window);
        for (int column = 1; column <= initialEnd; column++) {
            previous[column] = previous[column - 1]
                    + square((double) columnSeries[column - 1] - gapValue);
            if (previous[column] > cutoff) {
                previous[column] = Double.POSITIVE_INFINITY;
            }
        }

        for (int row = 1; row <= rowCount; row++) {
            Arrays.fill(current, Double.POSITIVE_INFINITY);
            if (row <= window && previous[0] != Double.POSITIVE_INFINITY) {
                current[0] = previous[0]
                        + square((double) rowSeries[row - 1] - gapValue);
                if (current[0] > cutoff) {
                    current[0] = Double.POSITIVE_INFINITY;
                }
            }

            int start = Math.max(1, row - window);
            int end = Math.min(columnCount, row + window);
            double rowMinimum = current[0];
            double rowValue = rowSeries[row - 1];
            double deleteRowCost = square(rowValue - gapValue);

            for (int column = start; column <= end; column++) {
                double columnValue = columnSeries[column - 1];
                double match = previous[column - 1]
                        + square(rowValue - columnValue);
                double deleteRow = previous[column] + deleteRowCost;
                double deleteColumn = current[column - 1]
                        + square(columnValue - gapValue);
                double value = minimum(match, deleteRow, deleteColumn);
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

        return previous[columnCount];
    }

    private static int resolveWindow(
            int configuredWindow,
            int firstLength,
            int secondLength
    ) {
        if (configuredWindow < 0) {
            return Math.max(firstLength, secondLength);
        }
        return Math.max(
                configuredWindow,
                Math.abs(firstLength - secondLength)
        );
    }

    private static double square(double value) {
        return value * value;
    }

    private static double minimum(
            double first,
            double second,
            double third
    ) {
        double minimum = first < second ? first : second;
        return minimum < third ? minimum : third;
    }

    private static double normalizedCutoff(double bestSoFar) {
        if (Double.isNaN(bestSoFar)) {
            throw new IllegalArgumentException(
                    "ERP bestSoFar cannot be NaN."
            );
        }
        return bestSoFar < 0.0 ? 0.0 : bestSoFar;
    }

    private static void validateGapValue(double gapValue) {
        if (!Double.isFinite(gapValue)) {
            throw new IllegalArgumentException(
                    "ERP gap value must be finite."
            );
        }
    }

    private static void requireNonempty(
            int firstLength,
            int secondLength
    ) {
        if (firstLength == 0 || secondLength == 0) {
            throw new IllegalArgumentException(
                    "ERP requires two nonempty time series."
            );
        }
    }

    private static IllegalArgumentException unsupportedPair(
            Object first,
            Object second
    ) {
        return new IllegalArgumentException(
                "ERP requires matching double[] or float[] inputs. Received "
                        + typeName(first) + " and " + typeName(second) + "."
        );
    }

    private static String typeName(Object value) {
        return value == null ? "null" : value.getClass().getTypeName();
    }

    public int get_random_window(ObjectDataset dataset, Random random) {
        int upperInclusive = Math.max(0, dataset.length() / 4);
        return random.nextInt(upperInclusive + 1);
    }

    public double get_random_g(ObjectDataset dataset, Random random) {
        double standardDeviation = DistanceTools.stdv_p(dataset);
        return standardDeviation * (0.2 + 0.8 * random.nextDouble());
    }
}
