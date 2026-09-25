package distance.elastic;

import core.AppContext;
import core.contracts.ObjectDataset;

import java.io.Serial;
import java.io.Serializable;
import java.util.Arrays;
import java.util.Random;

/**
 * Exact univariate Dynamic Time Warping using squared Euclidean local costs.
 *
 * <p>Every overload returns the accumulated squared DTW cost. A finite
 * {@code bestSoFar} is therefore interpreted in the same squared-cost units.
 * The implementation uses a Sakoe-Chiba band, two reusable rows, cell pruning,
 * and row-level early abandonment.</p>
 *
 * <p>Both {@code double[]} and {@code float[]} inputs are supported without
 * conversion or boxing. Float samples are widened before subtraction and
 * accumulation. The dynamic-programming recurrence is inherently dependent
 * across adjacent cells, so {@code AppContext.useVectorApi} does not select a
 * separate Vector API kernel for univariate DTW. The same optimized scalar
 * recurrence is used in both modes.</p>
 *
 * <p>This class is stateless and thread-safe. Input nonemptiness is required.
 * A negative window denotes an unconstrained alignment. A nonnegative window
 * is widened when necessary to include a feasible endpoint-to-endpoint path
 * for unequal-length inputs.</p>
 */
public final class DTW implements Serializable {

    @Serial
    private static final long serialVersionUID = 1L;

    public DTW() {
    }

    public double distance(
            Object first,
            Object second,
            double bestSoFar,
            int windowSize
    ) {
        if (first instanceof double[] firstValues
                && second instanceof double[] secondValues) {
            return distance(
                    firstValues,
                    secondValues,
                    bestSoFar,
                    windowSize
            );
        }

        if (first instanceof float[] firstValues
                && second instanceof float[] secondValues) {
            return distance(
                    firstValues,
                    secondValues,
                    bestSoFar,
                    windowSize
            );
        }

        throw unsupportedPair(first, second);
    }

    public double distance(
            Object first,
            Object second,
            int windowSize
    ) {
        return distance(
                first,
                second,
                Double.POSITIVE_INFINITY,
                windowSize
        );
    }

    private static double distance(
            double[] first,
            double[] second,
            double bestSoFar,
            int windowSize
    ) {
        requireNonempty(first.length, second.length);

        // Use the shorter series for columns to minimize row storage.
        if (second.length > first.length) {
            return distanceKernel(
                    first,
                    second,
                    bestSoFar,
                    windowSize
            );
        }
        return distanceKernel(
                second,
                first,
                bestSoFar,
                windowSize
        );
    }

    private static double distance(
            float[] first,
            float[] second,
            double bestSoFar,
            int windowSize
    ) {
        requireNonempty(first.length, second.length);

        if (second.length > first.length) {
            return distanceKernel(
                    first,
                    second,
                    bestSoFar,
                    windowSize
            );
        }
        return distanceKernel(
                second,
                first,
                bestSoFar,
                windowSize
        );
    }

    /**
     * Computes DTW with rows from {@code rowSeries} and columns from the
     * shorter {@code columnSeries}.
     */
    private static double distanceKernel(
            double[] rowSeries,
            double[] columnSeries,
            double bestSoFar,
            int windowSize
    ) {
        int rowCount = rowSeries.length;
        int columnCount = columnSeries.length;
        int window = resolveWindow(windowSize, rowCount, columnCount);
        double cutoff = normalizedCutoff(bestSoFar);

        double[] previous = new double[columnCount];
        double[] current = new double[columnCount];
        Arrays.fill(previous, Double.POSITIVE_INFINITY);

        for (int row = 0; row < rowCount; row++) {
            int start = Math.max(0, row - window);
            int end = Math.min(columnCount - 1, row + window);

            if (start > end) {
                return Double.POSITIVE_INFINITY;
            }

            if (start > 0) {
                current[start - 1] = Double.POSITIVE_INFINITY;
            }

            double rowMinimum = Double.POSITIVE_INFINITY;
            double rowValue = rowSeries[row];

            for (int column = start; column <= end; column++) {
                double difference = rowValue - columnSeries[column];
                double localCost = difference * difference;
                double accumulated;

                if (row == 0 && column == 0) {
                    accumulated = localCost;
                } else {
                    double diagonal = column == 0
                            ? Double.POSITIVE_INFINITY
                            : previous[column - 1];
                    double above = previous[column];
                    double left = column == start
                            ? Double.POSITIVE_INFINITY
                            : current[column - 1];
                    accumulated = localCost
                            + minimum(diagonal, above, left);
                }

                if (accumulated > cutoff) {
                    accumulated = Double.POSITIVE_INFINITY;
                } else if (accumulated < rowMinimum) {
                    rowMinimum = accumulated;
                }

                current[column] = accumulated;
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
            double bestSoFar,
            int windowSize
    ) {
        int rowCount = rowSeries.length;
        int columnCount = columnSeries.length;
        int window = resolveWindow(windowSize, rowCount, columnCount);
        double cutoff = normalizedCutoff(bestSoFar);

        double[] previous = new double[columnCount];
        double[] current = new double[columnCount];
        Arrays.fill(previous, Double.POSITIVE_INFINITY);

        for (int row = 0; row < rowCount; row++) {
            int start = Math.max(0, row - window);
            int end = Math.min(columnCount - 1, row + window);

            if (start > end) {
                return Double.POSITIVE_INFINITY;
            }

            if (start > 0) {
                current[start - 1] = Double.POSITIVE_INFINITY;
            }

            double rowMinimum = Double.POSITIVE_INFINITY;
            double rowValue = rowSeries[row];

            for (int column = start; column <= end; column++) {
                double difference = rowValue - (double) columnSeries[column];
                double localCost = difference * difference;
                double accumulated;

                if (row == 0 && column == 0) {
                    accumulated = localCost;
                } else {
                    double diagonal = column == 0
                            ? Double.POSITIVE_INFINITY
                            : previous[column - 1];
                    double above = previous[column];
                    double left = column == start
                            ? Double.POSITIVE_INFINITY
                            : current[column - 1];
                    accumulated = localCost
                            + minimum(diagonal, above, left);
                }

                if (accumulated > cutoff) {
                    accumulated = Double.POSITIVE_INFINITY;
                } else if (accumulated < rowMinimum) {
                    rowMinimum = accumulated;
                }

                current[column] = accumulated;
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

    private static double normalizedCutoff(double bestSoFar) {
        if (Double.isNaN(bestSoFar)) {
            throw new IllegalArgumentException(
                    "DTW bestSoFar cannot be NaN."
            );
        }
        return bestSoFar < 0.0 ? 0.0 : bestSoFar;
    }

    private static double minimum(
            double first,
            double second,
            double third
    ) {
        double minimum = first < second ? first : second;
        return minimum < third ? minimum : third;
    }

    private static void requireNonempty(
            int firstLength,
            int secondLength
    ) {
        if (firstLength == 0 || secondLength == 0) {
            throw new IllegalArgumentException(
                    "DTW requires two nonempty time series."
            );
        }
    }

    private static IllegalArgumentException unsupportedPair(
            Object first,
            Object second
    ) {
        return new IllegalArgumentException(
                "DTW requires matching double[] or float[] inputs. Received "
                        + typeName(first)
                        + " and "
                        + typeName(second)
                        + ". Mixed float/double pairs and boxed numeric "
                        + "arrays are not supported."
        );
    }

    private static String typeName(Object value) {
        return value == null
                ? "null"
                : value.getClass().getTypeName();
    }

    public int get_random_window(ObjectDataset dataset, Random random) {
        int upperExclusive = (AppContext.length + 1) / 4;
        return upperExclusive <= 1
                ? 0
                : random.nextInt(upperExclusive);
    }
}
