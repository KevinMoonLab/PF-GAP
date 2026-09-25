package distance.elastic;

import core.AppContext;
import core.contracts.ObjectDataset;
import distance.DistanceTools;

import java.io.Serial;
import java.io.Serializable;
import java.util.Arrays;
import java.util.Random;

/**
 * Longest Common Subsequence distance for primitive univariate series.
 *
 * <p>Two values match when their absolute difference is at most
 * {@code epsilon}. The similarity is the constrained LCSS length divided by
 * the shorter input length, and the returned distance is
 * {@code 1.0 - similarity}, therefore lying in {@code [0, 1]}.</p>
 *
 * <p>Both {@code double[]} and {@code float[]} inputs are supported without
 * conversion or boxing. Evaluation uses a Sakoe-Chiba band and two integer
 * rows. A negative window means unconstrained alignment, while a finite window
 * is widened when necessary for unequal input lengths.</p>
 *
 * <p>The {@code bestSoFar} parameter is accepted for the common distance
 * contract but is not used for cell-level early abandonment. LCSS maximizes a
 * subsequence length, so an intermediate low score does not provide the same
 * monotone cutoff as additive-cost elastic distances.</p>
 */
public final class LCSS implements Serializable {

    @Serial
    private static final long serialVersionUID = 1L;

    public LCSS() {
    }

    public double distance(
            Object first,
            Object second,
            double bestSoFar,
            int windowSize,
            double epsilon
    ) {
        validateEpsilon(epsilon);

        if (first instanceof double[] firstValues
                && second instanceof double[] secondValues) {
            return distance(firstValues, secondValues, windowSize, epsilon);
        }
        if (first instanceof float[] firstValues
                && second instanceof float[] secondValues) {
            return distance(firstValues, secondValues, windowSize, epsilon);
        }
        throw unsupportedPair(first, second);
    }

    private static double distance(
            double[] first,
            double[] second,
            int windowSize,
            double epsilon
    ) {
        requireNonempty(first.length, second.length);
        if (first.length >= second.length) {
            return distanceKernel(first, second, windowSize, epsilon);
        }
        return distanceKernel(second, first, windowSize, epsilon);
    }

    private static double distance(
            float[] first,
            float[] second,
            int windowSize,
            double epsilon
    ) {
        requireNonempty(first.length, second.length);
        if (first.length >= second.length) {
            return distanceKernel(first, second, windowSize, epsilon);
        }
        return distanceKernel(second, first, windowSize, epsilon);
    }

    private static double distanceKernel(
            double[] rowSeries,
            double[] columnSeries,
            int windowSize,
            double epsilon
    ) {
        int rowCount = rowSeries.length;
        int columnCount = columnSeries.length;
        int window = resolveWindow(windowSize, rowCount, columnCount);
        int[] previous = new int[columnCount + 1];
        int[] current = new int[columnCount + 1];

        for (int row = 1; row <= rowCount; row++) {
            Arrays.fill(current, 0);
            int start = Math.max(1, row - window);
            int end = Math.min(columnCount, row + window);
            double rowValue = rowSeries[row - 1];

            for (int column = start; column <= end; column++) {
                if (Math.abs(
                        rowValue - columnSeries[column - 1]
                ) <= epsilon) {
                    current[column] = previous[column - 1] + 1;
                } else {
                    current[column] = Math.max(
                            previous[column],
                            current[column - 1]
                    );
                }
            }

            int[] temporary = previous;
            previous = current;
            current = temporary;
        }

        return 1.0 - (double) previous[columnCount] / columnCount;
    }

    private static double distanceKernel(
            float[] rowSeries,
            float[] columnSeries,
            int windowSize,
            double epsilon
    ) {
        int rowCount = rowSeries.length;
        int columnCount = columnSeries.length;
        int window = resolveWindow(windowSize, rowCount, columnCount);
        int[] previous = new int[columnCount + 1];
        int[] current = new int[columnCount + 1];

        for (int row = 1; row <= rowCount; row++) {
            Arrays.fill(current, 0);
            int start = Math.max(1, row - window);
            int end = Math.min(columnCount, row + window);
            double rowValue = rowSeries[row - 1];

            for (int column = start; column <= end; column++) {
                if (Math.abs(
                        rowValue - (double) columnSeries[column - 1]
                ) <= epsilon) {
                    current[column] = previous[column - 1] + 1;
                } else {
                    current[column] = Math.max(
                            previous[column],
                            current[column - 1]
                    );
                }
            }

            int[] temporary = previous;
            previous = current;
            current = temporary;
        }

        return 1.0 - (double) previous[columnCount] / columnCount;
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

    private static void validateEpsilon(double epsilon) {
        if (!Double.isFinite(epsilon) || epsilon < 0.0) {
            throw new IllegalArgumentException(
                    "LCSS epsilon must be finite and nonnegative."
            );
        }
    }

    private static void requireNonempty(
            int firstLength,
            int secondLength
    ) {
        if (firstLength == 0 || secondLength == 0) {
            throw new IllegalArgumentException(
                    "LCSS requires two nonempty time series."
            );
        }
    }

    private static IllegalArgumentException unsupportedPair(
            Object first,
            Object second
    ) {
        return new IllegalArgumentException(
                "LCSS requires matching double[] or float[] inputs. Received "
                        + typeName(first) + " and " + typeName(second) + "."
        );
    }

    private static String typeName(Object value) {
        return value == null ? "null" : value.getClass().getTypeName();
    }

    public int get_random_window(ObjectDataset dataset, Random random) {
        int upperExclusive = (AppContext.length + 1) / 4;
        return upperExclusive <= 1
                ? 0
                : random.nextInt(upperExclusive);
    }

    public double get_random_epsilon(
            ObjectDataset dataset,
            Random random
    ) {
        double standardDeviation = DistanceTools.stdv_p(dataset);
        double floor = standardDeviation * 0.2;
        return floor + random.nextDouble()
                * (standardDeviation - floor);
    }
}
