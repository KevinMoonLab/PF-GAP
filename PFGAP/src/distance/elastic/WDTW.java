package distance.elastic;

import core.contracts.ObjectDataset;

import java.io.Serial;
import java.io.Serializable;
import java.util.Arrays;
import java.util.Random;

/**
 * Weighted Dynamic Time Warping with squared Euclidean local costs.
 *
 * <p>The logistic weight for a warp displacement {@code k} is</p>
 *
 * <pre>
 * 1 / (1 + exp(-g * (k - maxLength / 2)))
 * </pre>
 *
 * <p>Every overload returns accumulated weighted squared cost. A finite
 * {@code bestSoFar} is interpreted in those same units. Both {@code double[]}
 * and {@code float[]} inputs are supported without conversion or boxing.</p>
 *
 * <p>Distance-only evaluation uses two rows, cell pruning, and row-level early
 * abandonment. The shorter input is used for columns to minimize temporary
 * storage. Logistic weights are cached as an immutable profile. Concurrent
 * calls may harmlessly construct the same profile more than once during a
 * cache miss, but never observe a partially initialized profile.</p>
 *
 * <p>The dynamic-programming recurrence has a left-cell dependency, so a
 * separate Vector API kernel is not used. Enabling {@code AppContext.useVectorApi}
 * therefore does not alter this implementation.</p>
 */
public final class WDTW implements Serializable {

    @Serial
    private static final long serialVersionUID = 1L;

    private static final double WEIGHT_MAX = 1.0;

    private transient volatile WeightProfile cachedProfile;

    public WDTW() {
    }

    public double distance(
            Object first,
            Object second,
            double bestSoFar,
            double g
    ) {
        if (first instanceof double[] firstValues
                && second instanceof double[] secondValues) {
            return distance(
                    firstValues,
                    secondValues,
                    bestSoFar,
                    g
            );
        }

        if (first instanceof float[] firstValues
                && second instanceof float[] secondValues) {
            return distance(
                    firstValues,
                    secondValues,
                    bestSoFar,
                    g
            );
        }

        throw unsupportedPair(first, second);
    }

    public double distance(
            Object first,
            Object second,
            double g
    ) {
        return distance(
                first,
                second,
                Double.POSITIVE_INFINITY,
                g
        );
    }

    private double distance(
            double[] first,
            double[] second,
            double bestSoFar,
            double g
    ) {
        requireNonempty(first.length, second.length);
        double[] weights = weights(g, Math.max(first.length, second.length));

        if (first.length >= second.length) {
            return distanceKernel(
                    first,
                    second,
                    bestSoFar,
                    weights
            );
        }
        return distanceKernel(
                second,
                first,
                bestSoFar,
                weights
        );
    }

    private double distance(
            float[] first,
            float[] second,
            double bestSoFar,
            double g
    ) {
        requireNonempty(first.length, second.length);
        double[] weights = weights(g, Math.max(first.length, second.length));

        if (first.length >= second.length) {
            return distanceKernel(
                    first,
                    second,
                    bestSoFar,
                    weights
            );
        }
        return distanceKernel(
                second,
                first,
                bestSoFar,
                weights
        );
    }

    private static double distanceKernel(
            double[] rowSeries,
            double[] columnSeries,
            double bestSoFar,
            double[] weights
    ) {
        int rowCount = rowSeries.length;
        int columnCount = columnSeries.length;
        double cutoff = normalizedCutoff(bestSoFar);

        double[] previous = new double[columnCount];
        double[] current = new double[columnCount];
        Arrays.fill(previous, Double.POSITIVE_INFINITY);

        for (int row = 0; row < rowCount; row++) {
            double rowMinimum = Double.POSITIVE_INFINITY;
            double rowValue = rowSeries[row];

            for (int column = 0; column < columnCount; column++) {
                double predecessor;
                if (row == 0 && column == 0) {
                    predecessor = 0.0;
                } else {
                    double diagonal = column == 0
                            ? Double.POSITIVE_INFINITY
                            : previous[column - 1];
                    double above = previous[column];
                    double left = column == 0
                            ? Double.POSITIVE_INFINITY
                            : current[column - 1];
                    predecessor = minimum(diagonal, above, left);
                }

                double accumulated;
                if (predecessor == Double.POSITIVE_INFINITY) {
                    accumulated = Double.POSITIVE_INFINITY;
                } else {
                    double difference = rowValue - columnSeries[column];
                    double localCost = weights[Math.abs(row - column)]
                            * difference * difference;
                    accumulated = predecessor + localCost;
                    if (accumulated > cutoff) {
                        accumulated = Double.POSITIVE_INFINITY;
                    }
                }

                current[column] = accumulated;
                if (accumulated < rowMinimum) {
                    rowMinimum = accumulated;
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
            double bestSoFar,
            double[] weights
    ) {
        int rowCount = rowSeries.length;
        int columnCount = columnSeries.length;
        double cutoff = normalizedCutoff(bestSoFar);

        double[] previous = new double[columnCount];
        double[] current = new double[columnCount];
        Arrays.fill(previous, Double.POSITIVE_INFINITY);

        for (int row = 0; row < rowCount; row++) {
            double rowMinimum = Double.POSITIVE_INFINITY;
            double rowValue = rowSeries[row];

            for (int column = 0; column < columnCount; column++) {
                double predecessor;
                if (row == 0 && column == 0) {
                    predecessor = 0.0;
                } else {
                    double diagonal = column == 0
                            ? Double.POSITIVE_INFINITY
                            : previous[column - 1];
                    double above = previous[column];
                    double left = column == 0
                            ? Double.POSITIVE_INFINITY
                            : current[column - 1];
                    predecessor = minimum(diagonal, above, left);
                }

                double accumulated;
                if (predecessor == Double.POSITIVE_INFINITY) {
                    accumulated = Double.POSITIVE_INFINITY;
                } else {
                    double difference = rowValue
                            - (double) columnSeries[column];
                    double localCost = weights[Math.abs(row - column)]
                            * difference * difference;
                    accumulated = predecessor + localCost;
                    if (accumulated > cutoff) {
                        accumulated = Double.POSITIVE_INFINITY;
                    }
                }

                current[column] = accumulated;
                if (accumulated < rowMinimum) {
                    rowMinimum = accumulated;
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

    private double[] weights(double g, int maximumLength) {
        if (!Double.isFinite(g) || g < 0.0) {
            throw new IllegalArgumentException(
                    "WDTW g must be finite and nonnegative, but received "
                            + g
                            + "."
            );
        }

        WeightProfile profile = cachedProfile;
        if (profile != null
                && Double.doubleToLongBits(profile.g())
                == Double.doubleToLongBits(g)
                && profile.weights().length == maximumLength) {
            return profile.weights();
        }

        double[] generated = new double[maximumLength];
        double midpoint = maximumLength / 2.0;
        for (int displacement = 0;
             displacement < maximumLength;
             displacement++) {
            generated[displacement] = WEIGHT_MAX
                    / (1.0 + Math.exp(
                            -g * (displacement - midpoint)
                    ));
        }

        cachedProfile = new WeightProfile(g, generated);
        return generated;
    }

    private static double normalizedCutoff(double bestSoFar) {
        if (Double.isNaN(bestSoFar)) {
            throw new IllegalArgumentException(
                    "WDTW bestSoFar cannot be NaN."
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
                    "WDTW requires two nonempty time series."
            );
        }
    }

    private static IllegalArgumentException unsupportedPair(
            Object first,
            Object second
    ) {
        return new IllegalArgumentException(
                "WDTW requires matching double[] or float[] inputs. Received "
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

    public double get_random_g(ObjectDataset dataset, Random random) {
        return random.nextDouble();
    }

    private record WeightProfile(
            double g,
            double[] weights
    ) {
    }
}
