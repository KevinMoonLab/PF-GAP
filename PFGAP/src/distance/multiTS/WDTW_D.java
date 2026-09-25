package distance.multiTS;

import java.io.Serial;
import java.io.Serializable;
import java.util.Arrays;

/**
 * Dependent multivariate Weighted Dynamic Time Warping.
 *
 * <p>One shared warping path is learned across all selected channels. Local
 * costs are weighted squared Euclidean costs, and the returned value is the
 * accumulated weighted squared cost. Both {@code double[][]} and
 * {@code float[][]} dimension-major inputs are supported without conversion
 * or slicing.</p>
 *
 * <p>Distance evaluation uses two rows, immutable cached logistic weights,
 * cell pruning, row-level early abandonment, and dimension-loop abandonment
 * against each cell's remaining unweighted allowance.</p>
 */
public final class WDTW_D implements Serializable {

    @Serial
    private static final long serialVersionUID = 1L;

    private transient volatile WeightProfile cachedProfile;

    public WDTW_D() {
    }

    public double distance(
            Object first,
            Object second,
            double bestSoFar,
            double g
    ) {
        return distance(first, second, bestSoFar, g, null);
    }

    public double distance(
            Object first,
            Object second,
            double bestSoFar,
            double g,
            int[] selectedDimensions
    ) {
        if (first instanceof double[][] firstValues
                && second instanceof double[][] secondValues) {
            return distance(
                    firstValues, secondValues, bestSoFar, g,
                    selectedDimensions
            );
        }
        if (first instanceof float[][] firstValues
                && second instanceof float[][] secondValues) {
            return distance(
                    firstValues, secondValues, bestSoFar, g,
                    selectedDimensions
            );
        }
        throw unsupportedPair(first, second);
    }

    private double distance(
            double[][] first,
            double[][] second,
            double bestSoFar,
            double g,
            int[] selectedDimensions
    ) {
        requireNonempty(first, second);
        int firstLength = first[0].length;
        int secondLength = second[0].length;
        double[] weights = weights(g, Math.max(firstLength, secondLength));

        if (firstLength >= secondLength) {
            return distanceKernel(
                    first, second, bestSoFar, weights,
                    selectedDimensions
            );
        }
        return distanceKernel(
                second, first, bestSoFar, weights,
                selectedDimensions
        );
    }

    private double distance(
            float[][] first,
            float[][] second,
            double bestSoFar,
            double g,
            int[] selectedDimensions
    ) {
        requireNonempty(first, second);
        int firstLength = first[0].length;
        int secondLength = second[0].length;
        double[] weights = weights(g, Math.max(firstLength, secondLength));

        if (firstLength >= secondLength) {
            return distanceKernel(
                    first, second, bestSoFar, weights,
                    selectedDimensions
            );
        }
        return distanceKernel(
                second, first, bestSoFar, weights,
                selectedDimensions
        );
    }

    private static double distanceKernel(
            double[][] rowSeries,
            double[][] columnSeries,
            double bestSoFar,
            double[] weights,
            int[] selectedDimensions
    ) {
        int rowCount = rowSeries[0].length;
        int columnCount = columnSeries[0].length;
        double cutoff = normalizedCutoff(bestSoFar);
        double[] previous = new double[columnCount];
        double[] current = new double[columnCount];
        Arrays.fill(previous, Double.POSITIVE_INFINITY);

        for (int row = 0; row < rowCount; row++) {
            double rowMinimum = Double.POSITIVE_INFINITY;
            for (int column = 0; column < columnCount; column++) {
                double predecessor = predecessor(
                        previous, current, row, column
                );
                double accumulated;

                if (predecessor == Double.POSITIVE_INFINITY) {
                    accumulated = Double.POSITIVE_INFINITY;
                } else {
                    double weight = weights[Math.abs(row - column)];
                    double weightedAllowance = cutoff - predecessor;
                    double unweightedAllowance = weight == 0.0
                            ? Double.POSITIVE_INFINITY
                            : weightedAllowance / weight;
                    double localCost = selectedDimensions == null
                            ? localCostAll(
                                    rowSeries, columnSeries,
                                    row, column, unweightedAllowance
                            )
                            : localCostSelected(
                                    rowSeries, columnSeries,
                                    row, column, unweightedAllowance,
                                    selectedDimensions
                            );
                    accumulated = localCost > unweightedAllowance
                            ? Double.POSITIVE_INFINITY
                            : predecessor + weight * localCost;
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
            float[][] rowSeries,
            float[][] columnSeries,
            double bestSoFar,
            double[] weights,
            int[] selectedDimensions
    ) {
        int rowCount = rowSeries[0].length;
        int columnCount = columnSeries[0].length;
        double cutoff = normalizedCutoff(bestSoFar);
        double[] previous = new double[columnCount];
        double[] current = new double[columnCount];
        Arrays.fill(previous, Double.POSITIVE_INFINITY);

        for (int row = 0; row < rowCount; row++) {
            double rowMinimum = Double.POSITIVE_INFINITY;
            for (int column = 0; column < columnCount; column++) {
                double predecessor = predecessor(
                        previous, current, row, column
                );
                double accumulated;

                if (predecessor == Double.POSITIVE_INFINITY) {
                    accumulated = Double.POSITIVE_INFINITY;
                } else {
                    double weight = weights[Math.abs(row - column)];
                    double weightedAllowance = cutoff - predecessor;
                    double unweightedAllowance = weight == 0.0
                            ? Double.POSITIVE_INFINITY
                            : weightedAllowance / weight;
                    double localCost = selectedDimensions == null
                            ? localCostAll(
                                    rowSeries, columnSeries,
                                    row, column, unweightedAllowance
                            )
                            : localCostSelected(
                                    rowSeries, columnSeries,
                                    row, column, unweightedAllowance,
                                    selectedDimensions
                            );
                    accumulated = localCost > unweightedAllowance
                            ? Double.POSITIVE_INFINITY
                            : predecessor + weight * localCost;
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

    private static double predecessor(
            double[] previous,
            double[] current,
            int row,
            int column
    ) {
        if (row == 0 && column == 0) {
            return 0.0;
        }
        double diagonal = column == 0
                ? Double.POSITIVE_INFINITY
                : previous[column - 1];
        double above = previous[column];
        double left = column == 0
                ? Double.POSITIVE_INFINITY
                : current[column - 1];
        return minimum(diagonal, above, left);
    }

    private static double localCostAll(
            double[][] first,
            double[][] second,
            int firstTime,
            int secondTime,
            double allowance
    ) {
        double total = 0.0;
        for (int dimension = 0; dimension < first.length; dimension++) {
            double difference = first[dimension][firstTime]
                    - second[dimension][secondTime];
            total += difference * difference;
            if (total > allowance) {
                return total;
            }
        }
        return total;
    }

    private static double localCostAll(
            float[][] first,
            float[][] second,
            int firstTime,
            int secondTime,
            double allowance
    ) {
        double total = 0.0;
        for (int dimension = 0; dimension < first.length; dimension++) {
            double difference = (double) first[dimension][firstTime]
                    - (double) second[dimension][secondTime];
            total += difference * difference;
            if (total > allowance) {
                return total;
            }
        }
        return total;
    }

    private static double localCostSelected(
            double[][] first,
            double[][] second,
            int firstTime,
            int secondTime,
            double allowance,
            int[] selectedDimensions
    ) {
        double total = 0.0;
        for (int position = 0;
             position < selectedDimensions.length;
             position++) {
            int dimension = selectedDimensions[position];
            double difference = first[dimension][firstTime]
                    - second[dimension][secondTime];
            total += difference * difference;
            if (total > allowance) {
                return total;
            }
        }
        return total;
    }

    private static double localCostSelected(
            float[][] first,
            float[][] second,
            int firstTime,
            int secondTime,
            double allowance,
            int[] selectedDimensions
    ) {
        double total = 0.0;
        for (int position = 0;
             position < selectedDimensions.length;
             position++) {
            int dimension = selectedDimensions[position];
            double difference = (double) first[dimension][firstTime]
                    - (double) second[dimension][secondTime];
            total += difference * difference;
            if (total > allowance) {
                return total;
            }
        }
        return total;
    }

    private double[] weights(double g, int maximumLength) {
        if (!Double.isFinite(g) || g < 0.0) {
            throw new IllegalArgumentException(
                    "Dependent WDTW g must be finite and nonnegative."
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
            generated[displacement] = 1.0
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
                    "Dependent WDTW bestSoFar cannot be NaN."
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
            double[][] first,
            double[][] second
    ) {
        if (first.length == 0 || second.length == 0
                || first[0].length == 0 || second[0].length == 0) {
            throw new IllegalArgumentException(
                    "Dependent WDTW requires nonempty dimensions and time axes."
            );
        }
    }

    private static void requireNonempty(
            float[][] first,
            float[][] second
    ) {
        if (first.length == 0 || second.length == 0
                || first[0].length == 0 || second[0].length == 0) {
            throw new IllegalArgumentException(
                    "Dependent WDTW requires nonempty dimensions and time axes."
            );
        }
    }

    private static IllegalArgumentException unsupportedPair(
            Object first,
            Object second
    ) {
        return new IllegalArgumentException(
                "Dependent WDTW requires matching double[][] or float[][] "
                        + "inputs. Received " + typeName(first) + " and "
                        + typeName(second) + "."
        );
    }

    private static String typeName(Object value) {
        return value == null ? "null" : value.getClass().getTypeName();
    }

    private record WeightProfile(
            double g,
            double[] weights
    ) {
    }
}
