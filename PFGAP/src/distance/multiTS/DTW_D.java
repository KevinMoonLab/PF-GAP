package distance.multiTS;

import util.Pair;

import java.io.Serial;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * Dependent multivariate Dynamic Time Warping using squared Euclidean local
 * costs across channels.
 *
 * <p>One shared warping path is learned across all selected channels. Every
 * overload returns accumulated squared DTW cost, and a finite
 * {@code bestSoFar} is interpreted in the same units. Distance-only evaluation
 * uses two rows, cell pruning, and row-level early abandonment.</p>
 *
 * <p>Both {@code double[][]} and {@code float[][]} dimension-major inputs are
 * supported without conversion or slicing. A null selected-channel array uses
 * every channel. Rectangular channel layout and selected-index validity are
 * trusted dataset and splitter preconditions.</p>
 *
 * <p>The recurrence is scalar. Although each multivariate local cost is a
 * reduction across channels, Java {@code double[][]}/{@code float[][]} stores
 * those channel values in separate arrays, so they cannot be loaded as one
 * contiguous SIMD vector without gathering from distinct objects or changing
 * representation. Enabling the Vector API therefore does not alter this
 * kernel.</p>
 */
public final class DTW_D implements Serializable {

    @Serial
    private static final long serialVersionUID = 1L;

    private static final byte START = 0;
    private static final byte DIAGONAL = 1;
    private static final byte ABOVE = 2;
    private static final byte LEFT = 3;

    public DTW_D() {
    }

    public double distance(
            Object first,
            Object second,
            double bestSoFar,
            int windowSize
    ) {
        if (first instanceof double[][] firstValues
                && second instanceof double[][] secondValues) {
            return distance(
                    firstValues,
                    secondValues,
                    bestSoFar,
                    windowSize,
                    null
            );
        }

        if (first instanceof float[][] firstValues
                && second instanceof float[][] secondValues) {
            return distance(
                    firstValues,
                    secondValues,
                    bestSoFar,
                    windowSize,
                    null
            );
        }

        throw unsupportedPair(first, second);
    }

    public double distance(
            Object first,
            Object second,
            double bestSoFar,
            int windowSize,
            int[] selectedDimensions
    ) {
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

    private static double distance(
            double[][] first,
            double[][] second,
            double bestSoFar,
            int windowSize,
            int[] selectedDimensions
    ) {
        requireNonempty(first, second);
        int firstLength = first[0].length;
        int secondLength = second[0].length;

        // Put the shorter time axis in columns to minimize row storage.
        if (firstLength >= secondLength) {
            return distanceKernel(
                    first,
                    second,
                    bestSoFar,
                    windowSize,
                    selectedDimensions
            );
        }
        return distanceKernel(
                second,
                first,
                bestSoFar,
                windowSize,
                selectedDimensions
        );
    }

    private static double distance(
            float[][] first,
            float[][] second,
            double bestSoFar,
            int windowSize,
            int[] selectedDimensions
    ) {
        requireNonempty(first, second);
        int firstLength = first[0].length;
        int secondLength = second[0].length;

        if (firstLength >= secondLength) {
            return distanceKernel(
                    first,
                    second,
                    bestSoFar,
                    windowSize,
                    selectedDimensions
            );
        }
        return distanceKernel(
                second,
                first,
                bestSoFar,
                windowSize,
                selectedDimensions
        );
    }

    private static double distanceKernel(
            double[][] rowSeries,
            double[][] columnSeries,
            double bestSoFar,
            int windowSize,
            int[] selectedDimensions
    ) {
        int rowCount = rowSeries[0].length;
        int columnCount = columnSeries[0].length;
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
            for (int column = start; column <= end; column++) {
                double predecessor;
                if (row == 0 && column == 0) {
                    predecessor = 0.0;
                } else {
                    double diagonal = column == 0
                            ? Double.POSITIVE_INFINITY
                            : previous[column - 1];
                    double above = previous[column];
                    double left = column == start
                            ? Double.POSITIVE_INFINITY
                            : current[column - 1];
                    predecessor = minimum(diagonal, above, left);
                }

                double accumulated;
                if (predecessor == Double.POSITIVE_INFINITY) {
                    accumulated = Double.POSITIVE_INFINITY;
                } else {
                    double allowance = cutoff - predecessor;
                    double localCost = selectedDimensions == null
                            ? localCostAll(
                                    rowSeries,
                                    columnSeries,
                                    row,
                                    column,
                                    allowance
                            )
                            : localCostSelected(
                                    rowSeries,
                                    columnSeries,
                                    row,
                                    column,
                                    allowance,
                                    selectedDimensions
                            );
                    accumulated = localCost > allowance
                            ? Double.POSITIVE_INFINITY
                            : predecessor + localCost;
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
            int windowSize,
            int[] selectedDimensions
    ) {
        int rowCount = rowSeries[0].length;
        int columnCount = columnSeries[0].length;
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
            for (int column = start; column <= end; column++) {
                double predecessor;
                if (row == 0 && column == 0) {
                    predecessor = 0.0;
                } else {
                    double diagonal = column == 0
                            ? Double.POSITIVE_INFINITY
                            : previous[column - 1];
                    double above = previous[column];
                    double left = column == start
                            ? Double.POSITIVE_INFINITY
                            : current[column - 1];
                    predecessor = minimum(diagonal, above, left);
                }

                double accumulated;
                if (predecessor == Double.POSITIVE_INFINITY) {
                    accumulated = Double.POSITIVE_INFINITY;
                } else {
                    double allowance = cutoff - predecessor;
                    double localCost = selectedDimensions == null
                            ? localCostAll(
                                    rowSeries,
                                    columnSeries,
                                    row,
                                    column,
                                    allowance
                            )
                            : localCostSelected(
                                    rowSeries,
                                    columnSeries,
                                    row,
                                    column,
                                    allowance,
                                    selectedDimensions
                            );
                    accumulated = localCost > allowance
                            ? Double.POSITIVE_INFINITY
                            : predecessor + localCost;
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

    /**
     * Reconstructs an all-channel dependent-DTW path for double input.
     * Distance-only calls should use {@link #distance(Object, Object, double, int)}.
     */
    public List<Pair<Integer, Integer>> getAlignmentPath(
            double[][] first,
            double[][] second,
            int windowSize
    ) {
        requireNonempty(first, second);
        int firstLength = first[0].length;
        int secondLength = second[0].length;
        int window = resolveWindow(windowSize, firstLength, secondLength);

        double[] previous = new double[secondLength];
        double[] current = new double[secondLength];
        Arrays.fill(previous, Double.POSITIVE_INFINITY);
        byte[][] predecessor = new byte[firstLength][secondLength];

        for (int row = 0; row < firstLength; row++) {
            int start = Math.max(0, row - window);
            int end = Math.min(secondLength - 1, row + window);
            if (start > 0) {
                current[start - 1] = Double.POSITIVE_INFINITY;
            }

            for (int column = start; column <= end; column++) {
                double localCost = localCostAll(
                        first,
                        second,
                        row,
                        column,
                        Double.POSITIVE_INFINITY
                );

                if (row == 0 && column == 0) {
                    current[column] = localCost;
                    predecessor[row][column] = START;
                    continue;
                }

                double diagonal = column == 0
                        ? Double.POSITIVE_INFINITY
                        : previous[column - 1];
                double above = previous[column];
                double left = column == start
                        ? Double.POSITIVE_INFINITY
                        : current[column - 1];

                if (diagonal <= above && diagonal <= left) {
                    current[column] = localCost + diagonal;
                    predecessor[row][column] = DIAGONAL;
                } else if (above <= left) {
                    current[column] = localCost + above;
                    predecessor[row][column] = ABOVE;
                } else {
                    current[column] = localCost + left;
                    predecessor[row][column] = LEFT;
                }
            }

            double[] temporary = previous;
            previous = current;
            current = temporary;
        }

        if (previous[secondLength - 1] == Double.POSITIVE_INFINITY) {
            return Collections.emptyList();
        }

        List<Pair<Integer, Integer>> reversed = new ArrayList<>(
                firstLength + secondLength - 1
        );
        int row = firstLength - 1;
        int column = secondLength - 1;

        while (true) {
            reversed.add(new Pair<>(row, column));
            byte direction = predecessor[row][column];
            if (direction == START) {
                break;
            }
            if (direction == DIAGONAL) {
                row--;
                column--;
            } else if (direction == ABOVE) {
                row--;
            } else if (direction == LEFT) {
                column--;
            } else {
                return Collections.emptyList();
            }
        }

        Collections.reverse(reversed);
        return reversed;
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
                    "Dependent DTW bestSoFar cannot be NaN."
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
        if (first.length == 0
                || second.length == 0
                || first[0].length == 0
                || second[0].length == 0) {
            throw new IllegalArgumentException(
                    "Dependent DTW requires nonempty dimensions and time axes."
            );
        }
    }

    private static void requireNonempty(
            float[][] first,
            float[][] second
    ) {
        if (first.length == 0
                || second.length == 0
                || first[0].length == 0
                || second[0].length == 0) {
            throw new IllegalArgumentException(
                    "Dependent DTW requires nonempty dimensions and time axes."
            );
        }
    }

    private static IllegalArgumentException unsupportedPair(
            Object first,
            Object second
    ) {
        return new IllegalArgumentException(
                "Dependent DTW requires matching double[][] or float[][] "
                        + "inputs. Received "
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
}
