package distance.elastic;

import util.Pair;

import java.io.Serial;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Univariate Dynamic Time Warping with optional alignment-path reconstruction.
 *
 * <p>The distance-only method delegates to the optimized squared-cost
 * {@link DTW} kernel. Path reconstruction uses two cost rows and one byte of
 * predecessor information per reachable matrix cell. It does not retain a
 * full matrix of double costs or a three-dimensional integer backtrack array.</p>
 *
 * <p>Both {@code double[]} and {@code float[]} inputs are supported. A
 * negative window means unconstrained alignment. A finite window is widened
 * when necessary to preserve an endpoint-to-endpoint path for unequal-length
 * inputs.</p>
 *
 * <p>The class is stateless and safe for concurrent use. Path calculations do
 * not use {@code bestSoFar}: pruning by a competitive distance bound may remove
 * cells needed to reconstruct the exact unconstrained optimum path.</p>
 */
public final class DTWWithPath implements Serializable {

    @Serial
    private static final long serialVersionUID = 1L;

    private static final byte UNREACHABLE = 0;
    private static final byte START = 1;
    private static final byte DIAGONAL = 2;
    private static final byte ABOVE = 3;
    private static final byte LEFT = 4;

    private final DTW dtw;

    public DTWWithPath() {
        dtw = new DTW();
    }

    /** Returns accumulated squared DTW cost. */
    public double distance(
            double[] first,
            double[] second,
            int windowSize
    ) {
        return dtw.distance(
                first,
                second,
                Double.POSITIVE_INFINITY,
                windowSize
        );
    }

    /** Returns accumulated squared DTW cost. */
    public double distance(
            float[] first,
            float[] second,
            int windowSize
    ) {
        return dtw.distance(
                first,
                second,
                Double.POSITIVE_INFINITY,
                windowSize
        );
    }

    /** Returns accumulated squared DTW cost with finite-bound pruning. */
    public double distance(
            Object first,
            Object second,
            double bestSoFar,
            int windowSize
    ) {
        return dtw.distance(first, second, bestSoFar, windowSize);
    }

    public List<Pair<Integer, Integer>> getAlignmentPath(
            double[] first,
            double[] second,
            int windowSize
    ) {
        requireNonempty(first.length, second.length);
        return alignmentPath(first, second, windowSize);
    }

    public List<Pair<Integer, Integer>> getAlignmentPath(
            float[] first,
            float[] second,
            int windowSize
    ) {
        requireNonempty(first.length, second.length);
        return alignmentPath(first, second, windowSize);
    }

    public List<Pair<Integer, Integer>> getAlignmentPath(
            Object first,
            Object second,
            int windowSize
    ) {
        if (first instanceof double[] firstValues
                && second instanceof double[] secondValues) {
            return getAlignmentPath(
                    firstValues,
                    secondValues,
                    windowSize
            );
        }
        if (first instanceof float[] firstValues
                && second instanceof float[] secondValues) {
            return getAlignmentPath(
                    firstValues,
                    secondValues,
                    windowSize
            );
        }
        throw unsupportedPair(first, second);
    }

    private static List<Pair<Integer, Integer>> alignmentPath(
            double[] first,
            double[] second,
            int windowSize
    ) {
        int rowCount = first.length;
        int columnCount = second.length;
        int window = resolveWindow(windowSize, rowCount, columnCount);
        double[] previous = new double[columnCount];
        double[] current = new double[columnCount];
        byte[][] predecessors = new byte[rowCount][columnCount];

        for (int index = 0; index < columnCount; index++) {
            previous[index] = Double.POSITIVE_INFINITY;
        }

        for (int row = 0; row < rowCount; row++) {
            int start = Math.max(0, row - window);
            int end = Math.min(columnCount - 1, row + window);
            if (start > 0) {
                current[start - 1] = Double.POSITIVE_INFINITY;
            }

            double rowValue = first[row];
            for (int column = start; column <= end; column++) {
                double difference = rowValue - second[column];
                double localCost = difference * difference;

                if (row == 0 && column == 0) {
                    current[column] = localCost;
                    predecessors[row][column] = START;
                    continue;
                }

                double diagonal = row > 0 && column > 0
                        ? previous[column - 1]
                        : Double.POSITIVE_INFINITY;
                double above = row > 0
                        ? previous[column]
                        : Double.POSITIVE_INFINITY;
                double left = column > start
                        ? current[column - 1]
                        : Double.POSITIVE_INFINITY;

                if (diagonal <= above && diagonal <= left) {
                    current[column] = localCost + diagonal;
                    predecessors[row][column] = DIAGONAL;
                } else if (above <= left) {
                    current[column] = localCost + above;
                    predecessors[row][column] = ABOVE;
                } else {
                    current[column] = localCost + left;
                    predecessors[row][column] = LEFT;
                }
            }

            double[] temporary = previous;
            previous = current;
            current = temporary;
        }

        return reconstruct(predecessors, rowCount, columnCount);
    }

    private static List<Pair<Integer, Integer>> alignmentPath(
            float[] first,
            float[] second,
            int windowSize
    ) {
        int rowCount = first.length;
        int columnCount = second.length;
        int window = resolveWindow(windowSize, rowCount, columnCount);
        double[] previous = new double[columnCount];
        double[] current = new double[columnCount];
        byte[][] predecessors = new byte[rowCount][columnCount];

        for (int index = 0; index < columnCount; index++) {
            previous[index] = Double.POSITIVE_INFINITY;
        }

        for (int row = 0; row < rowCount; row++) {
            int start = Math.max(0, row - window);
            int end = Math.min(columnCount - 1, row + window);
            if (start > 0) {
                current[start - 1] = Double.POSITIVE_INFINITY;
            }

            double rowValue = first[row];
            for (int column = start; column <= end; column++) {
                double difference = rowValue - (double) second[column];
                double localCost = difference * difference;

                if (row == 0 && column == 0) {
                    current[column] = localCost;
                    predecessors[row][column] = START;
                    continue;
                }

                double diagonal = row > 0 && column > 0
                        ? previous[column - 1]
                        : Double.POSITIVE_INFINITY;
                double above = row > 0
                        ? previous[column]
                        : Double.POSITIVE_INFINITY;
                double left = column > start
                        ? current[column - 1]
                        : Double.POSITIVE_INFINITY;

                if (diagonal <= above && diagonal <= left) {
                    current[column] = localCost + diagonal;
                    predecessors[row][column] = DIAGONAL;
                } else if (above <= left) {
                    current[column] = localCost + above;
                    predecessors[row][column] = ABOVE;
                } else {
                    current[column] = localCost + left;
                    predecessors[row][column] = LEFT;
                }
            }

            double[] temporary = previous;
            previous = current;
            current = temporary;
        }

        return reconstruct(predecessors, rowCount, columnCount);
    }

    private static List<Pair<Integer, Integer>> reconstruct(
            byte[][] predecessors,
            int rowCount,
            int columnCount
    ) {
        int row = rowCount - 1;
        int column = columnCount - 1;
        if (predecessors[row][column] == UNREACHABLE) {
            return Collections.emptyList();
        }

        List<Pair<Integer, Integer>> reversed = new ArrayList<>(
                rowCount + columnCount - 1
        );

        while (true) {
            reversed.add(new Pair<>(row, column));
            byte direction = predecessors[row][column];

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

    private static void requireNonempty(
            int firstLength,
            int secondLength
    ) {
        if (firstLength == 0 || secondLength == 0) {
            throw new IllegalArgumentException(
                    "DTWWithPath requires two nonempty series."
            );
        }
    }

    private static IllegalArgumentException unsupportedPair(
            Object first,
            Object second
    ) {
        return new IllegalArgumentException(
                "DTWWithPath requires matching double[] or float[] inputs. "
                        + "Received " + typeName(first) + " and "
                        + typeName(second) + "."
        );
    }

    private static String typeName(Object value) {
        return value == null ? "null" : value.getClass().getTypeName();
    }
}
