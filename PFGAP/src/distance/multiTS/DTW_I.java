package distance.multiTS;

import core.contracts.ObjectDataset;
import distance.elastic.DTW;

import java.io.Serial;
import java.io.Serializable;
import java.util.Random;

/**
 * Independent multivariate Dynamic Time Warping.
 *
 * <p>Each selected channel is aligned independently with univariate DTW and
 * the resulting squared DTW costs are summed. Every overload therefore returns
 * an accumulated squared cost. A finite {@code bestSoFar} is interpreted in
 * the same units and is reduced by each completed channel cost.</p>
 *
 * <p>Both {@code double[][]} and {@code float[][]} are supported without
 * conversion or slicing. A null selected-channel array uses every channel.
 * Channel-count equality and selected-index validity are trusted dataset and
 * splitter preconditions.</p>
 *
 * <p>The class is stateless apart from its stateless univariate evaluator and
 * is safe for concurrent use. The univariate DTW recurrence is scalar because
 * adjacent dynamic-programming cells are dependent; enabling the Vector API
 * does not change this kernel.</p>
 */
public final class DTW_I implements Serializable {

    @Serial
    private static final long serialVersionUID = 1L;

    private final DTW dtw;

    public DTW_I() {
        dtw = new DTW();
    }

    public double distance(
            Object first,
            Object second,
            double bestSoFar,
            int windowSize
    ) {
        if (first instanceof double[][] firstValues
                && second instanceof double[][] secondValues) {
            return distanceAll(
                    firstValues,
                    secondValues,
                    bestSoFar,
                    windowSize
            );
        }

        if (first instanceof float[][] firstValues
                && second instanceof float[][] secondValues) {
            return distanceAll(
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
            double bestSoFar,
            int windowSize,
            int[] selectedDimensions
    ) {
        if (selectedDimensions == null) {
            return distance(first, second, bestSoFar, windowSize);
        }

        if (first instanceof double[][] firstValues
                && second instanceof double[][] secondValues) {
            return distanceSelected(
                    firstValues,
                    secondValues,
                    bestSoFar,
                    windowSize,
                    selectedDimensions
            );
        }

        if (first instanceof float[][] firstValues
                && second instanceof float[][] secondValues) {
            return distanceSelected(
                    firstValues,
                    secondValues,
                    bestSoFar,
                    windowSize,
                    selectedDimensions
            );
        }

        throw unsupportedPair(first, second);
    }

    private double distanceAll(
            double[][] first,
            double[][] second,
            double bestSoFar,
            int windowSize
    ) {
        double total = 0.0;

        for (int dimension = 0; dimension < first.length; dimension++) {
            double remaining = remainingBudget(bestSoFar, total);
            double channelCost = dtw.distance(
                    first[dimension],
                    second[dimension],
                    remaining,
                    windowSize
            );

            if (channelCost == Double.POSITIVE_INFINITY) {
                return Double.POSITIVE_INFINITY;
            }

            total += channelCost;
            if (total > bestSoFar) {
                return Double.POSITIVE_INFINITY;
            }
        }

        return total;
    }

    private double distanceAll(
            float[][] first,
            float[][] second,
            double bestSoFar,
            int windowSize
    ) {
        double total = 0.0;

        for (int dimension = 0; dimension < first.length; dimension++) {
            double remaining = remainingBudget(bestSoFar, total);
            double channelCost = dtw.distance(
                    first[dimension],
                    second[dimension],
                    remaining,
                    windowSize
            );

            if (channelCost == Double.POSITIVE_INFINITY) {
                return Double.POSITIVE_INFINITY;
            }

            total += channelCost;
            if (total > bestSoFar) {
                return Double.POSITIVE_INFINITY;
            }
        }

        return total;
    }

    private double distanceSelected(
            double[][] first,
            double[][] second,
            double bestSoFar,
            int windowSize,
            int[] selectedDimensions
    ) {
        double total = 0.0;

        for (int position = 0;
             position < selectedDimensions.length;
             position++) {
            int dimension = selectedDimensions[position];
            double remaining = remainingBudget(bestSoFar, total);
            double channelCost = dtw.distance(
                    first[dimension],
                    second[dimension],
                    remaining,
                    windowSize
            );

            if (channelCost == Double.POSITIVE_INFINITY) {
                return Double.POSITIVE_INFINITY;
            }

            total += channelCost;
            if (total > bestSoFar) {
                return Double.POSITIVE_INFINITY;
            }
        }

        return total;
    }

    private double distanceSelected(
            float[][] first,
            float[][] second,
            double bestSoFar,
            int windowSize,
            int[] selectedDimensions
    ) {
        double total = 0.0;

        for (int position = 0;
             position < selectedDimensions.length;
             position++) {
            int dimension = selectedDimensions[position];
            double remaining = remainingBudget(bestSoFar, total);
            double channelCost = dtw.distance(
                    first[dimension],
                    second[dimension],
                    remaining,
                    windowSize
            );

            if (channelCost == Double.POSITIVE_INFINITY) {
                return Double.POSITIVE_INFINITY;
            }

            total += channelCost;
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

    private static IllegalArgumentException unsupportedPair(
            Object first,
            Object second
    ) {
        return new IllegalArgumentException(
                "Independent DTW requires matching double[][] or float[][] "
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

    public int get_random_window(ObjectDataset dataset, Random random) {
        return dtw.get_random_window(dataset, random);
    }
}
