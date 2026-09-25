package distance.multiTS;

import core.contracts.ObjectDataset;
import distance.elastic.DDTW;

import java.io.Serial;
import java.io.Serializable;
import java.util.Random;

/**
 * Independent multivariate Derivative Dynamic Time Warping.
 *
 * <p>Each selected channel is transformed with the unified derivative
 * convention and aligned independently. The resulting squared DDTW costs are
 * summed. A finite {@code bestSoFar} is reduced by each completed channel cost,
 * allowing later channels to prune against the remaining budget.</p>
 *
 * <p>Both {@code double[][]} and {@code float[][]} inputs are supported without
 * slicing or whole-matrix conversion. A null selected-dimension array uses all
 * channels. The class has no mutable scratch state and is safe for concurrent
 * use.</p>
 */
public final class DDTW_I implements Serializable {

    @Serial
    private static final long serialVersionUID = 1L;

    private final DDTW ddtw;

    public DDTW_I() {
        ddtw = new DDTW();
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
            double channelCost = ddtw.distance(
                    first[dimension],
                    second[dimension],
                    remainingBudget(bestSoFar, total),
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
            double channelCost = ddtw.distance(
                    first[dimension],
                    second[dimension],
                    remainingBudget(bestSoFar, total),
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
            double channelCost = ddtw.distance(
                    first[dimension],
                    second[dimension],
                    remainingBudget(bestSoFar, total),
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
            double channelCost = ddtw.distance(
                    first[dimension],
                    second[dimension],
                    remainingBudget(bestSoFar, total),
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

    public int get_random_window(ObjectDataset dataset, Random random) {
        return ddtw.get_random_window(dataset, random);
    }

    private static IllegalArgumentException unsupportedPair(
            Object first,
            Object second
    ) {
        return new IllegalArgumentException(
                "Independent DDTW requires matching double[][] or float[][] "
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
