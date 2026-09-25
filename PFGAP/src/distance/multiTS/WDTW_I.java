package distance.multiTS;

import core.contracts.ObjectDataset;
import distance.elastic.WDTW;

import java.io.Serial;
import java.io.Serializable;
import java.util.Random;

/**
 * Independent multivariate Weighted Dynamic Time Warping.
 *
 * <p>Each selected channel is aligned independently and the resulting weighted
 * squared costs are summed. A finite {@code bestSoFar} is reduced by each
 * completed channel cost, allowing later channels to prune against the
 * remaining budget. Both {@code double[][]} and {@code float[][]} inputs are
 * supported without conversion or slicing.</p>
 */
public final class WDTW_I implements Serializable {

    @Serial
    private static final long serialVersionUID = 1L;

    private final WDTW wdtw;

    public WDTW_I() {
        wdtw = new WDTW();
    }

    public double distance(
            Object first,
            Object second,
            double bestSoFar,
            double g
    ) {
        if (first instanceof double[][] firstValues
                && second instanceof double[][] secondValues) {
            return distanceAll(firstValues, secondValues, bestSoFar, g);
        }
        if (first instanceof float[][] firstValues
                && second instanceof float[][] secondValues) {
            return distanceAll(firstValues, secondValues, bestSoFar, g);
        }
        throw unsupportedPair(first, second);
    }

    public double distance(
            Object first,
            Object second,
            double bestSoFar,
            double g,
            int[] selectedDimensions
    ) {
        if (selectedDimensions == null) {
            return distance(first, second, bestSoFar, g);
        }
        if (first instanceof double[][] firstValues
                && second instanceof double[][] secondValues) {
            return distanceSelected(
                    firstValues, secondValues, bestSoFar, g,
                    selectedDimensions
            );
        }
        if (first instanceof float[][] firstValues
                && second instanceof float[][] secondValues) {
            return distanceSelected(
                    firstValues, secondValues, bestSoFar, g,
                    selectedDimensions
            );
        }
        throw unsupportedPair(first, second);
    }

    private double distanceAll(
            double[][] first,
            double[][] second,
            double bestSoFar,
            double g
    ) {
        double total = 0.0;
        for (int dimension = 0; dimension < first.length; dimension++) {
            double channelCost = wdtw.distance(
                    first[dimension], second[dimension],
                    remainingBudget(bestSoFar, total), g
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
            double g
    ) {
        double total = 0.0;
        for (int dimension = 0; dimension < first.length; dimension++) {
            double channelCost = wdtw.distance(
                    first[dimension], second[dimension],
                    remainingBudget(bestSoFar, total), g
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
            double g,
            int[] selectedDimensions
    ) {
        double total = 0.0;
        for (int position = 0;
             position < selectedDimensions.length;
             position++) {
            int dimension = selectedDimensions[position];
            double channelCost = wdtw.distance(
                    first[dimension], second[dimension],
                    remainingBudget(bestSoFar, total), g
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
            double g,
            int[] selectedDimensions
    ) {
        double total = 0.0;
        for (int position = 0;
             position < selectedDimensions.length;
             position++) {
            int dimension = selectedDimensions[position];
            double channelCost = wdtw.distance(
                    first[dimension], second[dimension],
                    remainingBudget(bestSoFar, total), g
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

    public double get_random_g(ObjectDataset dataset, Random random) {
        return wdtw.get_random_g(dataset, random);
    }

    private static IllegalArgumentException unsupportedPair(
            Object first,
            Object second
    ) {
        return new IllegalArgumentException(
                "Independent WDTW requires matching double[][] or float[][] "
                        + "inputs. Received " + typeName(first) + " and "
                        + typeName(second) + "."
        );
    }

    private static String typeName(Object value) {
        return value == null ? "null" : value.getClass().getTypeName();
    }
}
