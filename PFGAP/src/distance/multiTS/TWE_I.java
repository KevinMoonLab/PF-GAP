package distance.multiTS;

import core.contracts.ObjectDataset;
import distance.elastic.TWE;

import java.io.Serial;
import java.io.Serializable;
import java.util.Random;

/** Independent multivariate TWE with summed selected-channel costs. */
public final class TWE_I implements Serializable {

    @Serial
    private static final long serialVersionUID = 1L;

    private final TWE twe;

    public TWE_I() {
        twe = new TWE();
    }

    public double distance(
            Object first,
            Object second,
            double bestSoFar,
            double nu,
            double lambda
    ) {
        return distance(
                first,
                second,
                bestSoFar,
                nu,
                lambda,
                null
        );
    }

    public double distance(
            Object first,
            Object second,
            double bestSoFar,
            double nu,
            double lambda,
            int[] selectedDimensions
    ) {
        if (first instanceof double[][] firstValues
                && second instanceof double[][] secondValues) {
            return distance(
                    firstValues,
                    secondValues,
                    bestSoFar,
                    nu,
                    lambda,
                    selectedDimensions
            );
        }
        if (first instanceof float[][] firstValues
                && second instanceof float[][] secondValues) {
            return distance(
                    firstValues,
                    secondValues,
                    bestSoFar,
                    nu,
                    lambda,
                    selectedDimensions
            );
        }
        throw unsupportedPair(first, second);
    }

    private double distance(
            double[][] first,
            double[][] second,
            double bestSoFar,
            double nu,
            double lambda,
            int[] selectedDimensions
    ) {
        double total = 0.0;
        int count = selectedDimensions == null
                ? first.length
                : selectedDimensions.length;

        for (int position = 0; position < count; position++) {
            int dimension = selectedDimensions == null
                    ? position
                    : selectedDimensions[position];
            double channelCost = twe.distance(
                    first[dimension],
                    second[dimension],
                    remainingBudget(bestSoFar, total),
                    nu,
                    lambda
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

    private double distance(
            float[][] first,
            float[][] second,
            double bestSoFar,
            double nu,
            double lambda,
            int[] selectedDimensions
    ) {
        double total = 0.0;
        int count = selectedDimensions == null
                ? first.length
                : selectedDimensions.length;

        for (int position = 0; position < count; position++) {
            int dimension = selectedDimensions == null
                    ? position
                    : selectedDimensions[position];
            double channelCost = twe.distance(
                    first[dimension],
                    second[dimension],
                    remainingBudget(bestSoFar, total),
                    nu,
                    lambda
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

    public double get_random_nu(ObjectDataset dataset, Random random) {
        return twe.get_random_nu(dataset, random);
    }

    public double get_random_lambda(ObjectDataset dataset, Random random) {
        return twe.get_random_lambda(dataset, random);
    }

    private static IllegalArgumentException unsupportedPair(
            Object first,
            Object second
    ) {
        return new IllegalArgumentException(
                "Independent TWE requires matching double[][] or float[][] "
                        + "inputs. Received " + typeName(first) + " and "
                        + typeName(second) + "."
        );
    }

    private static String typeName(Object value) {
        return value == null ? "null" : value.getClass().getTypeName();
    }
}
