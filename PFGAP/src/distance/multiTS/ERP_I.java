package distance.multiTS;

import core.contracts.ObjectDataset;
import distance.elastic.ERP;

import java.io.Serial;
import java.io.Serializable;
import java.util.Random;

/** Independent multivariate ERP with summed selected-channel costs. */
public final class ERP_I implements Serializable {

    @Serial
    private static final long serialVersionUID = 1L;

    private final ERP erp;

    public ERP_I() {
        erp = new ERP();
    }

    public double distance(
            Object first,
            Object second,
            double bestSoFar,
            int windowSize,
            double gapValue
    ) {
        return distance(
                first,
                second,
                bestSoFar,
                windowSize,
                gapValue,
                null
        );
    }

    public double distance(
            Object first,
            Object second,
            double bestSoFar,
            int windowSize,
            double gapValue,
            int[] selectedDimensions
    ) {
        if (first instanceof double[][] firstValues
                && second instanceof double[][] secondValues) {
            return distance(
                    firstValues,
                    secondValues,
                    bestSoFar,
                    windowSize,
                    gapValue,
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
                    gapValue,
                    selectedDimensions
            );
        }
        throw unsupportedPair(first, second);
    }

    private double distance(
            double[][] first,
            double[][] second,
            double bestSoFar,
            int windowSize,
            double gapValue,
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
            double channelCost = erp.distance(
                    first[dimension],
                    second[dimension],
                    remainingBudget(bestSoFar, total),
                    windowSize,
                    gapValue
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
            int windowSize,
            double gapValue,
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
            double channelCost = erp.distance(
                    first[dimension],
                    second[dimension],
                    remainingBudget(bestSoFar, total),
                    windowSize,
                    gapValue
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
        return erp.get_random_window(dataset, random);
    }

    public double get_random_g(ObjectDataset dataset, Random random) {
        return erp.get_random_g(dataset, random);
    }

    private static IllegalArgumentException unsupportedPair(
            Object first,
            Object second
    ) {
        return new IllegalArgumentException(
                "Independent ERP requires matching double[][] or float[][] "
                        + "inputs. Received " + typeName(first) + " and "
                        + typeName(second) + "."
        );
    }

    private static String typeName(Object value) {
        return value == null ? "null" : value.getClass().getTypeName();
    }
}
