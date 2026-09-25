package distance.multiTS;

import core.contracts.ObjectDataset;
import distance.elastic.ShapeHoG1dDTW;

import java.io.Serial;
import java.io.Serializable;
import java.util.Random;

/** Independent multivariate ShapeHoG-DTW with summed channel costs. */
public final class ShapeHoG1dDTW_I implements Serializable {

    @Serial
    private static final long serialVersionUID = 1L;

    private final ShapeHoG1dDTW hogDtw;

    public ShapeHoG1dDTW_I() {
        hogDtw = new ShapeHoG1dDTW();
    }

    public double distance(
            Object first,
            Object second,
            double bestSoFar,
            int windowSize
    ) {
        return distance(first, second, bestSoFar, windowSize, null);
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
                    firstValues, secondValues, bestSoFar,
                    windowSize, selectedDimensions
            );
        }
        if (first instanceof float[][] firstValues
                && second instanceof float[][] secondValues) {
            return distance(
                    firstValues, secondValues, bestSoFar,
                    windowSize, selectedDimensions
            );
        }
        throw unsupportedPair(first, second);
    }

    private double distance(
            double[][] first,
            double[][] second,
            double bestSoFar,
            int windowSize,
            int[] selectedDimensions
    ) {
        double total = 0.0;
        int count = selectedDimensions == null
                ? first.length : selectedDimensions.length;
        for (int position = 0; position < count; position++) {
            int dimension = selectedDimensions == null
                    ? position : selectedDimensions[position];
            double cost = hogDtw.distance(
                    first[dimension], second[dimension],
                    remainingBudget(bestSoFar, total), windowSize
            );
            if (cost == Double.POSITIVE_INFINITY) {
                return Double.POSITIVE_INFINITY;
            }
            total += cost;
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
            int[] selectedDimensions
    ) {
        double total = 0.0;
        int count = selectedDimensions == null
                ? first.length : selectedDimensions.length;
        for (int position = 0; position < count; position++) {
            int dimension = selectedDimensions == null
                    ? position : selectedDimensions[position];
            double cost = hogDtw.distance(
                    first[dimension], second[dimension],
                    remainingBudget(bestSoFar, total), windowSize
            );
            if (cost == Double.POSITIVE_INFINITY) {
                return Double.POSITIVE_INFINITY;
            }
            total += cost;
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
        return hogDtw.get_random_window(dataset, random);
    }

    private static IllegalArgumentException unsupportedPair(
            Object first,
            Object second
    ) {
        return new IllegalArgumentException(
                "Independent ShapeHoG-DTW requires matching double[][] or "
                        + "float[][] inputs. Received " + typeName(first)
                        + " and " + typeName(second) + "."
        );
    }

    private static String typeName(Object value) {
        return value == null ? "null" : value.getClass().getTypeName();
    }
}
