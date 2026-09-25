package distance.multiTS;

import distance.elastic.Euclidean;

import java.io.Serial;
import java.io.Serializable;

/**
 * Independent multivariate squared Euclidean distance.
 *
 * <p>Dimension selection is resolved only at the outer matrix level. For each
 * selected dimension, the complete temporal row is passed to the unselected
 * all-elements Euclidean kernel. Thus selected dimension indices are never
 * reinterpreted as temporal indices.</p>
 *
 * <p>The returned value is the sum of squared Euclidean row costs. When
 * {@code AppContext.useVectorApi} is enabled, each complete temporal row uses
 * the vectorized Euclidean kernel. Both {@code double[][]} and
 * {@code float[][]} inputs are supported without slicing or conversion.</p>
 */
public final class Euclidean_I implements Serializable {

    @Serial
    private static final long serialVersionUID = 1L;

    private final Euclidean euclidean;

    public Euclidean_I() {
        euclidean = new Euclidean();
    }

    public double distance(
            Object first,
            Object second,
            double bestSoFar
    ) {
        return distance(first, second, bestSoFar, null);
    }

    public double distance(
            Object first,
            Object second,
            double bestSoFar,
            int[] selectedDimensions
    ) {
        if (first instanceof double[][] firstValues
                && second instanceof double[][] secondValues) {
            return distance(
                    firstValues,
                    secondValues,
                    bestSoFar,
                    selectedDimensions
            );
        }
        if (first instanceof float[][] firstValues
                && second instanceof float[][] secondValues) {
            return distance(
                    firstValues,
                    secondValues,
                    bestSoFar,
                    selectedDimensions
            );
        }
        throw unsupportedPair(first, second);
    }

    private double distance(
            double[][] first,
            double[][] second,
            double bestSoFar,
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
            double rowCost = euclidean.distance(
                    first[dimension],
                    second[dimension],
                    remainingBudget(bestSoFar, total)
            );
            total += rowCost;
            if (total > bestSoFar) {
                return total;
            }
        }
        return total;
    }

    private double distance(
            float[][] first,
            float[][] second,
            double bestSoFar,
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
            double rowCost = euclidean.distance(
                    first[dimension],
                    second[dimension],
                    remainingBudget(bestSoFar, total)
            );
            total += rowCost;
            if (total > bestSoFar) {
                return total;
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
                "Euclidean_I requires matching double[][] or float[][] "
                        + "inputs. Received " + typeName(first) + " and "
                        + typeName(second) + "."
        );
    }

    private static String typeName(Object value) {
        return value == null ? "null" : value.getClass().getTypeName();
    }
}
