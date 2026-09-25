package distance.multiTS;

import distance.elastic.CID;

import java.io.Serial;
import java.io.Serializable;

/**
 * Independent multivariate Complexity-Invariant Distance.
 *
 * <p>CID is computed independently for every selected channel and the squared
 * channel costs are summed. A finite {@code bestSoFar} is reduced by each
 * completed channel cost. Both {@code double[][]} and {@code float[][]} inputs
 * are supported without slicing or conversion.</p>
 */
public final class CID_I implements Serializable {

    @Serial
    private static final long serialVersionUID = 1L;

    private final CID cid;

    public CID_I() {
        cid = new CID();
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
            double channelCost = cid.distance(
                    first[dimension],
                    second[dimension],
                    remainingBudget(bestSoFar, total)
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
            double channelCost = cid.distance(
                    first[dimension],
                    second[dimension],
                    remainingBudget(bestSoFar, total)
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
                "Independent CID requires matching double[][] or float[][] "
                        + "inputs. Received " + typeName(first) + " and "
                        + typeName(second) + "."
        );
    }

    private static String typeName(Object value) {
        return value == null
                ? "null"
                : value.getClass().getTypeName();
    }
}
