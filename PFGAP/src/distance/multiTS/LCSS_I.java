package distance.multiTS;

import core.contracts.ObjectDataset;
import distance.elastic.LCSS;

import java.io.Serial;
import java.io.Serializable;
import java.util.Random;

/** Independent multivariate LCSS with summed selected-channel distances. */
public final class LCSS_I implements Serializable {

    @Serial
    private static final long serialVersionUID = 1L;

    private final LCSS lcss;

    public LCSS_I() {
        lcss = new LCSS();
    }

    public double distance(
            Object first,
            Object second,
            double bestSoFar,
            int windowSize,
            double epsilon
    ) {
        return distance(
                first,
                second,
                bestSoFar,
                windowSize,
                epsilon,
                null
        );
    }

    public double distance(
            Object first,
            Object second,
            double bestSoFar,
            int windowSize,
            double epsilon,
            int[] selectedDimensions
    ) {
        if (first instanceof double[][] firstValues
                && second instanceof double[][] secondValues) {
            return distance(
                    firstValues,
                    secondValues,
                    bestSoFar,
                    windowSize,
                    epsilon,
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
                    epsilon,
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
            double epsilon,
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
            total += lcss.distance(
                    first[dimension],
                    second[dimension],
                    Double.POSITIVE_INFINITY,
                    windowSize,
                    epsilon
            );
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
            double epsilon,
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
            total += lcss.distance(
                    first[dimension],
                    second[dimension],
                    Double.POSITIVE_INFINITY,
                    windowSize,
                    epsilon
            );
            if (total > bestSoFar) {
                return Double.POSITIVE_INFINITY;
            }
        }
        return total;
    }

    public int get_random_window(ObjectDataset dataset, Random random) {
        return lcss.get_random_window(dataset, random);
    }

    public double get_random_epsilon(ObjectDataset dataset, Random random) {
        return lcss.get_random_epsilon(dataset, random);
    }

    private static IllegalArgumentException unsupportedPair(
            Object first,
            Object second
    ) {
        return new IllegalArgumentException(
                "Independent LCSS requires matching double[][] or float[][] "
                        + "inputs. Received " + typeName(first) + " and "
                        + typeName(second) + "."
        );
    }

    private static String typeName(Object value) {
        return value == null ? "null" : value.getClass().getTypeName();
    }
}
