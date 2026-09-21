package distance.elastic;

import java.io.Serial;
import java.io.Serializable;

/**
 * Euclidean distance kernels for primitive double and float vectors.
 *
 * <p>Overloads receiving {@code bestSoFar} return squared Euclidean distance
 * and may stop once the accumulated value exceeds that bound. Overloads
 * without {@code bestSoFar} return ordinary Euclidean distance.</p>
 *
 * <p>Float inputs remain stored as {@code float[]}. Values are widened
 * individually for double-precision subtraction and accumulation; no temporary
 * {@code double[]} is allocated.</p>
 *
 * <p>Methods remain synchronized pending the project-wide distance ownership
 * and concurrency audit.</p>
 */
public class Euclidean implements Serializable {

    @Serial
    private static final long serialVersionUID = 1L;

    public Euclidean() {
    }

    public synchronized double distance(
            Object first,
            Object second,
            double bestSoFar
    ) {
        if (first instanceof double[] firstValues
                && second instanceof double[] secondValues) {
            requireSameLength(firstValues.length, secondValues.length);
            return squaredDistance(
                    firstValues,
                    secondValues,
                    bestSoFar
            );
        }
        if (first instanceof float[] firstValues
                && second instanceof float[] secondValues) {
            requireSameLength(firstValues.length, secondValues.length);
            return squaredDistance(
                    firstValues,
                    secondValues,
                    bestSoFar
            );
        }
        throw unsupportedPair(first, second);
    }

    public synchronized double distance(
            Object first,
            Object second,
            double bestSoFar,
            int[] selectedDimensions
    ) {
        if (selectedDimensions == null) {
            return distance(first, second, bestSoFar);
        }
        if (selectedDimensions.length == 0) {
            throw new IllegalArgumentException(
                    "Selected dimensions cannot be empty."
            );
        }

        if (first instanceof double[] firstValues
                && second instanceof double[] secondValues) {
            requireSameLength(firstValues.length, secondValues.length);
            return squaredDistance(
                    firstValues,
                    secondValues,
                    bestSoFar,
                    selectedDimensions
            );
        }
        if (first instanceof float[] firstValues
                && second instanceof float[] secondValues) {
            requireSameLength(firstValues.length, secondValues.length);
            return squaredDistance(
                    firstValues,
                    secondValues,
                    bestSoFar,
                    selectedDimensions
            );
        }
        throw unsupportedPair(first, second);
    }

    public synchronized double distance(
            Object first,
            Object second
    ) {
        return Math.sqrt(
                distance(
                        first,
                        second,
                        Double.POSITIVE_INFINITY
                )
        );
    }

    public synchronized double distance(
            Object first,
            Object second,
            int[] selectedDimensions
    ) {
        return Math.sqrt(
                distance(
                        first,
                        second,
                        Double.POSITIVE_INFINITY,
                        selectedDimensions
                )
        );
    }

    private static double squaredDistance(
            double[] first,
            double[] second,
            double bestSoFar
    ) {
        double total = 0.0;
        for (int index = 0;
                index < first.length && total <= bestSoFar;
                index++) {
            double difference = first[index] - second[index];
            total += difference * difference;
        }
        return total;
    }

    private static double squaredDistance(
            float[] first,
            float[] second,
            double bestSoFar
    ) {
        double total = 0.0;
        for (int index = 0;
                index < first.length && total <= bestSoFar;
                index++) {
            double difference =
                    (double) first[index] - (double) second[index];
            total += difference * difference;
        }
        return total;
    }

    private static double squaredDistance(
            double[] first,
            double[] second,
            double bestSoFar,
            int[] selectedDimensions
    ) {
        double total = 0.0;
        for (int position = 0;
                position < selectedDimensions.length
                        && total <= bestSoFar;
                position++) {
            int feature = selectedDimensions[position];
            requireSelectedFeature(feature, first.length);
            double difference = first[feature] - second[feature];
            total += difference * difference;
        }
        return total;
    }

    private static double squaredDistance(
            float[] first,
            float[] second,
            double bestSoFar,
            int[] selectedDimensions
    ) {
        double total = 0.0;
        for (int position = 0;
                position < selectedDimensions.length
                        && total <= bestSoFar;
                position++) {
            int feature = selectedDimensions[position];
            requireSelectedFeature(feature, first.length);
            double difference =
                    (double) first[feature]
                            - (double) second[feature];
            total += difference * difference;
        }
        return total;
    }

    private static void requireSameLength(
            int firstLength,
            int secondLength
    ) {
        if (firstLength != secondLength) {
            throw new IllegalArgumentException(
                    "Euclidean distance requires equal-length vectors. "
                            + "Received "
                            + firstLength
                            + " and "
                            + secondLength
                            + "."
            );
        }
    }

    private static void requireSelectedFeature(
            int feature,
            int vectorLength
    ) {
        if (feature < 0 || feature >= vectorLength) {
            throw new IllegalArgumentException(
                    "Selected feature index "
                            + feature
                            + " is outside vector length "
                            + vectorLength
                            + "."
            );
        }
    }

    private static IllegalArgumentException unsupportedPair(
            Object first,
            Object second
    ) {
        return new IllegalArgumentException(
                "Euclidean distance requires matching double[] or float[] "
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
