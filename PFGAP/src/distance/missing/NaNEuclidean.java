package distance.missing;

import java.io.Serial;
import java.io.Serializable;

/**
 * Missing-value-aware Euclidean distance for primitive univariate series.
 *
 * <p>The squared distance is {@code (M / M_obs) * sum((x_i-y_i)^2)},
 * where {@code M_obs} counts positions observed in both vectors. If no
 * positions are jointly observed, the result is positive infinity.</p>
 *
 * <p>Supported pairs are matching {@code double[]} arrays or matching
 * {@code float[]} arrays. Missing values use primitive NaN. Mixed precision
 * and boxed numeric arrays are unsupported. Float values are widened
 * individually for double-precision arithmetic without array conversion.</p>
 */
public class NaNEuclidean implements Serializable {

    @Serial
    private static final long serialVersionUID = 1L;

    public NaNEuclidean() {
    }

    public synchronized double distance(Object first, Object second) {
        return distance(first, second, Double.POSITIVE_INFINITY);
    }

    /**
     * Computes ordinary NaN-Euclidean distance. Exact early abandoning is not
     * used because the final scale depends on the final joint-observation count.
     */
    public synchronized double distance(
            Object first,
            Object second,
            double bestSoFar
    ) {
        validateBestSoFar(bestSoFar);
        double squared = squaredDistance(first, second);
        if (Double.isInfinite(squared)) {
            return Double.POSITIVE_INFINITY;
        }
        double result = Math.sqrt(squared);
        return result > bestSoFar
                ? Double.POSITIVE_INFINITY
                : result;
    }

    public synchronized double squaredDistance(
            Object first,
            Object second
    ) {
        if (first instanceof double[] firstValues
                && second instanceof double[] secondValues) {
            return squaredDistanceDouble(firstValues, secondValues);
        }
        if (first instanceof float[] firstValues
                && second instanceof float[] secondValues) {
            return squaredDistanceFloat(firstValues, secondValues);
        }
        throw unsupportedPair(first, second);
    }

    public synchronized boolean isComputable(
            Object first,
            Object second
    ) {
        if (first instanceof double[] firstValues
                && second instanceof double[] secondValues) {
            return MissingDistanceTools.countJointlyObserved(
                    firstValues,
                    secondValues
            ) > 0;
        }
        if (first instanceof float[] firstValues
                && second instanceof float[] secondValues) {
            return MissingDistanceTools.countJointlyObserved(
                    firstValues,
                    secondValues
            ) > 0;
        }
        throw unsupportedPair(first, second);
    }

    private static double squaredDistanceDouble(
            double[] first,
            double[] second
    ) {
        MissingDistanceTools.validateSameLength(first, second);
        double sum = 0.0;
        int jointlyObserved = 0;
        for (int index = 0; index < first.length; index++) {
            double firstValue = first[index];
            double secondValue = second[index];
            if (Double.isNaN(firstValue)
                    || Double.isNaN(secondValue)) {
                continue;
            }
            sum += MissingDistanceTools.squaredDifference(
                    firstValue,
                    secondValue
            );
            jointlyObserved++;
        }
        return scaleSquaredDistance(first.length, jointlyObserved, sum);
    }

    private static double squaredDistanceFloat(
            float[] first,
            float[] second
    ) {
        MissingDistanceTools.validateSameLength(first, second);
        double sum = 0.0;
        int jointlyObserved = 0;
        for (int index = 0; index < first.length; index++) {
            float firstValue = first[index];
            float secondValue = second[index];
            if (Float.isNaN(firstValue)
                    || Float.isNaN(secondValue)) {
                continue;
            }
            sum += MissingDistanceTools.squaredDifference(
                    firstValue,
                    secondValue
            );
            jointlyObserved++;
        }
        return scaleSquaredDistance(first.length, jointlyObserved, sum);
    }

    private static double scaleSquaredDistance(
            int fullLength,
            int jointlyObserved,
            double sum
    ) {
        if (jointlyObserved == 0) {
            return Double.POSITIVE_INFINITY;
        }
        return MissingDistanceTools.scaleFactor(
                fullLength,
                jointlyObserved
        ) * sum;
    }

    private static void validateBestSoFar(double bestSoFar) {
        if (Double.isNaN(bestSoFar) || bestSoFar < 0.0) {
            throw new IllegalArgumentException(
                    "NaNEuclidean bestSoFar must be nonnegative and not NaN. "
                            + "Received: " + bestSoFar + "."
            );
        }
    }

    private static IllegalArgumentException unsupportedPair(
            Object first,
            Object second
    ) {
        return new IllegalArgumentException(
                "NaNEuclidean requires matching double[] or float[] inputs. "
                        + "Received "
                        + typeName(first)
                        + " and "
                        + typeName(second)
                        + ". Mixed float/double pairs and boxed numeric arrays "
                        + "are not supported."
        );
    }

    private static String typeName(Object value) {
        return value == null ? "null" : value.getClass().getTypeName();
    }
}
