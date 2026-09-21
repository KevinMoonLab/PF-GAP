package distance.missing;

import java.io.Serial;
import java.io.Serializable;

/**
 * Independent missing-value-aware Euclidean distance for primitive
 * dimension-major multivariate series.
 *
 * <p>Each dimension in the first series is compared with the corresponding
 * dimension in the second series using {@link NaNEuclidean}. The component
 * distances are then averaged across dimensions.</p>
 *
 * <p>Supported input pairs are matching {@code double[][]} arrays or matching
 * {@code float[][]} arrays. Missing numeric values use primitive NaN. Mixed
 * float/double pairs, boxed numeric arrays, and generic numeric object arrays
 * are not supported.</p>
 *
 * <p>If either series contains no dimensions, or if any corresponding
 * dimension has no jointly observed positions, the result is positive
 * infinity. The best-so-far threshold is applied only after the final average
 * is computed, because an individual component may exceed the final average
 * threshold without proving that the completed average will do so.</p>
 *
 * <p>Methods remain synchronized pending the broader distance ownership and
 * concurrency audit.</p>
 */
public class NaNEuclidean_I implements Serializable {

    @Serial
    private static final long serialVersionUID = 1L;

    private final NaNEuclidean nanEuclidean;

    public NaNEuclidean_I() {
        this.nanEuclidean = new NaNEuclidean();
    }

    /** Computes independent NaN-Euclidean distance. */
    public synchronized double distance(
            Object first,
            Object second
    ) {
        return distance(
                first,
                second,
                Double.POSITIVE_INFINITY
        );
    }

    /**
     * Computes the average NaN-Euclidean distance across corresponding
     * dimensions.
     */
    public synchronized double distance(
            Object first,
            Object second,
            double bestSoFar
    ) {
        validateBestSoFar(bestSoFar);

        double result;
        if (first instanceof double[][] firstValues
                && second instanceof double[][] secondValues) {
            result = distanceDouble(firstValues, secondValues);
        } else if (first instanceof float[][] firstValues
                && second instanceof float[][] secondValues) {
            result = distanceFloat(firstValues, secondValues);
        } else {
            throw unsupportedPair(first, second);
        }

        return result > bestSoFar
                ? Double.POSITIVE_INFINITY
                : result;
    }

    /**
     * Returns whether every corresponding dimension has at least one jointly
     * observed position.
     */
    public synchronized boolean isComputable(
            Object first,
            Object second
    ) {
        if (first instanceof double[][] firstValues
                && second instanceof double[][] secondValues) {
            validateDoubleRows(firstValues, secondValues);
            if (firstValues.length == 0) {
                return false;
            }
            for (int dimension = 0;
                    dimension < firstValues.length;
                    dimension++) {
                if (!nanEuclidean.isComputable(
                        firstValues[dimension],
                        secondValues[dimension]
                )) {
                    return false;
                }
            }
            return true;
        }

        if (first instanceof float[][] firstValues
                && second instanceof float[][] secondValues) {
            validateFloatRows(firstValues, secondValues);
            if (firstValues.length == 0) {
                return false;
            }
            for (int dimension = 0;
                    dimension < firstValues.length;
                    dimension++) {
                if (!nanEuclidean.isComputable(
                        firstValues[dimension],
                        secondValues[dimension]
                )) {
                    return false;
                }
            }
            return true;
        }

        throw unsupportedPair(first, second);
    }

    private double distanceDouble(
            double[][] first,
            double[][] second
    ) {
        validateDoubleRows(first, second);
        if (first.length == 0) {
            return Double.POSITIVE_INFINITY;
        }

        double totalDistance = 0.0;
        for (int dimension = 0;
                dimension < first.length;
                dimension++) {
            double componentDistance = nanEuclidean.distance(
                    first[dimension],
                    second[dimension],
                    Double.POSITIVE_INFINITY
            );
            if (Double.isInfinite(componentDistance)) {
                return Double.POSITIVE_INFINITY;
            }
            totalDistance += componentDistance;
        }
        return totalDistance / first.length;
    }

    private double distanceFloat(
            float[][] first,
            float[][] second
    ) {
        validateFloatRows(first, second);
        if (first.length == 0) {
            return Double.POSITIVE_INFINITY;
        }

        double totalDistance = 0.0;
        for (int dimension = 0;
                dimension < first.length;
                dimension++) {
            double componentDistance = nanEuclidean.distance(
                    first[dimension],
                    second[dimension],
                    Double.POSITIVE_INFINITY
            );
            if (Double.isInfinite(componentDistance)) {
                return Double.POSITIVE_INFINITY;
            }
            totalDistance += componentDistance;
        }
        return totalDistance / first.length;
    }

    private static void validateDoubleRows(
            double[][] first,
            double[][] second
    ) {
        MissingDistanceTools.validateSameRows(first, second);
        for (int dimension = 0;
                dimension < first.length;
                dimension++) {
            if (first[dimension] == null
                    || second[dimension] == null) {
                throw nullDimension(dimension);
            }
            MissingDistanceTools.validateSameLength(
                    first[dimension],
                    second[dimension]
            );
        }
    }

    private static void validateFloatRows(
            float[][] first,
            float[][] second
    ) {
        MissingDistanceTools.validateSameRows(first, second);
        for (int dimension = 0;
                dimension < first.length;
                dimension++) {
            if (first[dimension] == null
                    || second[dimension] == null) {
                throw nullDimension(dimension);
            }
            MissingDistanceTools.validateSameLength(
                    first[dimension],
                    second[dimension]
            );
        }
    }

    private static void validateBestSoFar(
            double bestSoFar
    ) {
        if (Double.isNaN(bestSoFar) || bestSoFar < 0.0) {
            throw new IllegalArgumentException(
                    "NaNEuclidean_I bestSoFar must be nonnegative and not "
                            + "NaN. Received: "
                            + bestSoFar
                            + "."
            );
        }
    }

    private static IllegalArgumentException nullDimension(
            int dimension
    ) {
        return new IllegalArgumentException(
                "NaNEuclidean_I encountered a null row at dimension "
                        + dimension
                        + "."
        );
    }

    private static IllegalArgumentException unsupportedPair(
            Object first,
            Object second
    ) {
        return new IllegalArgumentException(
                "NaNEuclidean_I requires matching double[][] or float[][] "
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
