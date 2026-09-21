package distance.missing;

import core.contracts.ObjectDataset;

import java.io.Serial;
import java.io.Serializable;
import java.util.Objects;
import java.util.Random;

/**
 * Independent DTW-AROW distance for primitive dimension-major multivariate
 * numeric time series containing missing values.
 *
 * <p>Each dimension in the first series is compared with the corresponding
 * dimension in the second series using {@link DTWAROW}. The component
 * distances are averaged across dimensions.</p>
 *
 * <p>Supported input pairs are matching {@code double[][]} arrays or matching
 * {@code float[][]} arrays. Missing numeric values use primitive NaN. Unequal
 * time lengths are supported within corresponding dimensions. Mixed
 * float/double pairs, boxed numeric arrays, and generic numeric object arrays
 * are not supported.</p>
 *
 * <p>If either series has no dimensions, or if any component DTW-AROW distance
 * is unavailable, the final result is positive infinity. The best-so-far
 * threshold is applied after the final average is computed, because an
 * individual component distance may exceed the final average threshold without
 * proving that the completed average will do so.</p>
 *
 * <p>Methods remain synchronized pending the broader distance ownership and
 * concurrency audit.</p>
 */
public class DTWAROW_I implements Serializable {

    @Serial
    private static final long serialVersionUID = 1L;

    private final DTWAROW dtwArow;

    public DTWAROW_I() {
        this.dtwArow = new DTWAROW();
    }

    /** Computes unconstrained independent DTW-AROW distance. */
    public synchronized double distance(
            Object first,
            Object second
    ) {
        return distance(
                first,
                second,
                Double.POSITIVE_INFINITY,
                -1
        );
    }

    /**
     * Computes unconstrained independent DTW-AROW distance with a best-so-far
     * threshold.
     */
    public synchronized double distance(
            Object first,
            Object second,
            double bestSoFar
    ) {
        return distance(first, second, bestSoFar, -1);
    }

    /**
     * Computes the average DTW-AROW distance across corresponding dimensions.
     *
     * @param first first {@code double[][]} or {@code float[][]} series
     * @param second matching second series
     * @param bestSoFar nonnegative ordinary-distance threshold
     * @param windowSize Sakoe-Chiba radius; {@code -1} means unconstrained
     * @return average component distance, or positive infinity when unavailable
     *         or greater than {@code bestSoFar}
     */
    public synchronized double distance(
            Object first,
            Object second,
            double bestSoFar,
            int windowSize
    ) {
        validateBestSoFar(bestSoFar);
        validateWindow(windowSize);

        double result;
        if (first instanceof double[][] firstValues
                && second instanceof double[][] secondValues) {
            result = distanceDouble(
                    firstValues,
                    secondValues,
                    windowSize
            );
        } else if (first instanceof float[][] firstValues
                && second instanceof float[][] secondValues) {
            result = distanceFloat(
                    firstValues,
                    secondValues,
                    windowSize
            );
        } else {
            throw unsupportedPair(first, second);
        }

        return result > bestSoFar
                ? Double.POSITIVE_INFINITY
                : result;
    }

    /**
     * Returns whether every corresponding component has an available
     * DTW-AROW alignment under the requested window.
     */
    public synchronized boolean isComputable(
            Object first,
            Object second,
            int windowSize
    ) {
        validateWindow(windowSize);
        if (first instanceof double[][] firstValues
                && second instanceof double[][] secondValues) {
            validateDoubleDimensions(firstValues, secondValues);
            if (firstValues.length == 0) {
                return false;
            }
            for (int dimension = 0;
                    dimension < firstValues.length;
                    dimension++) {
                if (Double.isInfinite(
                        dtwArow.distance(
                                firstValues[dimension],
                                secondValues[dimension],
                                Double.POSITIVE_INFINITY,
                                windowSize
                        )
                )) {
                    return false;
                }
            }
            return true;
        }

        if (first instanceof float[][] firstValues
                && second instanceof float[][] secondValues) {
            validateFloatDimensions(firstValues, secondValues);
            if (firstValues.length == 0) {
                return false;
            }
            for (int dimension = 0;
                    dimension < firstValues.length;
                    dimension++) {
                if (Double.isInfinite(
                        dtwArow.distance(
                                firstValues[dimension],
                                secondValues[dimension],
                                Double.POSITIVE_INFINITY,
                                windowSize
                        )
                )) {
                    return false;
                }
            }
            return true;
        }

        throw unsupportedPair(first, second);
    }

    /**
     * Convenience computability check using an unconstrained window.
     */
    public synchronized boolean isComputable(
            Object first,
            Object second
    ) {
        return isComputable(first, second, -1);
    }

    private double distanceDouble(
            double[][] first,
            double[][] second,
            int windowSize
    ) {
        validateDoubleDimensions(first, second);
        if (first.length == 0) {
            return Double.POSITIVE_INFINITY;
        }

        double totalDistance = 0.0;
        for (int dimension = 0;
                dimension < first.length;
                dimension++) {
            double componentDistance = dtwArow.distance(
                    first[dimension],
                    second[dimension],
                    Double.POSITIVE_INFINITY,
                    windowSize
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
            float[][] second,
            int windowSize
    ) {
        validateFloatDimensions(first, second);
        if (first.length == 0) {
            return Double.POSITIVE_INFINITY;
        }

        double totalDistance = 0.0;
        for (int dimension = 0;
                dimension < first.length;
                dimension++) {
            double componentDistance = dtwArow.distance(
                    first[dimension],
                    second[dimension],
                    Double.POSITIVE_INFINITY,
                    windowSize
            );
            if (Double.isInfinite(componentDistance)) {
                return Double.POSITIVE_INFINITY;
            }
            totalDistance += componentDistance;
        }
        return totalDistance / first.length;
    }

    private static void validateDoubleDimensions(
            double[][] first,
            double[][] second
    ) {
        MissingDistanceTools.validateSameRows(first, second);
        for (int dimension = 0;
                dimension < first.length;
                dimension++) {
            Objects.requireNonNull(
                    first[dimension],
                    nullDimensionMessage("first", dimension)
            );
            Objects.requireNonNull(
                    second[dimension],
                    nullDimensionMessage("second", dimension)
            );
        }
    }

    private static void validateFloatDimensions(
            float[][] first,
            float[][] second
    ) {
        MissingDistanceTools.validateSameRows(first, second);
        for (int dimension = 0;
                dimension < first.length;
                dimension++) {
            Objects.requireNonNull(
                    first[dimension],
                    nullDimensionMessage("first", dimension)
            );
            Objects.requireNonNull(
                    second[dimension],
                    nullDimensionMessage("second", dimension)
            );
        }
    }

    /** Delegates random window selection to the univariate distance. */
    public int get_random_window(
            ObjectDataset dataset,
            Random random
    ) {
        return dtwArow.get_random_window(
                Objects.requireNonNull(
                        dataset,
                        "Dataset cannot be null."
                ),
                Objects.requireNonNull(
                        random,
                        "Random cannot be null."
                )
        );
    }

    private static void validateBestSoFar(
            double bestSoFar
    ) {
        if (Double.isNaN(bestSoFar) || bestSoFar < 0.0) {
            throw new IllegalArgumentException(
                    "DTWAROW_I bestSoFar must be nonnegative and not NaN. "
                            + "Received: "
                            + bestSoFar
                            + "."
            );
        }
    }

    private static void validateWindow(int windowSize) {
        if (windowSize < -1) {
            throw new IllegalArgumentException(
                    "windowSize must be -1 or a nonnegative integer."
            );
        }
    }

    private static String nullDimensionMessage(
            String seriesName,
            int dimension
    ) {
        return "The "
                + seriesName
                + " series contains a null row at dimension "
                + dimension
                + ".";
    }

    private static IllegalArgumentException unsupportedPair(
            Object first,
            Object second
    ) {
        return new IllegalArgumentException(
                "DTWAROW_I requires matching double[][] or float[][] inputs. "
                        + "Received "
                        + typeName(first)
                        + " and "
                        + typeName(second)
                        + ". Mixed float/double pairs and boxed numeric arrays "
                        + "are not supported."
        );
    }

    private static String typeName(Object value) {
        return value == null
                ? "null"
                : value.getClass().getTypeName();
    }
}
