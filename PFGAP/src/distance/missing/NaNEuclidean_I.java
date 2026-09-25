package distance.missing;

import java.io.Serial;
import java.io.Serializable;
import java.util.Objects;

/**
 * Independent multivariate missing-value-aware Euclidean distance.
 *
 * <p>Each selected outer dimension is compared independently with
 * {@link NaNEuclidean}, and the resulting ordinary component distances are
 * summed. Selected dimension indices are never forwarded to the univariate
 * evaluator or interpreted as temporal indices.</p>
 *
 * <p>Matching {@code double[][]} and matching {@code float[][]} inputs are
 * supported without slicing or conversion. If any selected component has no
 * jointly observed temporal position, the independent distance is unavailable.
 * A null selected-dimension array evaluates every dimension.</p>
 *
 * <p>The class contains no mutable calculation state and is safe for concurrent
 * use. The univariate evaluator automatically uses its scalar or Vector API
 * kernel according to {@code AppContext.useVectorApi}.</p>
 */
public final class NaNEuclidean_I implements Serializable {

    @Serial
    private static final long serialVersionUID = 1L;

    private final NaNEuclidean nanEuclidean;

    public NaNEuclidean_I() {
        nanEuclidean = new NaNEuclidean();
    }

    public double distance(Object first, Object second) {
        return distance(
                first,
                second,
                Double.POSITIVE_INFINITY,
                null
        );
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
        validateBestSoFar(bestSoFar);

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

    public boolean isComputable(Object first, Object second) {
        return isComputable(first, second, null);
    }

    public boolean isComputable(
            Object first,
            Object second,
            int[] selectedDimensions
    ) {
        if (first instanceof double[][] firstValues
                && second instanceof double[][] secondValues) {
            validateRows(firstValues, secondValues);
            int count = selectedCount(
                    firstValues.length,
                    selectedDimensions
            );
            if (count == 0) {
                return false;
            }
            for (int position = 0; position < count; position++) {
                int dimension = selectedDimension(
                        position,
                        selectedDimensions
                );
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
            validateRows(firstValues, secondValues);
            int count = selectedCount(
                    firstValues.length,
                    selectedDimensions
            );
            if (count == 0) {
                return false;
            }
            for (int position = 0; position < count; position++) {
                int dimension = selectedDimension(
                        position,
                        selectedDimensions
                );
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

    private double distance(
            double[][] first,
            double[][] second,
            double bestSoFar,
            int[] selectedDimensions
    ) {
        validateRows(first, second);
        int count = selectedCount(first.length, selectedDimensions);
        if (count == 0) {
            return Double.POSITIVE_INFINITY;
        }

        double total = 0.0;
        for (int position = 0; position < count; position++) {
            int dimension = selectedDimension(
                    position,
                    selectedDimensions
            );
            double component = nanEuclidean.distance(
                    first[dimension],
                    second[dimension],
                    remainingBudget(bestSoFar, total)
            );
            if (Double.isInfinite(component)) {
                return Double.POSITIVE_INFINITY;
            }
            total += component;
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
        validateRows(first, second);
        int count = selectedCount(first.length, selectedDimensions);
        if (count == 0) {
            return Double.POSITIVE_INFINITY;
        }

        double total = 0.0;
        for (int position = 0; position < count; position++) {
            int dimension = selectedDimension(
                    position,
                    selectedDimensions
            );
            double component = nanEuclidean.distance(
                    first[dimension],
                    second[dimension],
                    remainingBudget(bestSoFar, total)
            );
            if (Double.isInfinite(component)) {
                return Double.POSITIVE_INFINITY;
            }
            total += component;
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

    private static int selectedCount(
            int dimensions,
            int[] selectedDimensions
    ) {
        return selectedDimensions == null
                ? dimensions
                : selectedDimensions.length;
    }

    private static int selectedDimension(
            int position,
            int[] selectedDimensions
    ) {
        return selectedDimensions == null
                ? position
                : selectedDimensions[position];
    }

    private static void validateRows(
            double[][] first,
            double[][] second
    ) {
        MissingDistanceTools.validateSameRows(first, second);
        for (int dimension = 0; dimension < first.length; dimension++) {
            Objects.requireNonNull(
                    first[dimension],
                    nullDimensionMessage("first", dimension)
            );
            Objects.requireNonNull(
                    second[dimension],
                    nullDimensionMessage("second", dimension)
            );
            MissingDistanceTools.validateSameLength(
                    first[dimension],
                    second[dimension]
            );
        }
    }

    private static void validateRows(
            float[][] first,
            float[][] second
    ) {
        MissingDistanceTools.validateSameRows(first, second);
        for (int dimension = 0; dimension < first.length; dimension++) {
            Objects.requireNonNull(
                    first[dimension],
                    nullDimensionMessage("first", dimension)
            );
            Objects.requireNonNull(
                    second[dimension],
                    nullDimensionMessage("second", dimension)
            );
            MissingDistanceTools.validateSameLength(
                    first[dimension],
                    second[dimension]
            );
        }
    }

    private static void validateBestSoFar(double bestSoFar) {
        if (Double.isNaN(bestSoFar) || bestSoFar < 0.0) {
            throw new IllegalArgumentException(
                    "NaNEuclidean_I bestSoFar must be nonnegative and not "
                            + "NaN. Received: " + bestSoFar + "."
            );
        }
    }

    private static String nullDimensionMessage(
            String seriesName,
            int dimension
    ) {
        return "The " + seriesName
                + " series contains a null row at dimension "
                + dimension + ".";
    }

    private static IllegalArgumentException unsupportedPair(
            Object first,
            Object second
    ) {
        return new IllegalArgumentException(
                "NaNEuclidean_I requires matching double[][] or float[][] "
                        + "inputs. Received " + typeName(first) + " and "
                        + typeName(second) + "."
        );
    }

    private static String typeName(Object value) {
        return value == null ? "null" : value.getClass().getTypeName();
    }
}
