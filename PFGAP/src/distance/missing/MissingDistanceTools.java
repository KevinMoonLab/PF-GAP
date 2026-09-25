package distance.missing;

import java.util.Objects;

/**
 * Primitive numeric utilities shared by missing-value-aware distances.
 *
 * <p>Supported numeric representations are {@code double[]}, {@code float[]},
 * {@code double[][]}, and {@code float[][]}. Missing numeric values are encoded
 * only as {@link Double#NaN} or {@link Float#NaN}. Boxed numeric arrays, null
 * numeric values, and generic numeric {@code Object[]} arrays are not part of
 * the supported numeric contract.</p>
 *
 * <p>This class contains representation-neutral primitives such as missingness
 * tests, local squared costs, shape validation, availability counts, and the
 * NaN-Euclidean scaling factor. It deliberately contains no DTW recurrence,
 * alignment, normalization, or imputation policy.</p>
 */
public final class MissingDistanceTools {

    private MissingDistanceTools() {
        // Utility class.
    }

    /** Returns whether a primitive double value is missing. */
    public static boolean isMissing(double value) {
        return Double.isNaN(value);
    }

    /** Returns whether a primitive float value is missing. */
    public static boolean isMissing(float value) {
        return Float.isNaN(value);
    }

    /** Returns the squared difference between two present double values. */
    public static double squaredDifference(
            double first,
            double second
    ) {
        double difference = first - second;
        return difference * difference;
    }

    /**
     * Returns the squared difference between two present float values.
     *
     * <p>Each source value is widened before subtraction, and the result is
     * accumulated as double by the calling distance.</p>
     */
    public static double squaredDifference(
            float first,
            float second
    ) {
        double difference =
                (double) first - (double) second;
        return difference * difference;
    }

    /**
     * Returns the squared difference when both double values are present;
     * otherwise returns zero.
     */
    public static double squaredDifferenceIfPresent(
            double first,
            double second
    ) {
        if (Double.isNaN(first) || Double.isNaN(second)) {
            return 0.0;
        }
        return squaredDifference(first, second);
    }

    /**
     * Returns the squared difference when both float values are present;
     * otherwise returns zero.
     */
    public static double squaredDifferenceIfPresent(
            float first,
            float second
    ) {
        if (Float.isNaN(first) || Float.isNaN(second)) {
            return 0.0;
        }
        return squaredDifference(first, second);
    }

    /** Validates equal lengths for two double vectors. */
    public static void validateSameLength(
            double[] first,
            double[] second
    ) {
        Objects.requireNonNull(first, "First series cannot be null.");
        Objects.requireNonNull(second, "Second series cannot be null.");
        requireSameLength(first.length, second.length);
    }

    /** Validates equal lengths for two float vectors. */
    public static void validateSameLength(
            float[] first,
            float[] second
    ) {
        Objects.requireNonNull(first, "First series cannot be null.");
        Objects.requireNonNull(second, "Second series cannot be null.");
        requireSameLength(first.length, second.length);
    }

    /** Validates equal dimension counts for two double matrices. */
    public static void validateSameRows(
            double[][] first,
            double[][] second
    ) {
        Objects.requireNonNull(first, "First series cannot be null.");
        Objects.requireNonNull(second, "Second series cannot be null.");
        requireSameRows(first.length, second.length);
    }

    /** Validates equal dimension counts for two float matrices. */
    public static void validateSameRows(
            float[][] first,
            float[][] second
    ) {
        Objects.requireNonNull(first, "First series cannot be null.");
        Objects.requireNonNull(second, "Second series cannot be null.");
        requireSameRows(first.length, second.length);
    }

    /** Validates equal dimension counts and corresponding row lengths. */
    public static void validateSameShape(
            double[][] first,
            double[][] second
    ) {
        validateSameRows(first, second);
        for (int dimension = 0;
                dimension < first.length;
                dimension++) {
            double[] firstRow = requireRow(
                    first[dimension],
                    "first",
                    dimension
            );
            double[] secondRow = requireRow(
                    second[dimension],
                    "second",
                    dimension
            );
            requireSameRowLength(
                    firstRow.length,
                    secondRow.length,
                    dimension
            );
        }
    }

    /** Validates equal dimension counts and corresponding row lengths. */
    public static void validateSameShape(
            float[][] first,
            float[][] second
    ) {
        validateSameRows(first, second);
        for (int dimension = 0;
                dimension < first.length;
                dimension++) {
            float[] firstRow = requireRow(
                    first[dimension],
                    "first",
                    dimension
            );
            float[] secondRow = requireRow(
                    second[dimension],
                    "second",
                    dimension
            );
            requireSameRowLength(
                    firstRow.length,
                    secondRow.length,
                    dimension
            );
        }
    }

    /** Counts present values in a double vector. */
    public static int countAvailable(double[] series) {
        Objects.requireNonNull(series, "Series cannot be null.");
        int count = 0;
        for (double value : series) {
            if (!Double.isNaN(value)) {
                count++;
            }
        }
        return count;
    }

    /** Counts present values in a float vector. */
    public static int countAvailable(float[] series) {
        Objects.requireNonNull(series, "Series cannot be null.");
        int count = 0;
        for (float value : series) {
            if (!Float.isNaN(value)) {
                count++;
            }
        }
        return count;
    }

    /** Counts present values across a double matrix. */
    public static int countAvailable(double[][] series) {
        Objects.requireNonNull(series, "Series cannot be null.");
        int count = 0;
        for (int dimension = 0;
                dimension < series.length;
                dimension++) {
            count += countAvailable(
                    requireRow(series[dimension], "series", dimension)
            );
        }
        return count;
    }

    /** Counts present values across a float matrix. */
    public static int countAvailable(float[][] series) {
        Objects.requireNonNull(series, "Series cannot be null.");
        int count = 0;
        for (int dimension = 0;
                dimension < series.length;
                dimension++) {
            count += countAvailable(
                    requireRow(series[dimension], "series", dimension)
            );
        }
        return count;
    }

    /** Counts jointly observed positions in equal-length double vectors. */
    public static int countJointlyObserved(
            double[] first,
            double[] second
    ) {
        validateSameLength(first, second);
        int count = 0;
        for (int index = 0; index < first.length; index++) {
            if (!Double.isNaN(first[index])
                    && !Double.isNaN(second[index])) {
                count++;
            }
        }
        return count;
    }

    /** Counts jointly observed positions in equal-length float vectors. */
    public static int countJointlyObserved(
            float[] first,
            float[] second
    ) {
        validateSameLength(first, second);
        int count = 0;
        for (int index = 0; index < first.length; index++) {
            if (!Float.isNaN(first[index])
                    && !Float.isNaN(second[index])) {
                count++;
            }
        }
        return count;
    }

    /**
     * Counts jointly observed positions across equal-shaped double matrices.
     */
    public static int countJointlyObserved(
            double[][] first,
            double[][] second
    ) {
        validateSameShape(first, second);
        int count = 0;
        for (int dimension = 0;
                dimension < first.length;
                dimension++) {
            count += countJointlyObserved(
                    first[dimension],
                    second[dimension]
            );
        }
        return count;
    }

    /**
     * Counts jointly observed positions across equal-shaped float matrices.
     */
    public static int countJointlyObserved(
            float[][] first,
            float[][] second
    ) {
        validateSameShape(first, second);
        int count = 0;
        for (int dimension = 0;
                dimension < first.length;
                dimension++) {
            count += countJointlyObserved(
                    first[dimension],
                    second[dimension]
            );
        }
        return count;
    }

    /**
     * Returns the NaN-Euclidean scale factor
     * {@code fullLength / jointlyObservedLength}.
     *
     * <p>If no positions are jointly observed, positive infinity is returned.
     * This makes the absence of comparable evidence explicit to the calling
     * distance.</p>
     */
    public static double scaleFactor(
            int fullLength,
            int jointlyObservedLength
    ) {
        if (fullLength < 0) {
            throw new IllegalArgumentException(
                    "Full length cannot be negative: " + fullLength
            );
        }
        if (jointlyObservedLength < 0) {
            throw new IllegalArgumentException(
                    "Jointly observed length cannot be negative: "
                            + jointlyObservedLength
            );
        }
        if (jointlyObservedLength > fullLength) {
            throw new IllegalArgumentException(
                    "Jointly observed length cannot exceed full length. "
                            + "Received jointlyObservedLength="
                            + jointlyObservedLength
                            + " and fullLength="
                            + fullLength
                            + "."
            );
        }
        if (jointlyObservedLength == 0) {
            return Double.POSITIVE_INFINITY;
        }
        return (double) fullLength / jointlyObservedLength;
    }

    private static void requireSameLength(
            int firstLength,
            int secondLength
    ) {
        if (firstLength != secondLength) {
            throw new IllegalArgumentException(
                    "Both series must have the same length. Received "
                            + firstLength
                            + " and "
                            + secondLength
                            + "."
            );
        }
    }

    private static void requireSameRows(
            int firstRows,
            int secondRows
    ) {
        if (firstRows != secondRows) {
            throw new IllegalArgumentException(
                    "Both multivariate series must have the same number "
                            + "of dimensions. Received "
                            + firstRows
                            + " and "
                            + secondRows
                            + "."
            );
        }
    }

    private static void requireSameRowLength(
            int firstLength,
            int secondLength,
            int dimension
    ) {
        if (firstLength != secondLength) {
            throw new IllegalArgumentException(
                    "Corresponding rows must have the same length at "
                            + "dimension "
                            + dimension
                            + ". Received "
                            + firstLength
                            + " and "
                            + secondLength
                            + "."
            );
        }
    }

    private static double[] requireRow(
            double[] row,
            String seriesName,
            int dimension
    ) {
        return Objects.requireNonNull(
                row,
                "The "
                        + seriesName
                        + " series contains a null row at dimension "
                        + dimension
                        + "."
        );
    }

    private static float[] requireRow(
            float[] row,
            String seriesName,
            int dimension
    ) {
        return Objects.requireNonNull(
                row,
                "The "
                        + seriesName
                        + " series contains a null row at dimension "
                        + dimension
                        + "."
        );
    }
}
