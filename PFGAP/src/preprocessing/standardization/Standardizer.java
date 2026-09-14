package preprocessing.standardization;

import datasets.ListObjectDataset;
import datasets.readers.lazy.LazySeriesRef;

import java.util.List;
import java.util.Objects;

/**
 * Applies prepared standardization statistics to realized numeric data.
 *
 * <p>Supported representations are {@code double[]}, {@code Double[]},
 * {@code double[][]}, and {@code Double[][]}. Multivariate arrays use
 * dimension-major orientation: {@code data[dimension][time]}.</p>
 *
 * <p>For {@link StandardizationScope#PER_DIMENSION}, one-dimensional
 * {@code double[]} and {@code Double[]} instances are interpreted as tabular
 * rows. Each array position is standardized with the corresponding fitted
 * feature statistics.</p>
 *
 * <p>For {@link StandardizationScope#GLOBAL}, every accepted value uses the
 * single global center and scale.</p>
 *
 * <p>Transformation is performed in place and in one pass over the values.
 * This class intentionally does not perform a preliminary full-dataset
 * validation pass. Reader configuration and the configured numeric and
 * missing-value modes are treated as trusted input contracts. Cheap
 * structural checks are retained so configuration errors fail near their
 * source.</p>
 *
 * <p>{@code null} boxed values and {@code NaN} values are preserved. No
 * standardized copy of the dataset is created.</p>
 */
public final class Standardizer {

    private Standardizer() {
        // Utility class.
    }

    /**
     * Standardizes every realized instance in a dataset in place.
     *
     * <p>Centers and scales are defensively copied from the immutable
     * statistics object once for the entire dataset, rather than once for
     * every instance or dimension.</p>
     *
     * @param dataset eager numeric dataset
     * @param stats prepared standardization statistics
     * @param featureNames ordered realized-dimension names, or null when
     *                     unavailable
     * @return the same dataset instance
     */
    public static ListObjectDataset transformInPlace(
            ListObjectDataset dataset,
            StandardizationStats stats,
            List<String> featureNames
    ) {
        Objects.requireNonNull(
                dataset,
                "Dataset cannot be null."
        );

        Objects.requireNonNull(
                stats,
                "StandardizationStats cannot be null."
        );

        List<Object> data =
                Objects.requireNonNull(
                        dataset.getData(),
                        "Dataset data cannot be null."
                );

        stats.validateFeatureCompatibility(
                featureNames
        );

        StandardizationScope scope =
                stats.getScope();

        double[] centers =
                stats.getCenters();

        double[] scales =
                stats.getScales();

        validatePreparedStatistics(
                scope,
                centers,
                scales
        );

        for (Object instance : data) {
            transformInstanceInPlace(
                    instance,
                    scope,
                    centers,
                    scales
            );
        }

        return dataset;
    }

    /**
     * Standardizes every realized instance without named-feature validation.
     *
     * @param dataset eager numeric dataset
     * @param stats prepared standardization statistics
     * @return the same dataset instance
     */
    public static ListObjectDataset transformInPlace(
            ListObjectDataset dataset,
            StandardizationStats stats
    ) {
        Objects.requireNonNull(
                stats,
                "StandardizationStats cannot be null."
        );

        if (stats.hasFeatureNames()) {
            throw new IllegalArgumentException(
                    "Standardization statistics contain ordered feature "
                            + "names. Supply feature names to "
                            + "transformInPlace(dataset, stats, featureNames)."
            );
        }

        return transformInPlace(
                dataset,
                stats,
                null
        );
    }

    /**
     * Standardizes one realized numeric instance in place.
     *
     * <p>This method is suitable for individually materialized lazy
     * instances. Centers and scales are copied once for this transformation
     * call. Eager dataset transformation should use
     * {@link #transformInPlace(ListObjectDataset, StandardizationStats)} so
     * the copied arrays are reused across every instance.</p>
     *
     * @param series realized numeric instance
     * @param stats prepared standardization statistics
     * @return the same instance object
     */
    public static Object transformInstanceInPlace(
            Object series,
            StandardizationStats stats
    ) {
        Objects.requireNonNull(
                series,
                "Series cannot be null."
        );

        Objects.requireNonNull(
                stats,
                "StandardizationStats cannot be null."
        );

        if (series instanceof LazySeriesRef) {
            throw new UnsupportedOperationException(
                    "Cannot standardize a LazySeriesRef directly. "
                            + "Standardize the realized series after "
                            + "materialization."
            );
        }

        StandardizationScope scope =
                stats.getScope();

        double[] centers =
                stats.getCenters();

        double[] scales =
                stats.getScales();

        validatePreparedStatistics(
                scope,
                centers,
                scales
        );

        return transformInstanceInPlace(
                series,
                scope,
                centers,
                scales
        );
    }

    /**
     * Dispatches one realized instance to its representation-specific
     * transformation path.
     */
    private static Object transformInstanceInPlace(
            Object series,
            StandardizationScope scope,
            double[] centers,
            double[] scales
    ) {
        Objects.requireNonNull(
                series,
                "Series cannot be null."
        );

        if (series instanceof LazySeriesRef) {
            throw new UnsupportedOperationException(
                    "Cannot standardize a LazySeriesRef directly. "
                            + "Standardize the realized series after "
                            + "materialization."
            );
        }

        if (series instanceof double[] values) {
            transformPrimitiveOneDimensionalInstance(
                    values,
                    scope,
                    centers,
                    scales
            );

            return values;
        }

        if (series instanceof Double[] values) {
            transformBoxedOneDimensionalInstance(
                    values,
                    scope,
                    centers,
                    scales
            );

            return values;
        }

        if (series instanceof double[][] values) {
            transformPrimitiveMultivariateInstance(
                    values,
                    scope,
                    centers,
                    scales
            );

            return values;
        }

        if (series instanceof Double[][] values) {
            transformBoxedMultivariateInstance(
                    values,
                    scope,
                    centers,
                    scales
            );

            return values;
        }

        throw new IllegalArgumentException(
                "Unsupported standardization series type: "
                        + series.getClass().getTypeName()
                        + ". Expected double[], Double[], double[][], or "
                        + "Double[][]."
        );
    }

    /**
     * Transforms one primitive one-dimensional instance.
     *
     * <p>PER_DIMENSION interprets the instance as a tabular row.
     * GLOBAL applies the single prepared statistic group to every value.</p>
     */
    private static void transformPrimitiveOneDimensionalInstance(
            double[] values,
            StandardizationScope scope,
            double[] centers,
            double[] scales
    ) {
        if (scope == StandardizationScope.PER_DIMENSION) {
            requireDimensionCompatibility(
                    values.length,
                    centers.length,
                    "Tabular row"
            );

            transformPrimitiveTabularRow(
                    values,
                    centers,
                    scales
            );

            return;
        }

        requireGlobalStatistics(
                centers,
                scales
        );

        transformPrimitiveDimension(
                values,
                centers[0],
                scales[0]
        );
    }

    /**
     * Transforms one boxed one-dimensional instance.
     *
     * <p>PER_DIMENSION interprets the instance as a tabular row.
     * GLOBAL applies the single prepared statistic group to every value.</p>
     */
    private static void transformBoxedOneDimensionalInstance(
            Double[] values,
            StandardizationScope scope,
            double[] centers,
            double[] scales
    ) {
        if (scope == StandardizationScope.PER_DIMENSION) {
            requireDimensionCompatibility(
                    values.length,
                    centers.length,
                    "Tabular row"
            );

            transformBoxedTabularRow(
                    values,
                    centers,
                    scales
            );

            return;
        }

        requireGlobalStatistics(
                centers,
                scales
        );

        transformBoxedDimension(
                values,
                centers[0],
                scales[0]
        );
    }

    /**
     * Transforms one primitive dimension-major multivariate instance.
     */
    private static void transformPrimitiveMultivariateInstance(
            double[][] values,
            StandardizationScope scope,
            double[] centers,
            double[] scales
    ) {
        requirePositiveDimensionCount(
                values.length
        );

        if (scope == StandardizationScope.PER_DIMENSION) {
            requireDimensionCompatibility(
                    values.length,
                    centers.length,
                    "Multivariate series"
            );

            for (int dimension = 0;
                 dimension < values.length;
                 dimension++) {

                double[] dimensionValues =
                        Objects.requireNonNull(
                                values[dimension],
                                "Numeric series contains a null dimension at "
                                        + dimension
                                        + "."
                        );

                transformPrimitiveDimension(
                        dimensionValues,
                        centers[dimension],
                        scales[dimension]
                );
            }

            return;
        }

        requireGlobalStatistics(
                centers,
                scales
        );

        double center =
                centers[0];

        double scale =
                scales[0];

        for (int dimension = 0;
             dimension < values.length;
             dimension++) {

            double[] dimensionValues =
                    Objects.requireNonNull(
                            values[dimension],
                            "Numeric series contains a null dimension at "
                                    + dimension
                                    + "."
                    );

            transformPrimitiveDimension(
                    dimensionValues,
                    center,
                    scale
            );
        }
    }

    /**
     * Transforms one boxed dimension-major multivariate instance.
     */
    private static void transformBoxedMultivariateInstance(
            Double[][] values,
            StandardizationScope scope,
            double[] centers,
            double[] scales
    ) {
        requirePositiveDimensionCount(
                values.length
        );

        if (scope == StandardizationScope.PER_DIMENSION) {
            requireDimensionCompatibility(
                    values.length,
                    centers.length,
                    "Multivariate series"
            );

            for (int dimension = 0;
                 dimension < values.length;
                 dimension++) {

                Double[] dimensionValues =
                        Objects.requireNonNull(
                                values[dimension],
                                "Numeric series contains a null dimension at "
                                        + dimension
                                        + "."
                        );

                transformBoxedDimension(
                        dimensionValues,
                        centers[dimension],
                        scales[dimension]
                );
            }

            return;
        }

        requireGlobalStatistics(
                centers,
                scales
        );

        double center =
                centers[0];

        double scale =
                scales[0];

        for (int dimension = 0;
             dimension < values.length;
             dimension++) {

            Double[] dimensionValues =
                    Objects.requireNonNull(
                            values[dimension],
                            "Numeric series contains a null dimension at "
                                    + dimension
                                    + "."
                    );

            transformBoxedDimension(
                    dimensionValues,
                    center,
                    scale
            );
        }
    }

    /**
     * Applies one center and scale per primitive tabular feature.
     */
    private static void transformPrimitiveTabularRow(
            double[] values,
            double[] centers,
            double[] scales
    ) {
        for (int feature = 0;
             feature < values.length;
             feature++) {

            double value =
                    values[feature];

            if (!Double.isNaN(
                    value
            )) {
                values[feature] =
                        (value - centers[feature])
                                / scales[feature];
            }
        }
    }

    /**
     * Applies one center and scale per boxed tabular feature.
     */
    private static void transformBoxedTabularRow(
            Double[] values,
            double[] centers,
            double[] scales
    ) {
        for (int feature = 0;
             feature < values.length;
             feature++) {

            Double value =
                    values[feature];

            if (value != null
                    && !Double.isNaN(
                    value
            )) {
                values[feature] =
                        (value - centers[feature])
                                / scales[feature];
            }
        }
    }

    /**
     * Applies one prepared center and scale to every accepted value in one
     * primitive series dimension.
     */
    private static void transformPrimitiveDimension(
            double[] values,
            double center,
            double scale
    ) {
        for (int index = 0;
             index < values.length;
             index++) {

            double value =
                    values[index];

            if (!Double.isNaN(
                    value
            )) {
                values[index] =
                        (value - center)
                                / scale;
            }
        }
    }

    /**
     * Applies one prepared center and scale to every accepted value in one
     * boxed series dimension.
     */
    private static void transformBoxedDimension(
            Double[] values,
            double center,
            double scale
    ) {
        for (int index = 0;
             index < values.length;
             index++) {

            Double value =
                    values[index];

            if (value != null
                    && !Double.isNaN(
                    value
            )) {
                values[index] =
                        (value - center)
                                / scale;
            }
        }
    }

    /**
     * Performs one-time validation of copied prepared-statistics arrays.
     */
    private static void validatePreparedStatistics(
            StandardizationScope scope,
            double[] centers,
            double[] scales
    ) {
        Objects.requireNonNull(
                scope,
                "Standardization scope cannot be null."
        );

        Objects.requireNonNull(
                centers,
                "Standardization centers cannot be null."
        );

        Objects.requireNonNull(
                scales,
                "Standardization scales cannot be null."
        );

        if (centers.length == 0) {
            throw new IllegalArgumentException(
                    "Standardization statistics must contain at least "
                            + "one statistic group."
            );
        }

        if (centers.length != scales.length) {
            throw new IllegalArgumentException(
                    "Standardization centers and scales must have "
                            + "identical lengths. Received centers="
                            + centers.length
                            + " and scales="
                            + scales.length
                            + "."
            );
        }

        if (scope != StandardizationScope.GLOBAL
                && scope != StandardizationScope.PER_DIMENSION) {

            throw new UnsupportedOperationException(
                    "Prepared reusable statistics do not support scope "
                            + scope
                            + "."
            );
        }

        if (scope == StandardizationScope.GLOBAL
                && centers.length != 1) {

            throw new IllegalArgumentException(
                    "GLOBAL standardization requires exactly one "
                            + "statistic group, but received "
                            + centers.length
                            + "."
            );
        }
    }

    private static void requireGlobalStatistics(
            double[] centers,
            double[] scales
    ) {
        if (centers.length != 1
                || scales.length != 1) {

            throw new IllegalArgumentException(
                    "GLOBAL standardization requires exactly one "
                            + "center and one scale."
            );
        }
    }

    private static void requirePositiveDimensionCount(
            int dimensionCount
    ) {
        if (dimensionCount < 1) {
            throw new IllegalArgumentException(
                    "Numeric data must contain at least one "
                            + "realized dimension."
            );
        }
    }

    /**
     * Validates either a tabular feature count or a multivariate dimension
     * count against the prepared PER_DIMENSION statistics.
     */
    private static void requireDimensionCompatibility(
            int actualDimensionCount,
            int expectedDimensionCount,
            String representationName
    ) {
        requirePositiveDimensionCount(
                actualDimensionCount
        );

        if (actualDimensionCount != expectedDimensionCount) {
            throw new IllegalArgumentException(
                    representationName
                            + " contains "
                            + actualDimensionCount
                            + " realized dimensions or features, but "
                            + "PER_DIMENSION statistics contain "
                            + expectedDimensionCount
                            + " groups."
            );
        }
    }
}