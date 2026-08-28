package preprocessing.standardization;

import datasets.ListObjectDataset;
import datasets.readers.lazy.LazySeriesRef;

import java.util.List;
import java.util.Objects;

/**
 * Applies fitted standardization statistics to eager numeric data.
 *
 * Phase 1 supports these realized instance representations:
 *
 *     double[]
 *     Double[]
 *     double[][]
 *     Double[][]
 *
 * Arrays are transformed in place. No duplicate dataset or complete
 * standardized copy is created.
 *
 * Univariate arrays are treated as one-dimensional series.
 *
 * Multivariate arrays are expected to use dimension-major orientation:
 *
 *     data[dimension][time]
 *
 * Missing-value behavior:
 *
 *     Double null:
 *         preserved as null
 *
 *     NaN:
 *         preserved as NaN
 *
 *     positive or negative infinity:
 *         rejected
 *
 * LazySeriesRef objects are rejected by the dataset-level methods. Lazy
 * standardization will later be applied by the series reader immediately
 * after materialization.
 *
 * This class is stateless and thread-safe, provided separate threads do not
 * modify the same arrays concurrently.
 */
public final class Standardizer {

    private Standardizer() {
    }

    /**
     * Standardizes every instance in a dataset in place.
     *
     * If feature names are supplied, they are validated against the fitted
     * statistics before any dataset values are modified.
     *
     * @param dataset eager numeric dataset
     * @param stats fitted standardization statistics
     * @param featureNames ordered feature names, or null when unavailable
     * @return the same dataset instance after in-place transformation
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

        if (dataset.getData() == null) {
            throw new IllegalArgumentException(
                    "Dataset data cannot be null."
            );
        }

        stats.validateFeatureCompatibility(
                featureNames
        );

        /*
         * Validate every instance before modifying any values. This avoids
         * partially standardizing a dataset and then discovering an
         * incompatible instance near the end.
         */
        validateDataset(
                dataset,
                stats
        );

        for (int instanceIndex = 0;
             instanceIndex < dataset.getData().size();
             instanceIndex++) {

            Object instance =
                    dataset.getData().get(
                            instanceIndex
                    );

            transformInstanceInPlace(
                    instance,
                    stats
            );
        }

        return dataset;
    }

    /**
     * Standardizes every instance in a dataset in place without explicit
     * feature-name validation.
     *
     * Statistics containing feature names require the caller to use the
     * overload that supplies feature names. This prevents named statistics
     * from being applied to unverified feature ordering.
     *
     * @param dataset eager numeric dataset
     * @param stats fitted standardization statistics
     * @return the same dataset instance after in-place transformation
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
                            + "names. Supply the dataset's feature names to "
                            + "transformInPlace(dataset, stats, featureNames) "
                            + "so their order can be validated."
            );
        }

        return transformInPlace(
                dataset,
                stats,
                null
        );
    }

    /**
     * Standardizes one realized numeric series in place.
     *
     * This method is useful for both:
     *
     *     eager dataset transformation
     *
     * and, later:
     *
     *     standardization immediately after lazy materialization
     *
     * @param series realized numeric series
     * @param stats fitted standardization statistics
     * @return the same series object after in-place transformation
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

        validateInstance(
                series,
                stats,
                -1
        );

        if (series instanceof double[] values) {
            transformPrimitiveUnivariateInPlace(
                    values,
                    stats
            );

            return values;
        }

        if (series instanceof Double[] values) {
            transformBoxedUnivariateInPlace(
                    values,
                    stats
            );

            return values;
        }

        if (series instanceof double[][] values) {
            transformPrimitiveMultivariateInPlace(
                    values,
                    stats
            );

            return values;
        }

        if (series instanceof Double[][] values) {
            transformBoxedMultivariateInPlace(
                    values,
                    stats
            );

            return values;
        }

        /*
         * validateInstance(...) already rejects unsupported types, so this
         * branch is unreachable unless the validation and dispatch logic
         * become inconsistent.
         */
        throw unsupportedSeriesType(series);
    }

    /**
     * Validates a complete dataset without modifying it.
     */
    private static void validateDataset(
            ListObjectDataset dataset,
            StandardizationStats stats
    ) {
        for (int instanceIndex = 0;
             instanceIndex < dataset.getData().size();
             instanceIndex++) {

            Object instance =
                    dataset.getData().get(
                            instanceIndex
                    );

            validateInstance(
                    instance,
                    stats,
                    instanceIndex
            );
        }
    }

    /**
     * Validates the series representation, dimension count, and individual
     * values without modifying the series.
     */
    private static void validateInstance(
            Object series,
            StandardizationStats stats,
            int instanceIndex
    ) {
        if (series == null) {
            throw new IllegalArgumentException(
                    instancePrefix(instanceIndex)
                            + "contains a null series."
            );
        }

        if (series instanceof LazySeriesRef) {
            throw new UnsupportedOperationException(
                    instancePrefix(instanceIndex)
                            + "contains a LazySeriesRef. Eager dataset "
                            + "standardization cannot transform lazy "
                            + "references directly. Apply statistics after "
                            + "the series is materialized."
            );
        }

        if (series instanceof double[] values) {
            validateUnivariateLength(
                    values.length,
                    instanceIndex
            );

            validateDimensionCompatibility(
                    1,
                    stats,
                    instanceIndex
            );

            validatePrimitiveValues(
                    values,
                    instanceIndex,
                    0
            );

            return;
        }

        if (series instanceof Double[] values) {
            validateUnivariateLength(
                    values.length,
                    instanceIndex
            );

            validateDimensionCompatibility(
                    1,
                    stats,
                    instanceIndex
            );

            validateBoxedValues(
                    values,
                    instanceIndex,
                    0
            );

            return;
        }

        if (series instanceof double[][] values) {
            validateDimensionCompatibility(
                    values.length,
                    stats,
                    instanceIndex
            );

            validatePrimitiveMultivariate(
                    values,
                    instanceIndex
            );

            return;
        }

        if (series instanceof Double[][] values) {
            validateDimensionCompatibility(
                    values.length,
                    stats,
                    instanceIndex
            );

            validateBoxedMultivariate(
                    values,
                    instanceIndex
            );

            return;
        }

        throw unsupportedSeriesType(
                series,
                instanceIndex
        );
    }

    private static void transformPrimitiveUnivariateInPlace(
            double[] series,
            StandardizationStats stats
    ) {
        int groupIndex =
                stats.getGroupIndexForDimension(0);

        double center =
                stats.getCenter(groupIndex);

        double scale =
                stats.getScale(groupIndex);

        for (int timeIndex = 0;
             timeIndex < series.length;
             timeIndex++) {

            double value =
                    series[timeIndex];

            if (Double.isNaN(value)) {
                continue;
            }

            series[timeIndex] =
                    standardizeFiniteValue(
                            value,
                            center,
                            scale
                    );
        }
    }

    private static void transformBoxedUnivariateInPlace(
            Double[] series,
            StandardizationStats stats
    ) {
        int groupIndex =
                stats.getGroupIndexForDimension(0);

        double center =
                stats.getCenter(groupIndex);

        double scale =
                stats.getScale(groupIndex);

        for (int timeIndex = 0;
             timeIndex < series.length;
             timeIndex++) {

            Double value =
                    series[timeIndex];

            if (value == null
                    || Double.isNaN(value)) {

                continue;
            }

            series[timeIndex] =
                    standardizeFiniteValue(
                            value,
                            center,
                            scale
                    );
        }
    }

    private static void transformPrimitiveMultivariateInPlace(
            double[][] series,
            StandardizationStats stats
    ) {
        for (int dimensionIndex = 0;
             dimensionIndex < series.length;
             dimensionIndex++) {

            int groupIndex =
                    stats.getGroupIndexForDimension(
                            dimensionIndex
                    );

            double center =
                    stats.getCenter(groupIndex);

            double scale =
                    stats.getScale(groupIndex);

            double[] dimension =
                    series[dimensionIndex];

            for (int timeIndex = 0;
                 timeIndex < dimension.length;
                 timeIndex++) {

                double value =
                        dimension[timeIndex];

                if (Double.isNaN(value)) {
                    continue;
                }

                dimension[timeIndex] =
                        standardizeFiniteValue(
                                value,
                                center,
                                scale
                        );
            }
        }
    }

    private static void transformBoxedMultivariateInPlace(
            Double[][] series,
            StandardizationStats stats
    ) {
        for (int dimensionIndex = 0;
             dimensionIndex < series.length;
             dimensionIndex++) {

            int groupIndex =
                    stats.getGroupIndexForDimension(
                            dimensionIndex
                    );

            double center =
                    stats.getCenter(groupIndex);

            double scale =
                    stats.getScale(groupIndex);

            Double[] dimension =
                    series[dimensionIndex];

            for (int timeIndex = 0;
                 timeIndex < dimension.length;
                 timeIndex++) {

                Double value =
                        dimension[timeIndex];

                if (value == null
                        || Double.isNaN(value)) {

                    continue;
                }

                dimension[timeIndex] =
                        standardizeFiniteValue(
                                value,
                                center,
                                scale
                        );
            }
        }
    }

    private static double standardizeFiniteValue(
            double value,
            double center,
            double scale
    ) {
        /*
         * All inputs should already have been validated. Retain these checks
         * so direct future use cannot silently introduce infinities.
         */
        if (!Double.isFinite(value)) {
            throw new IllegalArgumentException(
                    "Cannot standardize non-finite value: "
                            + value
            );
        }

        if (!Double.isFinite(center)) {
            throw new IllegalArgumentException(
                    "Standardization center must be finite: "
                            + center
            );
        }

        if (!Double.isFinite(scale)
                || scale <= 0.0) {

            throw new IllegalArgumentException(
                    "Standardization scale must be finite and positive: "
                            + scale
            );
        }

        double standardized =
                (value - center)
                        / scale;

        if (!Double.isFinite(standardized)) {
            throw new ArithmeticException(
                    "Standardization produced a non-finite value from "
                            + "value="
                            + value
                            + ", center="
                            + center
                            + ", scale="
                            + scale
                            + "."
            );
        }

        return standardized;
    }

    private static void validateDimensionCompatibility(
            int dimensionCount,
            StandardizationStats stats,
            int instanceIndex
    ) {
        if (dimensionCount <= 0) {
            throw new IllegalArgumentException(
                    instancePrefix(instanceIndex)
                            + "contains no dimensions."
            );
        }

        if (stats.getScope()
                == StandardizationScope.PER_DIMENSION
                && dimensionCount
                != stats.getStatisticGroupCount()) {

            throw new IllegalArgumentException(
                    instancePrefix(instanceIndex)
                            + "contains "
                            + dimensionCount
                            + " dimensions, but the fitted "
                            + "PER_DIMENSION statistics contain "
                            + stats.getStatisticGroupCount()
                            + " groups."
            );
        }

        /*
         * GLOBAL statistics contain one group and can apply to any positive
         * dimension count.
         */
    }

    private static void validateUnivariateLength(
            int length,
            int instanceIndex
    ) {
        if (length == 0) {
            throw new IllegalArgumentException(
                    instancePrefix(instanceIndex)
                            + "is an empty univariate series."
            );
        }
    }

    private static void validatePrimitiveMultivariate(
            double[][] series,
            int instanceIndex
    ) {
        if (series.length == 0) {
            throw new IllegalArgumentException(
                    instancePrefix(instanceIndex)
                            + "contains no dimensions."
            );
        }

        for (int dimensionIndex = 0;
             dimensionIndex < series.length;
             dimensionIndex++) {

            double[] dimension =
                    series[dimensionIndex];

            if (dimension == null) {
                throw new IllegalArgumentException(
                        instancePrefix(instanceIndex)
                                + "contains a null primitive dimension at "
                                + "index "
                                + dimensionIndex
                                + "."
                );
            }

            if (dimension.length == 0) {
                throw new IllegalArgumentException(
                        instancePrefix(instanceIndex)
                                + "contains an empty dimension at index "
                                + dimensionIndex
                                + "."
                );
            }

            validatePrimitiveValues(
                    dimension,
                    instanceIndex,
                    dimensionIndex
            );
        }
    }

    private static void validateBoxedMultivariate(
            Double[][] series,
            int instanceIndex
    ) {
        if (series.length == 0) {
            throw new IllegalArgumentException(
                    instancePrefix(instanceIndex)
                            + "contains no dimensions."
            );
        }

        for (int dimensionIndex = 0;
             dimensionIndex < series.length;
             dimensionIndex++) {

            Double[] dimension =
                    series[dimensionIndex];

            if (dimension == null) {
                throw new IllegalArgumentException(
                        instancePrefix(instanceIndex)
                                + "contains a null boxed dimension at index "
                                + dimensionIndex
                                + "."
                );
            }

            if (dimension.length == 0) {
                throw new IllegalArgumentException(
                        instancePrefix(instanceIndex)
                                + "contains an empty dimension at index "
                                + dimensionIndex
                                + "."
                );
            }

            validateBoxedValues(
                    dimension,
                    instanceIndex,
                    dimensionIndex
            );
        }
    }

    private static void validatePrimitiveValues(
            double[] values,
            int instanceIndex,
            int dimensionIndex
    ) {
        for (int timeIndex = 0;
             timeIndex < values.length;
             timeIndex++) {

            double value =
                    values[timeIndex];

            /*
             * NaN is treated as a preserved missing value.
             */
            if (Double.isNaN(value)) {
                continue;
            }

            if (Double.isInfinite(value)) {
                throw infiniteValueException(
                        value,
                        instanceIndex,
                        dimensionIndex,
                        timeIndex
                );
            }
        }
    }

    private static void validateBoxedValues(
            Double[] values,
            int instanceIndex,
            int dimensionIndex
    ) {
        for (int timeIndex = 0;
             timeIndex < values.length;
             timeIndex++) {

            Double value =
                    values[timeIndex];

            /*
             * Null and NaN are treated as preserved missing values.
             */
            if (value == null
                    || Double.isNaN(value)) {

                continue;
            }

            if (Double.isInfinite(value)) {
                throw infiniteValueException(
                        value,
                        instanceIndex,
                        dimensionIndex,
                        timeIndex
                );
            }
        }
    }

    private static IllegalArgumentException infiniteValueException(
            double value,
            int instanceIndex,
            int dimensionIndex,
            int timeIndex
    ) {
        return new IllegalArgumentException(
                instancePrefix(instanceIndex)
                        + "contains infinite value "
                        + value
                        + " at dimension "
                        + dimensionIndex
                        + ", time index "
                        + timeIndex
                        + "."
        );
    }

    private static IllegalArgumentException unsupportedSeriesType(
            Object series
    ) {
        return unsupportedSeriesType(
                series,
                -1
        );
    }

    private static IllegalArgumentException unsupportedSeriesType(
            Object series,
            int instanceIndex
    ) {
        return new IllegalArgumentException(
                instancePrefix(instanceIndex)
                        + "has unsupported type "
                        + series.getClass().getName()
                        + ". Phase 1 standardization supports double[], "
                        + "Double[], double[][], and Double[][]."
        );
    }

    private static String instancePrefix(
            int instanceIndex
    ) {
        if (instanceIndex < 0) {
            return "Series ";
        }

        return "Dataset instance "
                + instanceIndex
                + " ";
    }
}