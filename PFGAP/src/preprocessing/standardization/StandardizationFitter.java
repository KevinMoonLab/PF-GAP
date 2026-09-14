package preprocessing.standardization;

import datasets.ListObjectDataset;
import datasets.readers.lazy.LazySeriesRef;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * Fits reusable standardization statistics from an eager numeric dataset.
 *
 * <p>The fitter performs one value traversal. It trusts the configured
 * numeric and missing-value contracts and retains only cheap structural
 * checks. Primitive and boxed NaN values, plus boxed null values, are skipped.
 * {@link OnlineMoments} owns finite-value and accumulator-overflow checks.</p>
 *
 * <p>Supported instance representations are {@code double[]},
 * {@code Double[]}, {@code double[][]}, and {@code Double[][]}.
 * Multivariate arrays are dimension-major.</p>
 *
 * <p>When {@link StandardizationScope#PER_DIMENSION} is selected for
 * {@code double[]} or {@code Double[]} instances, each array position is
 * interpreted as one tabular feature and is fitted across dataset instances.
 * All rows must consequently contain the same number of features.</p>
 *
 * <p>For {@link StandardizationScope#GLOBAL}, {@code double[]} and
 * {@code Double[]} retain their univariate-series interpretation, and all
 * values from every instance contribute to one reusable statistic group.</p>
 *
 * <p>For {@code double[][]} and {@code Double[][]}, the outer array remains
 * the dimension-major axis. GLOBAL combines values from all dimensions into
 * one group, while PER_DIMENSION fits one group per outer-array dimension.</p>
 */
public final class StandardizationFitter {

    public static final double CONSTANT_SCALE =
            1.0;

    private StandardizationFitter() {
        // Utility class.
    }

    public static StandardizationStats fit(
            ListObjectDataset dataset
    ) {
        return fit(
                dataset,
                StandardizationMethod.Z_SCORE,
                StandardizationScope.PER_DIMENSION,
                VarianceConvention.POPULATION,
                Collections.emptyList()
        );
    }

    public static StandardizationStats fit(
            ListObjectDataset dataset,
            StandardizationScope scope
    ) {
        return fit(
                dataset,
                StandardizationMethod.Z_SCORE,
                scope,
                VarianceConvention.POPULATION,
                Collections.emptyList()
        );
    }

    public static StandardizationStats fit(
            ListObjectDataset dataset,
            StandardizationScope scope,
            VarianceConvention varianceConvention
    ) {
        return fit(
                dataset,
                StandardizationMethod.Z_SCORE,
                scope,
                varianceConvention,
                Collections.emptyList()
        );
    }

    public static StandardizationStats fit(
            ListObjectDataset dataset,
            StandardizationScope scope,
            VarianceConvention varianceConvention,
            List<String> featureNames
    ) {
        return fit(
                dataset,
                StandardizationMethod.Z_SCORE,
                scope,
                varianceConvention,
                featureNames
        );
    }

    /**
     * Fits reusable statistics in one pass over accepted numeric values.
     *
     * @param dataset eager numeric training dataset
     * @param method standardization method
     * @param scope reusable-statistics scope
     * @param varianceConvention variance denominator convention
     * @param featureNames optional ordered realized-dimension names
     * @return fitted standardization statistics
     */
    public static StandardizationStats fit(
            ListObjectDataset dataset,
            StandardizationMethod method,
            StandardizationScope scope,
            VarianceConvention varianceConvention,
            List<String> featureNames
    ) {
        Objects.requireNonNull(
                dataset,
                "Dataset cannot be null."
        );

        Objects.requireNonNull(
                method,
                "StandardizationMethod cannot be null."
        );

        Objects.requireNonNull(
                scope,
                "StandardizationScope cannot be null."
        );

        Objects.requireNonNull(
                varianceConvention,
                "VarianceConvention cannot be null."
        );

        if (method != StandardizationMethod.Z_SCORE) {
            throw new UnsupportedOperationException(
                    "Standardization method "
                            + method
                            + " is not implemented."
            );
        }

        if (scope != StandardizationScope.GLOBAL
                && scope != StandardizationScope.PER_DIMENSION) {

            throw new UnsupportedOperationException(
                    "Standardization scope "
                            + scope
                            + " is not implemented."
            );
        }

        List<Object> data =
                Objects.requireNonNull(
                        dataset.getData(),
                        "Dataset data cannot be null."
                );

        if (data.isEmpty()) {
            throw new IllegalArgumentException(
                    "Cannot fit standardization statistics "
                            + "from an empty dataset."
            );
        }

        Object firstInstance =
                firstRealizedInstance(
                        data
                );

        int dimensionCount =
                dimensionCountOf(
                        firstInstance,
                        scope
                );

        List<String> normalizedFeatureNames =
                validateAndCopyFeatureNames(
                        featureNames,
                        dimensionCount,
                        scope
                );

        OnlineMoments[] moments =
                createAccumulators(
                        scope,
                        dimensionCount
                );

        for (Object instance : data) {
            accumulateInstance(
                    instance,
                    dimensionCount,
                    scope,
                    moments
            );
        }

        return buildStatistics(
                method,
                scope,
                varianceConvention,
                normalizedFeatureNames,
                moments
        );
    }

    /**
     * Returns the first eager realized instance in the supplied data.
     */
    private static Object firstRealizedInstance(
            List<Object> data
    ) {
        for (Object instance : data) {
            if (instance == null) {
                continue;
            }

            if (instance instanceof LazySeriesRef) {
                throw new UnsupportedOperationException(
                        "Standardization fitting requires eager realized data."
                );
            }

            return instance;
        }

        throw new IllegalArgumentException(
                "Training dataset contains no realized instances."
        );
    }

    /**
     * Returns the number of realized dimensions represented by one instance
     * under the requested reusable-statistics scope.
     *
     * <p>For PER_DIMENSION, a one-dimensional numeric array is interpreted as
     * one tabular row, and each array position is a realized feature.</p>
     *
     * <p>For GLOBAL, a one-dimensional numeric array remains one univariate
     * representation because all values contribute to the same statistic
     * group.</p>
     */
    private static int dimensionCountOf(
            Object instance,
            StandardizationScope scope
    ) {
        if (instance instanceof double[] values) {
            if (scope == StandardizationScope.PER_DIMENSION) {
                requirePositiveDimensionCount(
                        values.length
                );

                return values.length;
            }

            return 1;
        }

        if (instance instanceof Double[] values) {
            if (scope == StandardizationScope.PER_DIMENSION) {
                requirePositiveDimensionCount(
                        values.length
                );

                return values.length;
            }

            return 1;
        }

        if (instance instanceof double[][] matrix) {
            requirePositiveDimensionCount(
                    matrix.length
            );

            return matrix.length;
        }

        if (instance instanceof Double[][] matrix) {
            requirePositiveDimensionCount(
                    matrix.length
            );

            return matrix.length;
        }

        throw unsupportedSeriesType(
                instance
        );
    }

    /**
     * Creates one accumulator for GLOBAL standardization or one accumulator
     * per realized dimension for PER_DIMENSION standardization.
     */
    private static OnlineMoments[] createAccumulators(
            StandardizationScope scope,
            int dimensionCount
    ) {
        int groupCount =
                scope == StandardizationScope.GLOBAL
                        ? 1
                        : dimensionCount;

        OnlineMoments[] moments =
                new OnlineMoments[groupCount];

        for (int group = 0;
             group < groupCount;
             group++) {

            moments[group] =
                    new OnlineMoments();
        }

        return moments;
    }

    /**
     * Accumulates one supported eager instance.
     */
    private static void accumulateInstance(
            Object instance,
            int expectedDimensionCount,
            StandardizationScope scope,
            OnlineMoments[] moments
    ) {
        Objects.requireNonNull(
                instance,
                "Training instance cannot be null."
        );

        if (instance instanceof LazySeriesRef) {
            throw new UnsupportedOperationException(
                    "Standardization fitting requires eager realized data."
            );
        }

        if (instance instanceof double[] values) {
            accumulatePrimitiveOneDimensionalInstance(
                    values,
                    expectedDimensionCount,
                    scope,
                    moments
            );

            return;
        }

        if (instance instanceof Double[] values) {
            accumulateBoxedOneDimensionalInstance(
                    values,
                    expectedDimensionCount,
                    scope,
                    moments
            );

            return;
        }

        if (instance instanceof double[][] matrix) {
            accumulatePrimitiveMultivariateInstance(
                    matrix,
                    expectedDimensionCount,
                    scope,
                    moments
            );

            return;
        }

        if (instance instanceof Double[][] matrix) {
            accumulateBoxedMultivariateInstance(
                    matrix,
                    expectedDimensionCount,
                    scope,
                    moments
            );

            return;
        }

        throw unsupportedSeriesType(
                instance
        );
    }

    /**
     * Accumulates one primitive one-dimensional instance.
     *
     * <p>PER_DIMENSION interprets the instance as a tabular row.
     * GLOBAL interprets it as a univariate series contributing to one
     * statistic group.</p>
     */
    private static void accumulatePrimitiveOneDimensionalInstance(
            double[] values,
            int expectedDimensionCount,
            StandardizationScope scope,
            OnlineMoments[] moments
    ) {
        if (scope == StandardizationScope.PER_DIMENSION) {
            requireExpectedDimensionCount(
                    values.length,
                    expectedDimensionCount
            );

            accumulatePrimitiveTabularRow(
                    values,
                    moments
            );

            return;
        }

        requireUnivariateCompatibility(
                expectedDimensionCount
        );

        accumulatePrimitiveDimension(
                values,
                moments[0]
        );
    }

    /**
     * Accumulates one boxed one-dimensional instance.
     *
     * <p>PER_DIMENSION interprets the instance as a tabular row.
     * GLOBAL interprets it as a univariate series contributing to one
     * statistic group.</p>
     */
    private static void accumulateBoxedOneDimensionalInstance(
            Double[] values,
            int expectedDimensionCount,
            StandardizationScope scope,
            OnlineMoments[] moments
    ) {
        if (scope == StandardizationScope.PER_DIMENSION) {
            requireExpectedDimensionCount(
                    values.length,
                    expectedDimensionCount
            );

            accumulateBoxedTabularRow(
                    values,
                    moments
            );

            return;
        }

        requireUnivariateCompatibility(
                expectedDimensionCount
        );

        accumulateBoxedDimension(
                values,
                moments[0]
        );
    }

    /**
     * Accumulates one primitive dimension-major multivariate instance.
     */
    private static void accumulatePrimitiveMultivariateInstance(
            double[][] matrix,
            int expectedDimensionCount,
            StandardizationScope scope,
            OnlineMoments[] moments
    ) {
        requireExpectedDimensionCount(
                matrix.length,
                expectedDimensionCount
        );

        for (int dimension = 0;
             dimension < matrix.length;
             dimension++) {

            double[] values =
                    Objects.requireNonNull(
                            matrix[dimension],
                            "Training series contains a null dimension."
                    );

            accumulatePrimitiveDimension(
                    values,
                    accumulator(
                            scope,
                            moments,
                            dimension
                    )
            );
        }
    }

    /**
     * Accumulates one boxed dimension-major multivariate instance.
     */
    private static void accumulateBoxedMultivariateInstance(
            Double[][] matrix,
            int expectedDimensionCount,
            StandardizationScope scope,
            OnlineMoments[] moments
    ) {
        requireExpectedDimensionCount(
                matrix.length,
                expectedDimensionCount
        );

        for (int dimension = 0;
             dimension < matrix.length;
             dimension++) {

            Double[] values =
                    Objects.requireNonNull(
                            matrix[dimension],
                            "Training series contains a null dimension."
                    );

            accumulateBoxedDimension(
                    values,
                    accumulator(
                            scope,
                            moments,
                            dimension
                    )
            );
        }
    }

    /**
     * Returns the reusable accumulator for one multivariate dimension.
     */
    private static OnlineMoments accumulator(
            StandardizationScope scope,
            OnlineMoments[] moments,
            int dimension
    ) {
        return scope == StandardizationScope.GLOBAL
                ? moments[0]
                : moments[dimension];
    }

    /**
     * Accumulates one primitive tabular row into one statistic group per
     * feature position.
     */
    private static void accumulatePrimitiveTabularRow(
            double[] values,
            OnlineMoments[] moments
    ) {
        for (int feature = 0;
             feature < values.length;
             feature++) {

            double value =
                    values[feature];

            if (!Double.isNaN(
                    value
            )) {
                moments[feature].add(
                        value
                );
            }
        }
    }

    /**
     * Accumulates one boxed tabular row into one statistic group per feature
     * position.
     */
    private static void accumulateBoxedTabularRow(
            Double[] values,
            OnlineMoments[] moments
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
                moments[feature].add(
                        value
                );
            }
        }
    }

    /**
     * Accumulates every accepted value from one primitive series dimension
     * into one statistic group.
     */
    private static void accumulatePrimitiveDimension(
            double[] values,
            OnlineMoments moments
    ) {
        for (double value : values) {
            if (!Double.isNaN(
                    value
            )) {
                moments.add(
                        value
                );
            }
        }
    }

    /**
     * Accumulates every accepted value from one boxed series dimension into
     * one statistic group.
     */
    private static void accumulateBoxedDimension(
            Double[] values,
            OnlineMoments moments
    ) {
        for (Double value : values) {
            if (value != null
                    && !Double.isNaN(
                    value
            )) {
                moments.add(
                        value
                );
            }
        }
    }

    /**
     * Converts completed online accumulators into immutable reusable
     * statistics.
     */
    private static StandardizationStats buildStatistics(
            StandardizationMethod method,
            StandardizationScope scope,
            VarianceConvention varianceConvention,
            List<String> featureNames,
            OnlineMoments[] moments
    ) {
        long[] counts =
                new long[moments.length];

        double[] centers =
                new double[moments.length];

        double[] scales =
                new double[moments.length];

        for (int group = 0;
             group < moments.length;
             group++) {

            OnlineMoments accumulator =
                    moments[group];

            if (!accumulator.hasObservations()) {
                throw new IllegalArgumentException(
                        "Standardization statistic group "
                                + group
                                + " contains no observations."
                );
            }

            counts[group] =
                    accumulator.getCount();

            centers[group] =
                    accumulator.getMean();

            scales[group] =
                    fittedScale(
                            accumulator,
                            varianceConvention
                    );
        }

        return new StandardizationStats(
                method,
                scope,
                varianceConvention,
                featureNames,
                counts,
                centers,
                scales
        );
    }

    /**
     * Returns the fitted standard deviation or the configured constant-group
     * scale when variance cannot be calculated or is zero.
     */
    private static double fittedScale(
            OnlineMoments moments,
            VarianceConvention varianceConvention
    ) {
        if (!moments.canCalculateVariance(
                varianceConvention
        )) {
            return CONSTANT_SCALE;
        }

        double standardDeviation =
                moments.getStandardDeviation(
                        varianceConvention
                );

        return standardDeviation == 0.0
                ? CONSTANT_SCALE
                : standardDeviation;
    }

    /**
     * Validates and defensively copies ordered feature names.
     *
     * <p>For PER_DIMENSION tabular data, the expected count is the length of
     * each double[] or Double[] row. For dimension-major multivariate data,
     * it is the number of outer-array dimensions.</p>
     */
    private static List<String> validateAndCopyFeatureNames(
            List<String> featureNames,
            int dimensionCount,
            StandardizationScope scope
    ) {
        if (featureNames == null
                || featureNames.isEmpty()) {

            return Collections.emptyList();
        }

        if (featureNames.size() != dimensionCount) {
            throw new IllegalArgumentException(
                    "Training data contains "
                            + dimensionCount
                            + " realized dimension(s), but "
                            + featureNames.size()
                            + " feature name(s) were supplied for "
                            + scope
                            + " standardization."
            );
        }

        List<String> copy =
                new ArrayList<>(
                        featureNames.size()
                );

        for (String featureName : featureNames) {
            if (featureName == null
                    || featureName.isBlank()) {

                throw new IllegalArgumentException(
                        "Standardization feature names cannot be "
                                + "null or blank."
                );
            }

            copy.add(
                    featureName.trim()
            );
        }

        return Collections.unmodifiableList(
                copy
        );
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

    private static void requireUnivariateCompatibility(
            int expectedDimensionCount
    ) {
        requireExpectedDimensionCount(
                1,
                expectedDimensionCount
        );
    }

    /**
     * Validates the realized dimension count.
     *
     * <p>For PER_DIMENSION one-dimensional data, this compares the current
     * tabular row length with the feature count established from the first
     * realized row.</p>
     */
    private static void requireExpectedDimensionCount(
            int actual,
            int expected
    ) {
        if (actual != expected) {
            throw new IllegalArgumentException(
                    "Training instance contains "
                            + actual
                            + " realized dimensions or features; expected "
                            + expected
                            + ". PER_DIMENSION tabular rows must have "
                            + "consistent lengths."
            );
        }
    }

    private static IllegalArgumentException unsupportedSeriesType(
            Object instance
    ) {
        return new IllegalArgumentException(
                "Unsupported standardization training type: "
                        + instance.getClass().getTypeName()
                        + ". Expected double[], Double[], double[][], "
                        + "or Double[][]."
        );
    }
}