package preprocessing.standardization;

import datasets.ListObjectDataset;
import datasets.readers.lazy.LazySeriesRef;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * Fits reusable affine standardization statistics from an eager numeric
 * training dataset.
 *
 * <p>This fitter is used only by scopes that reuse training-set statistics:
 * {@link StandardizationScope#GLOBAL} and
 * {@link StandardizationScope#PER_DIMENSION}. Per-series scopes calculate
 * local parameters during transformation and are represented by
 * {@link PerSeriesStandardizationState}.</p>
 *
 * <p>The fitter performs one traversal of accepted numeric values and supports
 * {@code double[]}, {@code float[]}, {@code double[][]}, and
 * {@code float[][]}. Multivariate arrays are dimension-major. Primitive NaN
 * values are treated as missing and skipped. Infinite values are rejected so
 * they cannot corrupt fitted parameters.</p>
 *
 * <p>Z-score and mean-centering use {@link OnlineMoments}; min-max scaling uses
 * {@link OnlineRange}. Statistics are accumulated in double precision even
 * when the source representation is float.</p>
 */
public final class StandardizationFitter {

    /**
     * Scale stored for constant groups and for groups whose requested sample
     * variance cannot be calculated from the available observation count.
     */
    public static final double CONSTANT_SCALE = 1.0;

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
     * Fits reusable statistics in one traversal of accepted numeric values.
     *
     * @param dataset eager numeric training dataset
     * @param method affine standardization method
     * @param scope reusable-statistics scope
     * @param varianceConvention variance convention used by z-score fitting
     * @param featureNames optional ordered realized-dimension names
     * @return immutable fitted statistics
     */
    public static StandardizationStats fit(
            ListObjectDataset dataset,
            StandardizationMethod method,
            StandardizationScope scope,
            VarianceConvention varianceConvention,
            List<String> featureNames
    ) {
        Objects.requireNonNull(dataset, "Dataset cannot be null.");
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

        method.requireImplemented();
        scope.requireImplemented();
        requireReusableMethod(method);
        requireReusableScope(scope);

        List<Object> data = Objects.requireNonNull(
                dataset.getData(),
                "Dataset data cannot be null."
        );
        if (data.isEmpty()) {
            throw new IllegalArgumentException(
                    "Cannot fit standardization statistics from an empty "
                            + "dataset."
            );
        }

        Object firstInstance = firstRealizedInstance(data);
        int dimensionCount = dimensionCountOf(firstInstance, scope);
        List<String> normalizedFeatureNames =
                validateAndCopyFeatureNames(
                        featureNames,
                        dimensionCount,
                        scope
                );

        int groupCount = scope.statisticGroupCount(dimensionCount);
        return switch (method) {
            case Z_SCORE, MEAN_CENTER ->
                    fitMomentStatistics(
                            data,
                            method,
                            scope,
                            varianceConvention,
                            normalizedFeatureNames,
                            dimensionCount,
                            groupCount
                    );
            case MIN_MAX ->
                    fitRangeStatistics(
                            data,
                            method,
                            scope,
                            varianceConvention,
                            normalizedFeatureNames,
                            dimensionCount,
                            groupCount
                    );
            case NONE, ROBUST ->
                    throw new UnsupportedOperationException(
                            "Standardization method "
                                    + method
                                    + " cannot fit reusable statistics."
                    );
        };
    }

    private static StandardizationStats fitMomentStatistics(
            List<Object> data,
            StandardizationMethod method,
            StandardizationScope scope,
            VarianceConvention varianceConvention,
            List<String> featureNames,
            int dimensionCount,
            int groupCount
    ) {
        OnlineMoments[] moments = createMomentAccumulators(groupCount);
        MomentSink sink = new MomentSink(moments);

        for (Object instance : data) {
            accumulateInstance(
                    instance,
                    dimensionCount,
                    scope,
                    sink
            );
        }

        long[] counts = new long[groupCount];
        double[] centers = new double[groupCount];
        double[] scales = new double[groupCount];

        for (int group = 0; group < groupCount; group++) {
            OnlineMoments accumulator = moments[group];
            requireObservations(accumulator.hasObservations(), group, method);
            counts[group] = accumulator.getCount();
            centers[group] = accumulator.getMean();
            scales[group] = switch (method) {
                case Z_SCORE -> fittedStandardDeviation(
                        accumulator,
                        varianceConvention,
                        group
                );
                case MEAN_CENTER -> CONSTANT_SCALE;
                case NONE, MIN_MAX, ROBUST ->
                        throw new IllegalStateException(
                                "Unexpected moment-based method: " + method
                        );
            };
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

    private static StandardizationStats fitRangeStatistics(
            List<Object> data,
            StandardizationMethod method,
            StandardizationScope scope,
            VarianceConvention varianceConvention,
            List<String> featureNames,
            int dimensionCount,
            int groupCount
    ) {
        OnlineRange[] ranges = createRangeAccumulators(groupCount);
        RangeSink sink = new RangeSink(ranges);

        for (Object instance : data) {
            accumulateInstance(
                    instance,
                    dimensionCount,
                    scope,
                    sink
            );
        }

        long[] counts = new long[groupCount];
        double[] centers = new double[groupCount];
        double[] scales = new double[groupCount];

        for (int group = 0; group < groupCount; group++) {
            OnlineRange accumulator = ranges[group];
            requireObservations(accumulator.hasObservations(), group, method);
            counts[group] = accumulator.getCount();
            centers[group] = accumulator.getMinimum();
            scales[group] = fittedRange(accumulator, group);
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

    private static OnlineMoments[] createMomentAccumulators(
            int groupCount
    ) {
        OnlineMoments[] moments = new OnlineMoments[groupCount];
        for (int group = 0; group < groupCount; group++) {
            moments[group] = new OnlineMoments();
        }
        return moments;
    }

    private static OnlineRange[] createRangeAccumulators(
            int groupCount
    ) {
        OnlineRange[] ranges = new OnlineRange[groupCount];
        for (int group = 0; group < groupCount; group++) {
            ranges[group] = new OnlineRange();
        }
        return ranges;
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
                throw lazyFittingUnsupported();
            }
            return instance;
        }
        throw new IllegalArgumentException(
                "Training dataset contains no realized instances."
        );
    }

    /**
     * Determines the realized dimension count used for compatibility checks
     * and feature-name validation.
     *
     * <p>For PER_DIMENSION, one-dimensional input is interpreted as a tabular
     * row whose positions are features. For GLOBAL, one-dimensional input is
     * one univariate representation. The outer length of two-dimensional input
     * is always the realized channel count.</p>
     */
    private static int dimensionCountOf(
            Object instance,
            StandardizationScope scope
    ) {
        if (instance instanceof double[] values) {
            if (scope == StandardizationScope.PER_DIMENSION) {
                requirePositiveDimensionCount(values.length);
                return values.length;
            }
            return 1;
        }
        if (instance instanceof float[] values) {
            if (scope == StandardizationScope.PER_DIMENSION) {
                requirePositiveDimensionCount(values.length);
                return values.length;
            }
            return 1;
        }
        if (instance instanceof double[][] matrix) {
            requirePositiveDimensionCount(matrix.length);
            return matrix.length;
        }
        if (instance instanceof float[][] matrix) {
            requirePositiveDimensionCount(matrix.length);
            return matrix.length;
        }
        throw unsupportedSeriesType(instance);
    }

    private static void accumulateInstance(
            Object instance,
            int expectedDimensionCount,
            StandardizationScope scope,
            ValueSink sink
    ) {
        Objects.requireNonNull(
                instance,
                "Training instance cannot be null."
        );
        if (instance instanceof LazySeriesRef) {
            throw lazyFittingUnsupported();
        }
        if (instance instanceof double[] values) {
            accumulateDoubleOneDimensional(
                    values,
                    expectedDimensionCount,
                    scope,
                    sink
            );
            return;
        }
        if (instance instanceof float[] values) {
            accumulateFloatOneDimensional(
                    values,
                    expectedDimensionCount,
                    scope,
                    sink
            );
            return;
        }
        if (instance instanceof double[][] matrix) {
            accumulateDoubleMultivariate(
                    matrix,
                    expectedDimensionCount,
                    scope,
                    sink
            );
            return;
        }
        if (instance instanceof float[][] matrix) {
            accumulateFloatMultivariate(
                    matrix,
                    expectedDimensionCount,
                    scope,
                    sink
            );
            return;
        }
        throw unsupportedSeriesType(instance);
    }

    private static void accumulateDoubleOneDimensional(
            double[] values,
            int expectedDimensionCount,
            StandardizationScope scope,
            ValueSink sink
    ) {
        if (scope == StandardizationScope.PER_DIMENSION) {
            requireExpectedDimensionCount(
                    values.length,
                    expectedDimensionCount
            );
            for (int feature = 0; feature < values.length; feature++) {
                addAccepted(values[feature], feature, sink);
            }
            return;
        }

        requireUnivariateCompatibility(expectedDimensionCount);
        accumulateDoubleDimension(values, 0, sink);
    }

    private static void accumulateFloatOneDimensional(
            float[] values,
            int expectedDimensionCount,
            StandardizationScope scope,
            ValueSink sink
    ) {
        if (scope == StandardizationScope.PER_DIMENSION) {
            requireExpectedDimensionCount(
                    values.length,
                    expectedDimensionCount
            );
            for (int feature = 0; feature < values.length; feature++) {
                addAccepted(values[feature], feature, sink);
            }
            return;
        }

        requireUnivariateCompatibility(expectedDimensionCount);
        accumulateFloatDimension(values, 0, sink);
    }

    private static void accumulateDoubleMultivariate(
            double[][] matrix,
            int expectedDimensionCount,
            StandardizationScope scope,
            ValueSink sink
    ) {
        requireExpectedDimensionCount(
                matrix.length,
                expectedDimensionCount
        );
        for (int dimension = 0; dimension < matrix.length; dimension++) {
            double[] values = Objects.requireNonNull(
                    matrix[dimension],
                    "Training series contains a null dimension at index "
                            + dimension
                            + "."
            );
            int group = groupForDimension(scope, dimension);
            accumulateDoubleDimension(values, group, sink);
        }
    }

    private static void accumulateFloatMultivariate(
            float[][] matrix,
            int expectedDimensionCount,
            StandardizationScope scope,
            ValueSink sink
    ) {
        requireExpectedDimensionCount(
                matrix.length,
                expectedDimensionCount
        );
        for (int dimension = 0; dimension < matrix.length; dimension++) {
            float[] values = Objects.requireNonNull(
                    matrix[dimension],
                    "Training series contains a null dimension at index "
                            + dimension
                            + "."
            );
            int group = groupForDimension(scope, dimension);
            accumulateFloatDimension(values, group, sink);
        }
    }

    private static void accumulateDoubleDimension(
            double[] values,
            int group,
            ValueSink sink
    ) {
        for (double value : values) {
            addAccepted(value, group, sink);
        }
    }

    private static void accumulateFloatDimension(
            float[] values,
            int group,
            ValueSink sink
    ) {
        for (float value : values) {
            addAccepted(value, group, sink);
        }
    }

    /**
     * Skips the numeric missing sentinel and rejects infinities before invoking
     * the trusted accumulator hot path.
     */
    private static void addAccepted(
            double value,
            int group,
            ValueSink sink
    ) {
        if (Double.isNaN(value)) {
            return;
        }
        if (!Double.isFinite(value)) {
            throw new IllegalArgumentException(
                    "Standardization fitting encountered a nonfinite value "
                            + "in statistic group "
                            + group
                            + ": "
                            + value
                            + ". NaN is the supported numeric missing "
                            + "sentinel; infinities are not supported."
            );
        }
        sink.add(group, value);
    }

    private static int groupForDimension(
            StandardizationScope scope,
            int dimension
    ) {
        return scope == StandardizationScope.GLOBAL
                ? 0
                : dimension;
    }

    private static double fittedStandardDeviation(
            OnlineMoments moments,
            VarianceConvention varianceConvention,
            int group
    ) {
        if (!moments.canCalculateVariance(varianceConvention)) {
            return CONSTANT_SCALE;
        }

        double standardDeviation =
                moments.getStandardDeviation(varianceConvention);
        if (!Double.isFinite(standardDeviation)) {
            throw new ArithmeticException(
                    "Z-score fitting produced a nonfinite standard "
                            + "deviation for statistic group "
                            + group
                            + "."
            );
        }
        return standardDeviation == 0.0
                ? CONSTANT_SCALE
                : standardDeviation;
    }

    private static double fittedRange(
            OnlineRange range,
            int group
    ) {
        if (range.isConstant()) {
            return CONSTANT_SCALE;
        }

        double fittedRange = range.getRange();
        if (!Double.isFinite(fittedRange) || fittedRange <= 0.0) {
            throw new ArithmeticException(
                    "Min-max fitting produced an invalid range for "
                            + "statistic group "
                            + group
                            + ": "
                            + fittedRange
                            + "."
            );
        }
        return fittedRange;
    }

    private static void requireObservations(
            boolean hasObservations,
            int group,
            StandardizationMethod method
    ) {
        if (!hasObservations) {
            throw new IllegalArgumentException(
                    "Standardization statistic group "
                            + group
                            + " contains no observed values for method "
                            + method
                            + "."
            );
        }
    }

    private static List<String> validateAndCopyFeatureNames(
            List<String> featureNames,
            int dimensionCount,
            StandardizationScope scope
    ) {
        if (featureNames == null || featureNames.isEmpty()) {
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

        List<String> copy = new ArrayList<>(featureNames.size());
        for (String featureName : featureNames) {
            if (featureName == null || featureName.isBlank()) {
                throw new IllegalArgumentException(
                        "Standardization feature names cannot be null or "
                                + "blank."
                );
            }
            copy.add(featureName.trim());
        }
        return Collections.unmodifiableList(copy);
    }

    private static void requireReusableMethod(
            StandardizationMethod method
    ) {
        if (method == StandardizationMethod.NONE) {
            throw new IllegalArgumentException(
                    "StandardizationFitter should not be invoked for "
                            + "StandardizationMethod.NONE."
            );
        }
        if (method == StandardizationMethod.ROBUST) {
            throw new UnsupportedOperationException(
                    "ROBUST standardization is recognized but its quantile "
                            + "convention and fitting strategy are not yet "
                            + "implemented."
            );
        }
    }

    private static void requireReusableScope(
            StandardizationScope scope
    ) {
        if (!scope.usesTrainingStatistics()) {
            throw new IllegalArgumentException(
                    "StandardizationFitter fits reusable training statistics "
                            + "only for GLOBAL and PER_DIMENSION. Scope "
                            + scope
                            + " calculates local parameters during each "
                            + "series transformation."
            );
        }
    }

    private static void requirePositiveDimensionCount(
            int dimensionCount
    ) {
        if (dimensionCount < 1) {
            throw new IllegalArgumentException(
                    "Numeric data must contain at least one realized "
                            + "dimension."
            );
        }
    }

    private static void requireUnivariateCompatibility(
            int expectedDimensionCount
    ) {
        requireExpectedDimensionCount(1, expectedDimensionCount);
    }

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
                            + ". PER_DIMENSION tabular rows and multivariate "
                            + "series must have consistent realized "
                            + "dimension counts."
            );
        }
    }

    private static UnsupportedOperationException lazyFittingUnsupported() {
        return new UnsupportedOperationException(
                "StandardizationFitter requires eager realized data. Lazy "
                        + "training standardization currently requires "
                        + "externally supplied reusable statistics."
        );
    }

    private static IllegalArgumentException unsupportedSeriesType(
            Object instance
    ) {
        return new IllegalArgumentException(
                "Unsupported standardization training type: "
                        + instance.getClass().getTypeName()
                        + ". Expected double[], float[], double[][], or "
                        + "float[][]."
        );
    }

    /**
     * Small internal abstraction that shares representation traversal without
     * boxing values or branching on the fitting method in the innermost loop.
     */
    private interface ValueSink {
        void add(int group, double value);
    }

    private static final class MomentSink implements ValueSink {
        private final OnlineMoments[] moments;

        private MomentSink(
                OnlineMoments[] moments
        ) {
            this.moments = moments;
        }

        @Override
        public void add(
                int group,
                double value
        ) {
            moments[group].add(value);
        }
    }

    private static final class RangeSink implements ValueSink {
        private final OnlineRange[] ranges;

        private RangeSink(
                OnlineRange[] ranges
        ) {
            this.ranges = ranges;
        }

        @Override
        public void add(
                int group,
                double value
        ) {
            ranges[group].add(value);
        }
    }
}
