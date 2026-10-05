package preprocessing.standardization;

import core.AppContext;
import datasets.ListObjectDataset;
import datasets.readers.lazy.LazySeriesRef;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * Applies affine forward and inverse standardization to realized primitive
 * numeric data.
 *
 * <p>Supported representations are {@code double[]}, {@code float[]},
 * {@code double[][]}, and {@code float[][]}. Multivariate arrays are
 * dimension-major. Transformations are performed in place, preserve primitive
 * NaN missing values, and preserve float-backed storage.</p>
 *
 * <p>Reusable {@link StandardizationStats} support GLOBAL and PER_DIMENSION.
 * PER_SERIES and PER_SERIES_PER_DIMENSION calculate local parameters from each
 * realized series and return {@link PerSeriesStandardizationState} so the
 * transformation can later be inverted.</p>
 *
 * <p>Contiguous homogeneous-parameter arrays use Java Vector API kernels when
 * {@link AppContext#useVectorApi} is enabled and the array is large enough.
 * Scalar kernels remain available for disabled vectorization, short arrays,
 * and scalar tails. One-dimensional PER_DIMENSION tabular rows also have a
 * vector path that loads contiguous center and reciprocal-scale arrays.</p>
 */
public final class Standardizer {

    private Standardizer() {
        // Utility class.
    }

    /* --------------------------------------------------------------------- */
    /* Reusable training-statistic forward transformation                    */
    /* --------------------------------------------------------------------- */

    public static ListObjectDataset transformInPlace(
            ListObjectDataset dataset,
            StandardizationStats stats,
            List<String> featureNames
    ) {
        return transformInPlace(
                dataset,
                stats,
                featureNames,
                AppContext.useVectorApi
        );
    }

    public static ListObjectDataset transformInPlace(
            ListObjectDataset dataset,
            StandardizationStats stats,
            List<String> featureNames,
            boolean useVectorApi
    ) {
        Objects.requireNonNull(dataset, "Dataset cannot be null.");
        Objects.requireNonNull(
                stats,
                "StandardizationStats cannot be null."
        );
        List<Object> data = Objects.requireNonNull(
                dataset.getData(),
                "Dataset data cannot be null."
        );
        stats.validateFeatureCompatibility(featureNames);

        PreparedParameters parameters = prepare(stats);
        for (Object instance : data) {
            transformReusableInstanceInPlace(
                    instance,
                    parameters,
                    useVectorApi,
                    false
            );
        }
        return dataset;
    }

    public static ListObjectDataset transformInPlace(
            ListObjectDataset dataset,
            StandardizationStats stats
    ) {
        requireUnnamedCompatibility(stats);
        return transformInPlace(
                dataset,
                stats,
                null,
                AppContext.useVectorApi
        );
    }

    public static Object transformInstanceInPlace(
            Object series,
            StandardizationStats stats
    ) {
        return transformInstanceInPlace(
                series,
                stats,
                AppContext.useVectorApi
        );
    }

    public static Object transformInstanceInPlace(
            Object series,
            StandardizationStats stats,
            boolean useVectorApi
    ) {
        Objects.requireNonNull(
                stats,
                "StandardizationStats cannot be null."
        );
        transformReusableInstanceInPlace(
                series,
                prepare(stats),
                useVectorApi,
                false
        );
        return series;
    }

    /* --------------------------------------------------------------------- */
    /* Reusable training-statistic inverse transformation                    */
    /* --------------------------------------------------------------------- */

    public static ListObjectDataset inverseTransformInPlace(
            ListObjectDataset dataset,
            StandardizationStats stats,
            List<String> featureNames
    ) {
        return inverseTransformInPlace(
                dataset,
                stats,
                featureNames,
                AppContext.useVectorApi
        );
    }

    public static ListObjectDataset inverseTransformInPlace(
            ListObjectDataset dataset,
            StandardizationStats stats,
            List<String> featureNames,
            boolean useVectorApi
    ) {
        Objects.requireNonNull(dataset, "Dataset cannot be null.");
        Objects.requireNonNull(
                stats,
                "StandardizationStats cannot be null."
        );
        List<Object> data = Objects.requireNonNull(
                dataset.getData(),
                "Dataset data cannot be null."
        );
        stats.validateFeatureCompatibility(featureNames);

        PreparedParameters parameters = prepare(stats);
        for (Object instance : data) {
            transformReusableInstanceInPlace(
                    instance,
                    parameters,
                    useVectorApi,
                    true
            );
        }
        return dataset;
    }

    public static ListObjectDataset inverseTransformInPlace(
            ListObjectDataset dataset,
            StandardizationStats stats
    ) {
        requireUnnamedCompatibility(stats);
        return inverseTransformInPlace(
                dataset,
                stats,
                null,
                AppContext.useVectorApi
        );
    }

    public static Object inverseTransformInstanceInPlace(
            Object series,
            StandardizationStats stats
    ) {
        return inverseTransformInstanceInPlace(
                series,
                stats,
                AppContext.useVectorApi
        );
    }

    public static Object inverseTransformInstanceInPlace(
            Object series,
            StandardizationStats stats,
            boolean useVectorApi
    ) {
        Objects.requireNonNull(
                stats,
                "StandardizationStats cannot be null."
        );
        transformReusableInstanceInPlace(
                series,
                prepare(stats),
                useVectorApi,
                true
        );
        return series;
    }

    /* --------------------------------------------------------------------- */
    /* Per-series fitting and forward transformation                         */
    /* --------------------------------------------------------------------- */

    /**
     * Calculates and applies local parameters to every eager dataset instance.
     * The returned immutable list is aligned with dataset order and contains
     * the state required for inverse transformation.
     */
    public static List<PerSeriesStandardizationState>
    transformPerSeriesInPlace(
            ListObjectDataset dataset,
            StandardizationMethod method,
            StandardizationScope scope,
            VarianceConvention varianceConvention
    ) {
        return transformPerSeriesInPlace(
                dataset,
                method,
                scope,
                varianceConvention,
                AppContext.useVectorApi
        );
    }

    public static List<PerSeriesStandardizationState>
    transformPerSeriesInPlace(
            ListObjectDataset dataset,
            StandardizationMethod method,
            StandardizationScope scope,
            VarianceConvention varianceConvention,
            boolean useVectorApi
    ) {
        Objects.requireNonNull(dataset, "Dataset cannot be null.");
        List<Object> data = Objects.requireNonNull(
                dataset.getData(),
                "Dataset data cannot be null."
        );
        requirePerSeriesConfiguration(method, scope, varianceConvention);

        List<PerSeriesStandardizationState> states =
                new ArrayList<>(data.size());
        for (Object instance : data) {
            states.add(
                    transformPerSeriesInstanceInPlace(
                            instance,
                            method,
                            scope,
                            varianceConvention,
                            useVectorApi
                    )
            );
        }
        return Collections.unmodifiableList(states);
    }

    public static PerSeriesStandardizationState
    transformPerSeriesInstanceInPlace(
            Object series,
            StandardizationMethod method,
            StandardizationScope scope,
            VarianceConvention varianceConvention
    ) {
        return transformPerSeriesInstanceInPlace(
                series,
                method,
                scope,
                varianceConvention,
                AppContext.useVectorApi
        );
    }

    public static PerSeriesStandardizationState
    transformPerSeriesInstanceInPlace(
            Object series,
            StandardizationMethod method,
            StandardizationScope scope,
            VarianceConvention varianceConvention,
            boolean useVectorApi
    ) {
        requireRealizedSeries(series);
        requirePerSeriesConfiguration(method, scope, varianceConvention);

        PerSeriesStandardizationState state = fitPerSeriesState(
                series,
                method,
                scope,
                varianceConvention
        );
        transformLocalInstanceInPlace(
                series,
                state,
                useVectorApi,
                false
        );
        return state;
    }

    /* --------------------------------------------------------------------- */
    /* Per-series inverse transformation                                     */
    /* --------------------------------------------------------------------- */

    public static Object inverseTransformInstanceInPlace(
            Object series,
            PerSeriesStandardizationState state
    ) {
        return inverseTransformInstanceInPlace(
                series,
                state,
                AppContext.useVectorApi
        );
    }

    public static Object inverseTransformInstanceInPlace(
            Object series,
            PerSeriesStandardizationState state,
            boolean useVectorApi
    ) {
        transformLocalInstanceInPlace(
                series,
                Objects.requireNonNull(
                        state,
                        "Per-series state cannot be null."
                ),
                useVectorApi,
                true
        );
        return series;
    }

    /* --------------------------------------------------------------------- */
    /* Reusable dispatch                                                     */
    /* --------------------------------------------------------------------- */

    private static void transformReusableInstanceInPlace(
            Object series,
            PreparedParameters parameters,
            boolean useVectorApi,
            boolean inverse
    ) {
        requireRealizedSeries(series);

        if (series instanceof double[] values) {
            transformDoubleOneDimensional(
                    values,
                    parameters.scope,
                    parameters.centers,
                    parameters.scales,
                    parameters.inverseScales,
                    useVectorApi,
                    inverse
            );
            return;
        }
        if (series instanceof float[] values) {
            transformFloatOneDimensional(
                    values,
                    parameters.scope,
                    parameters.centers,
                    parameters.scales,
                    parameters.inverseScales,
                    useVectorApi,
                    inverse
            );
            return;
        }
        if (series instanceof double[][] values) {
            transformDoubleMultivariate(
                    values,
                    parameters.scope,
                    parameters.centers,
                    parameters.scales,
                    parameters.inverseScales,
                    useVectorApi,
                    inverse
            );
            return;
        }
        if (series instanceof float[][] values) {
            transformFloatMultivariate(
                    values,
                    parameters.scope,
                    parameters.centers,
                    parameters.scales,
                    parameters.inverseScales,
                    useVectorApi,
                    inverse
            );
            return;
        }
        throw unsupportedSeriesType(series);
    }

    private static void transformDoubleOneDimensional(
            double[] values,
            StandardizationScope scope,
            double[] centers,
            double[] scales,
            double[] inverseScales,
            boolean useVectorApi,
            boolean inverse
    ) {
        if (scope == StandardizationScope.PER_DIMENSION) {
            requireDimensionCompatibility(
                    values.length,
                    centers.length,
                    "Tabular row"
            );
            transformDoubleTabular(
                    values,
                    centers,
                    inverse ? scales : inverseScales,
                    useVectorApi,
                    inverse
            );
            return;
        }
        requireGlobalParameters(centers, scales);
        transformDoubleDimension(
                values,
                centers[0],
                inverse ? scales[0] : inverseScales[0],
                useVectorApi,
                inverse
        );
    }

    private static void transformFloatOneDimensional(
            float[] values,
            StandardizationScope scope,
            double[] centers,
            double[] scales,
            double[] inverseScales,
            boolean useVectorApi,
            boolean inverse
    ) {
        if (scope == StandardizationScope.PER_DIMENSION) {
            requireDimensionCompatibility(
                    values.length,
                    centers.length,
                    "Tabular row"
            );
            transformFloatTabular(
                    values,
                    centers,
                    inverse ? scales : inverseScales,
                    useVectorApi,
                    inverse
            );
            return;
        }
        requireGlobalParameters(centers, scales);
        transformFloatDimension(
                values,
                centers[0],
                inverse ? scales[0] : inverseScales[0],
                useVectorApi,
                inverse
        );
    }

    private static void transformDoubleMultivariate(
            double[][] values,
            StandardizationScope scope,
            double[] centers,
            double[] scales,
            double[] inverseScales,
            boolean useVectorApi,
            boolean inverse
    ) {
        requirePositiveDimensionCount(values.length);
        if (scope == StandardizationScope.PER_DIMENSION) {
            requireDimensionCompatibility(
                    values.length,
                    centers.length,
                    "Multivariate series"
            );
        } else {
            requireGlobalParameters(centers, scales);
        }

        for (int dimension = 0; dimension < values.length; dimension++) {
            double[] row = requireDimension(values[dimension], dimension);
            int group = scope == StandardizationScope.GLOBAL ? 0 : dimension;
            transformDoubleDimension(
                    row,
                    centers[group],
                    inverse ? scales[group] : inverseScales[group],
                    useVectorApi,
                    inverse
            );
        }
    }

    private static void transformFloatMultivariate(
            float[][] values,
            StandardizationScope scope,
            double[] centers,
            double[] scales,
            double[] inverseScales,
            boolean useVectorApi,
            boolean inverse
    ) {
        requirePositiveDimensionCount(values.length);
        if (scope == StandardizationScope.PER_DIMENSION) {
            requireDimensionCompatibility(
                    values.length,
                    centers.length,
                    "Multivariate series"
            );
        } else {
            requireGlobalParameters(centers, scales);
        }

        for (int dimension = 0; dimension < values.length; dimension++) {
            float[] row = requireDimension(values[dimension], dimension);
            int group = scope == StandardizationScope.GLOBAL ? 0 : dimension;
            transformFloatDimension(
                    row,
                    centers[group],
                    inverse ? scales[group] : inverseScales[group],
                    useVectorApi,
                    inverse
            );
        }
    }

    /* --------------------------------------------------------------------- */
    /* Per-series state fitting and dispatch                                 */
    /* --------------------------------------------------------------------- */

    private static PerSeriesStandardizationState fitPerSeriesState(
            Object series,
            StandardizationMethod method,
            StandardizationScope scope,
            VarianceConvention varianceConvention
    ) {
        int dimensionCount = dimensionCount(series);
        int groupCount = scope.perSeriesGroupCount(dimensionCount);
        long[] counts = new long[groupCount];
        double[] centers = new double[groupCount];
        double[] scales = new double[groupCount];

        if (method == StandardizationMethod.MIN_MAX) {
            OnlineRange[] ranges = new OnlineRange[groupCount];
            for (int group = 0; group < groupCount; group++) {
                ranges[group] = new OnlineRange();
            }
            accumulateLocal(series, scope, (group, value) ->
                    ranges[group].add(value));
            for (int group = 0; group < groupCount; group++) {
                OnlineRange range = ranges[group];
                requireObservedLocalGroup(range.hasObservations(), group);
                counts[group] = range.getCount();
                centers[group] = range.getMinimum();
                scales[group] = range.isConstant()
                        ? StandardizationFitter.CONSTANT_SCALE
                        : requireFinitePositiveRange(range.getRange(), group);
            }
        } else {
            OnlineMoments[] moments = new OnlineMoments[groupCount];
            for (int group = 0; group < groupCount; group++) {
                moments[group] = new OnlineMoments();
            }
            accumulateLocal(series, scope, (group, value) ->
                    moments[group].add(value));
            for (int group = 0; group < groupCount; group++) {
                OnlineMoments moment = moments[group];
                requireObservedLocalGroup(moment.hasObservations(), group);
                counts[group] = moment.getCount();
                centers[group] = moment.getMean();
                scales[group] = method == StandardizationMethod.MEAN_CENTER
                        ? StandardizationFitter.CONSTANT_SCALE
                        : fittedLocalStandardDeviation(
                                moment,
                                varianceConvention,
                                group
                        );
            }
        }

        return new PerSeriesStandardizationState(
                method,
                scope,
                varianceConvention,
                counts,
                centers,
                scales
        );
    }

    private static void accumulateLocal(
            Object series,
            StandardizationScope scope,
            LocalValueSink sink
    ) {
        if (series instanceof double[] values) {
            accumulateDoubleLocal(values, 0, sink);
            return;
        }
        if (series instanceof float[] values) {
            accumulateFloatLocal(values, 0, sink);
            return;
        }
        if (series instanceof double[][] matrix) {
            requirePositiveDimensionCount(matrix.length);
            for (int dimension = 0; dimension < matrix.length; dimension++) {
                int group = scope == StandardizationScope.PER_SERIES
                        ? 0 : dimension;
                accumulateDoubleLocal(
                        requireDimension(matrix[dimension], dimension),
                        group,
                        sink
                );
            }
            return;
        }
        if (series instanceof float[][] matrix) {
            requirePositiveDimensionCount(matrix.length);
            for (int dimension = 0; dimension < matrix.length; dimension++) {
                int group = scope == StandardizationScope.PER_SERIES
                        ? 0 : dimension;
                accumulateFloatLocal(
                        requireDimension(matrix[dimension], dimension),
                        group,
                        sink
                );
            }
            return;
        }
        throw unsupportedSeriesType(series);
    }

    private static void accumulateDoubleLocal(
            double[] values,
            int group,
            LocalValueSink sink
    ) {
        for (double value : values) {
            addLocalValue(value, group, sink);
        }
    }

    private static void accumulateFloatLocal(
            float[] values,
            int group,
            LocalValueSink sink
    ) {
        for (float value : values) {
            addLocalValue(value, group, sink);
        }
    }

    private static void addLocalValue(
            double value,
            int group,
            LocalValueSink sink
    ) {
        if (Double.isNaN(value)) {
            return;
        }
        if (!Double.isFinite(value)) {
            throw new IllegalArgumentException(
                    "Per-series standardization encountered a nonfinite "
                            + "value in parameter group "
                            + group
                            + ": "
                            + value
                            + "."
            );
        }
        sink.add(group, value);
    }

    private static void transformLocalInstanceInPlace(
            Object series,
            PerSeriesStandardizationState state,
            boolean useVectorApi,
            boolean inverse
    ) {
        requireRealizedSeries(series);
        double[] centers = state.getCenters();
        double[] scales = state.getScales();
        double[] factors = inverse
                ? scales
                : reciprocalScales(scales);
        StandardizationScope scope = state.getScope();

        if (series instanceof double[] values) {
            requireSingleLocalGroup(centers);
            transformDoubleDimension(
                    values,
                    centers[0],
                    factors[0],
                    useVectorApi,
                    inverse
            );
            return;
        }
        if (series instanceof float[] values) {
            requireSingleLocalGroup(centers);
            transformFloatDimension(
                    values,
                    centers[0],
                    factors[0],
                    useVectorApi,
                    inverse
            );
            return;
        }
        if (series instanceof double[][] matrix) {
            requireLocalDimensionCompatibility(matrix.length, scope, centers);
            for (int dimension = 0; dimension < matrix.length; dimension++) {
                int group = scope == StandardizationScope.PER_SERIES
                        ? 0 : dimension;
                transformDoubleDimension(
                        requireDimension(matrix[dimension], dimension),
                        centers[group],
                        factors[group],
                        useVectorApi,
                        inverse
                );
            }
            return;
        }
        if (series instanceof float[][] matrix) {
            requireLocalDimensionCompatibility(matrix.length, scope, centers);
            for (int dimension = 0; dimension < matrix.length; dimension++) {
                int group = scope == StandardizationScope.PER_SERIES
                        ? 0 : dimension;
                transformFloatDimension(
                        requireDimension(matrix[dimension], dimension),
                        centers[group],
                        factors[group],
                        useVectorApi,
                        inverse
                );
            }
            return;
        }
        throw unsupportedSeriesType(series);
    }

    /* --------------------------------------------------------------------- */
    /* Scalar and Vector API kernels                                         */
    /* --------------------------------------------------------------------- */

    private static void transformDoubleDimension(
            double[] values,
            double center,
            double factor,
            boolean useVectorApi,
            boolean inverse
    ) {
        if (useVectorApi) {
            VectorBridge.transformDoubleDimension(
                    values, center, factor, inverse
            );
            return;
        }
        transformDoubleDimensionScalar(values, center, factor, inverse);
    }

    private static void transformDoubleDimensionScalar(
            double[] values,
            double center,
            double factor,
            boolean inverse
    ) {
        if (inverse) {
            for (int index = 0; index < values.length; index++) {
                values[index] = values[index] * factor + center;
            }
        } else {
            for (int index = 0; index < values.length; index++) {
                values[index] = (values[index] - center) * factor;
            }
        }
    }

    private static void transformFloatDimension(
            float[] values,
            double center,
            double factor,
            boolean useVectorApi,
            boolean inverse
    ) {
        float floatCenter = (float) center;
        float floatFactor = (float) factor;
        if (useVectorApi) {
            VectorBridge.transformFloatDimension(
                    values, floatCenter, floatFactor, inverse
            );
            return;
        }
        transformFloatDimensionScalar(
                values, floatCenter, floatFactor, inverse
        );
    }

    private static void transformFloatDimensionScalar(
            float[] values,
            float center,
            float factor,
            boolean inverse
    ) {
        if (inverse) {
            for (int index = 0; index < values.length; index++) {
                values[index] = values[index] * factor + center;
            }
        } else {
            for (int index = 0; index < values.length; index++) {
                values[index] = (values[index] - center) * factor;
            }
        }
    }

    private static void transformDoubleTabular(
            double[] values,
            double[] centers,
            double[] factors,
            boolean useVectorApi,
            boolean inverse
    ) {
        if (useVectorApi) {
            VectorBridge.transformDoubleTabular(
                    values, centers, factors, inverse
            );
            return;
        }
        transformDoubleTabularScalar(values, centers, factors, inverse);
    }

    private static void transformDoubleTabularScalar(
            double[] values,
            double[] centers,
            double[] factors,
            boolean inverse
    ) {
        if (inverse) {
            for (int index = 0; index < values.length; index++) {
                values[index] = values[index] * factors[index]
                        + centers[index];
            }
        } else {
            for (int index = 0; index < values.length; index++) {
                values[index] = (values[index] - centers[index])
                        * factors[index];
            }
        }
    }

    private static void transformFloatTabular(
            float[] values,
            double[] centers,
            double[] factors,
            boolean useVectorApi,
            boolean inverse
    ) {
        if (inverse) {
            for (int index = 0; index < values.length; index++) {
                values[index] = (float) (
                        (double) values[index] * factors[index]
                                + centers[index]
                );
            }
        } else {
            for (int index = 0; index < values.length; index++) {
                values[index] = (float) (
                        ((double) values[index] - centers[index])
                                * factors[index]
                );
            }
        }
    }

    /**
     * Reflective bridge that prevents scalar-only class loading from resolving
     * any incubator Vector API type. It initializes only on a vector request.
     */
    private static final class VectorBridge {
        private static final Class<?> KERNELS;

        static {
            try {
                KERNELS = Class.forName(
                        Standardizer.class.getName() + "$VectorKernels"
                );
            } catch (ClassNotFoundException | LinkageError exception) {
                throw unavailable(exception);
            }
        }

        private VectorBridge() {
        }

        private static void transformDoubleDimension(
                double[] values, double center, double factor, boolean inverse
        ) {
            invoke("transformDoubleDimension",
                    new Class<?>[]{double[].class, double.class,
                            double.class, boolean.class},
                    values, center, factor, inverse);
        }

        private static void transformFloatDimension(
                float[] values, float center, float factor, boolean inverse
        ) {
            invoke("transformFloatDimension",
                    new Class<?>[]{float[].class, float.class,
                            float.class, boolean.class},
                    values, center, factor, inverse);
        }

        private static void transformDoubleTabular(
                double[] values, double[] centers, double[] factors,
                boolean inverse
        ) {
            invoke("transformDoubleTabular",
                    new Class<?>[]{double[].class, double[].class,
                            double[].class, boolean.class},
                    values, centers, factors, inverse);
        }

        private static void invoke(
                String name, Class<?>[] parameterTypes, Object... arguments
        ) {
            try {
                java.lang.reflect.Method method =
                        KERNELS.getDeclaredMethod(name, parameterTypes);
                method.setAccessible(true);
                method.invoke(null, arguments);
            } catch (java.lang.reflect.InvocationTargetException exception) {
                Throwable cause = exception.getCause();
                if (cause instanceof RuntimeException runtimeException) {
                    throw runtimeException;
                }
                if (cause instanceof Error error) {
                    throw error;
                }
                throw new IllegalStateException(
                        "Vector standardization kernel failed.", cause
                );
            } catch (ReflectiveOperationException | LinkageError exception) {
                throw unavailable(exception);
            }
        }

        private static IllegalStateException unavailable(Throwable cause) {
            return new IllegalStateException(
                    "Vector API standardization was requested, but "
                            + "jdk.incubator.vector is unavailable. Launch "
                            + "Java with --add-modules jdk.incubator.vector "
                            + "or set use_vector_api=false.",
                    cause
            );
        }
    }

    /** All incubator references are isolated in this lazily loaded class. */
    private static final class VectorKernels {
        private static final int VECTOR_LOOP_MULTIPLIER = 2;
        private static final
        jdk.incubator.vector.VectorSpecies<Double> DOUBLE_SPECIES =
                jdk.incubator.vector.DoubleVector.SPECIES_PREFERRED;
        private static final
        jdk.incubator.vector.VectorSpecies<Float> FLOAT_SPECIES =
                jdk.incubator.vector.FloatVector.SPECIES_PREFERRED;

        private VectorKernels() {
        }

        private static void transformDoubleDimension(
                double[] values, double center, double factor, boolean inverse
        ) {
            if (values.length < DOUBLE_SPECIES.length()
                    * VECTOR_LOOP_MULTIPLIER) {
                transformDoubleDimensionScalar(
                        values, center, factor, inverse
                );
                return;
            }
            int index = 0;
            int upperBound = DOUBLE_SPECIES.loopBound(values.length);
            jdk.incubator.vector.DoubleVector centerVector =
                    jdk.incubator.vector.DoubleVector.broadcast(
                            DOUBLE_SPECIES, center
                    );
            jdk.incubator.vector.DoubleVector factorVector =
                    jdk.incubator.vector.DoubleVector.broadcast(
                            DOUBLE_SPECIES, factor
                    );
            for (; index < upperBound; index += DOUBLE_SPECIES.length()) {
                jdk.incubator.vector.DoubleVector vector =
                        jdk.incubator.vector.DoubleVector.fromArray(
                                DOUBLE_SPECIES, values, index
                        );
                (inverse
                        ? vector.mul(factorVector).add(centerVector)
                        : vector.sub(centerVector).mul(factorVector))
                        .intoArray(values, index);
            }
            if (inverse) {
                for (; index < values.length; index++) {
                    values[index] = values[index] * factor + center;
                }
            } else {
                for (; index < values.length; index++) {
                    values[index] = (values[index] - center) * factor;
                }
            }
        }

        private static void transformFloatDimension(
                float[] values, float center, float factor, boolean inverse
        ) {
            if (values.length < FLOAT_SPECIES.length()
                    * VECTOR_LOOP_MULTIPLIER) {
                transformFloatDimensionScalar(
                        values, center, factor, inverse
                );
                return;
            }
            int index = 0;
            int upperBound = FLOAT_SPECIES.loopBound(values.length);
            jdk.incubator.vector.FloatVector centerVector =
                    jdk.incubator.vector.FloatVector.broadcast(
                            FLOAT_SPECIES, center
                    );
            jdk.incubator.vector.FloatVector factorVector =
                    jdk.incubator.vector.FloatVector.broadcast(
                            FLOAT_SPECIES, factor
                    );
            for (; index < upperBound; index += FLOAT_SPECIES.length()) {
                jdk.incubator.vector.FloatVector vector =
                        jdk.incubator.vector.FloatVector.fromArray(
                                FLOAT_SPECIES, values, index
                        );
                (inverse
                        ? vector.mul(factorVector).add(centerVector)
                        : vector.sub(centerVector).mul(factorVector))
                        .intoArray(values, index);
            }
            if (inverse) {
                for (; index < values.length; index++) {
                    values[index] = values[index] * factor + center;
                }
            } else {
                for (; index < values.length; index++) {
                    values[index] = (values[index] - center) * factor;
                }
            }
        }

        private static void transformDoubleTabular(
                double[] values, double[] centers, double[] factors,
                boolean inverse
        ) {
            if (values.length < DOUBLE_SPECIES.length()
                    * VECTOR_LOOP_MULTIPLIER) {
                transformDoubleTabularScalar(
                        values, centers, factors, inverse
                );
                return;
            }
            int index = 0;
            int upperBound = DOUBLE_SPECIES.loopBound(values.length);
            for (; index < upperBound; index += DOUBLE_SPECIES.length()) {
                jdk.incubator.vector.DoubleVector valuesVector =
                        jdk.incubator.vector.DoubleVector.fromArray(
                                DOUBLE_SPECIES, values, index
                        );
                jdk.incubator.vector.DoubleVector centersVector =
                        jdk.incubator.vector.DoubleVector.fromArray(
                                DOUBLE_SPECIES, centers, index
                        );
                jdk.incubator.vector.DoubleVector factorsVector =
                        jdk.incubator.vector.DoubleVector.fromArray(
                                DOUBLE_SPECIES, factors, index
                        );
                (inverse
                        ? valuesVector.mul(factorsVector).add(centersVector)
                        : valuesVector.sub(centersVector).mul(factorsVector))
                        .intoArray(values, index);
            }
            if (inverse) {
                for (; index < values.length; index++) {
                    values[index] = values[index] * factors[index]
                            + centers[index];
                }
            } else {
                for (; index < values.length; index++) {
                    values[index] = (values[index] - centers[index])
                            * factors[index];
                }
            }
        }
    }

    /* --------------------------------------------------------------------- */
    /* Validation and preparation                                            */
    /* --------------------------------------------------------------------- */

    private static PreparedParameters prepare(
            StandardizationStats stats
    ) {
        StandardizationScope scope = stats.getScope();
        if (!scope.usesTrainingStatistics()) {
            throw new IllegalArgumentException(
                    "Reusable StandardizationStats cannot use scope "
                            + scope
                            + "."
            );
        }
        double[] centers = stats.getCenters();
        double[] scales = stats.getScales();
        validatePreparedParameters(scope, centers, scales);
        return new PreparedParameters(
                scope,
                centers,
                scales,
                reciprocalScales(scales)
        );
    }

    private static double[] reciprocalScales(
            double[] scales
    ) {
        double[] inverseScales = new double[scales.length];
        for (int group = 0; group < scales.length; group++) {
            inverseScales[group] = 1.0 / scales[group];
        }
        return inverseScales;
    }

    private static void validatePreparedParameters(
            StandardizationScope scope,
            double[] centers,
            double[] scales
    ) {
        Objects.requireNonNull(scope, "Scope cannot be null.");
        Objects.requireNonNull(centers, "Centers cannot be null.");
        Objects.requireNonNull(scales, "Scales cannot be null.");
        if (centers.length == 0 || centers.length != scales.length) {
            throw new IllegalArgumentException(
                    "Prepared centers and scales must have identical, "
                            + "nonzero lengths."
            );
        }
        if (scope == StandardizationScope.GLOBAL && centers.length != 1) {
            throw new IllegalArgumentException(
                    "GLOBAL standardization requires one parameter group."
            );
        }
    }

    private static void requirePerSeriesConfiguration(
            StandardizationMethod method,
            StandardizationScope scope,
            VarianceConvention varianceConvention
    ) {
        Objects.requireNonNull(method, "Method cannot be null.");
        Objects.requireNonNull(scope, "Scope cannot be null.");
        Objects.requireNonNull(
                varianceConvention,
                "Variance convention cannot be null."
        );
        method.requireImplemented();
        scope.requireImplemented();
        if (method == StandardizationMethod.NONE) {
            throw new IllegalArgumentException(
                    "Per-series transformation state is unnecessary for NONE."
            );
        }
        if (!scope.usesPerSeriesStatistics()) {
            throw new IllegalArgumentException(
                    "Per-series transformation requires PER_SERIES or "
                            + "PER_SERIES_PER_DIMENSION, but received "
                            + scope
                            + "."
            );
        }
    }

    private static int dimensionCount(
            Object series
    ) {
        if (series instanceof double[] || series instanceof float[]) {
            return 1;
        }
        if (series instanceof double[][] matrix) {
            requirePositiveDimensionCount(matrix.length);
            return matrix.length;
        }
        if (series instanceof float[][] matrix) {
            requirePositiveDimensionCount(matrix.length);
            return matrix.length;
        }
        throw unsupportedSeriesType(series);
    }

    private static double fittedLocalStandardDeviation(
            OnlineMoments moments,
            VarianceConvention convention,
            int group
    ) {
        if (!moments.canCalculateVariance(convention)) {
            return StandardizationFitter.CONSTANT_SCALE;
        }
        double standardDeviation = moments.getStandardDeviation(convention);
        if (!Double.isFinite(standardDeviation)) {
            throw new ArithmeticException(
                    "Per-series z-score fitting produced a nonfinite standard "
                            + "deviation for group "
                            + group
                            + "."
            );
        }
        return standardDeviation == 0.0
                ? StandardizationFitter.CONSTANT_SCALE
                : standardDeviation;
    }

    private static double requireFinitePositiveRange(
            double range,
            int group
    ) {
        if (!Double.isFinite(range) || range <= 0.0) {
            throw new ArithmeticException(
                    "Per-series min-max fitting produced an invalid range "
                            + "for group "
                            + group
                            + ": "
                            + range
                            + "."
            );
        }
        return range;
    }

    private static void requireObservedLocalGroup(
            boolean observed,
            int group
    ) {
        if (!observed) {
            throw new IllegalArgumentException(
                    "Per-series parameter group "
                            + group
                            + " contains no observed values."
            );
        }
    }

    private static void requireUnnamedCompatibility(
            StandardizationStats stats
    ) {
        Objects.requireNonNull(
                stats,
                "StandardizationStats cannot be null."
        );
        if (stats.hasFeatureNames()) {
            throw new IllegalArgumentException(
                    "Statistics contain ordered feature names. Supply feature "
                            + "names to the dataset transformation overload."
            );
        }
    }

    private static void requireRealizedSeries(
            Object series
    ) {
        Objects.requireNonNull(series, "Series cannot be null.");
        if (series instanceof LazySeriesRef) {
            throw new UnsupportedOperationException(
                    "Cannot transform a LazySeriesRef directly. Transform the "
                            + "realized numeric series during or after "
                            + "materialization."
            );
        }
    }

    private static void requireGlobalParameters(
            double[] centers,
            double[] scales
    ) {
        if (centers.length != 1 || scales.length != 1) {
            throw new IllegalArgumentException(
                    "GLOBAL standardization requires one center and scale."
            );
        }
    }

    private static void requireSingleLocalGroup(
            double[] centers
    ) {
        if (centers.length != 1) {
            throw new IllegalArgumentException(
                    "Univariate per-series transformation requires one local "
                            + "parameter group, but received "
                            + centers.length
                            + "."
            );
        }
    }

    private static void requireLocalDimensionCompatibility(
            int dimensionCount,
            StandardizationScope scope,
            double[] centers
    ) {
        requirePositiveDimensionCount(dimensionCount);
        int expected = scope == StandardizationScope.PER_SERIES
                ? 1
                : dimensionCount;
        if (centers.length != expected) {
            throw new IllegalArgumentException(
                    "Per-series state contains "
                            + centers.length
                            + " groups; expected "
                            + expected
                            + " for "
                            + scope
                            + "."
            );
        }
    }

    private static void requireDimensionCompatibility(
            int actual,
            int expected,
            String representation
    ) {
        requirePositiveDimensionCount(actual);
        if (actual != expected) {
            throw new IllegalArgumentException(
                    representation
                            + " contains "
                            + actual
                            + " dimensions or features, but statistics contain "
                            + expected
                            + " groups."
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

    private static double[] requireDimension(
            double[] dimension,
            int index
    ) {
        return Objects.requireNonNull(
                dimension,
                "Numeric series contains a null dimension at index "
                        + index
                        + "."
        );
    }

    private static float[] requireDimension(
            float[] dimension,
            int index
    ) {
        return Objects.requireNonNull(
                dimension,
                "Numeric series contains a null dimension at index "
                        + index
                        + "."
        );
    }

    private static IllegalArgumentException unsupportedSeriesType(
            Object series
    ) {
        return new IllegalArgumentException(
                "Unsupported standardization series type: "
                        + series.getClass().getTypeName()
                        + ". Expected double[], float[], double[][], or "
                        + "float[][]."
        );
    }

    private record PreparedParameters(
            StandardizationScope scope,
            double[] centers,
            double[] scales,
            double[] inverseScales
    ) {
    }

    @FunctionalInterface
    private interface LocalValueSink {
        void add(int group, double value);
    }
}
