package preprocessing.standardization;

import datasets.ListObjectDataset;
import datasets.readers.lazy.LazySeriesRef;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * Fits reusable standardization statistics from an eager numeric
 * ListObjectDataset.
 *
 * Phase 1 supports:
 *
 *     method:
 *         Z_SCORE
 *
 *     scopes:
 *         GLOBAL
 *         PER_DIMENSION
 *
 *     instance representations:
 *         double[]
 *         Double[]
 *         double[][]
 *         Double[][]
 *
 * Univariate arrays are treated as containing one dimension.
 *
 * Multivariate arrays are expected to use dimension-major orientation:
 *
 *     data[dimension][time]
 *
 * Series may have unequal lengths. For PER_DIMENSION fitting, values from
 * dimension d are accumulated together across every training instance and
 * time point.
 *
 * Missing-value behavior:
 *
 *     Double null:
 *         skipped
 *
 *     NaN:
 *         skipped
 *
 *     positive or negative infinity:
 *         rejected
 *
 * LazySeriesRef instances are rejected. Initial Phase 1 fitting operates on
 * eager datasets only. Lazy fitting can later be implemented as an explicit
 * streaming pass over the source files.
 */
public final class StandardizationFitter {

    /**
     * Default scale used for a constant group or a group with too few
     * observations to calculate variance under the requested convention.
     *
     * Using scale 1.0 causes centered constant values to become zero:
     *
     *     (x - center) / 1.0 = 0.0
     */
    public static final double CONSTANT_SCALE = 1.0;

    private StandardizationFitter() {
    }

    /**
     * Fits z-score statistics using per-dimension scope and population
     * variance.
     *
     * @param dataset eager numeric training dataset
     * @return fitted standardization statistics
     */
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

    /**
     * Fits z-score statistics using the supplied scope and population
     * variance.
     *
     * @param dataset eager numeric training dataset
     * @param scope global or per-dimension fitting scope
     * @return fitted standardization statistics
     */
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

    /**
     * Fits z-score statistics using the supplied scope and variance
     * convention.
     *
     * @param dataset eager numeric training dataset
     * @param scope global or per-dimension fitting scope
     * @param varianceConvention population or sample variance
     * @return fitted standardization statistics
     */
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

    /**
     * Fits z-score statistics with optional ordered feature names.
     *
     * @param dataset eager numeric training dataset
     * @param scope global or per-dimension fitting scope
     * @param varianceConvention population or sample variance
     * @param featureNames ordered feature names, or an empty list
     * @return fitted standardization statistics
     */
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
     * Fits reusable standardization statistics from an eager numeric
     * training dataset.
     *
     * In Phase 1, only Z_SCORE with GLOBAL or PER_DIMENSION scope is
     * implemented.
     *
     * @param dataset eager numeric training dataset
     * @param method standardization formula
     * @param scope statistics-fitting scope
     * @param varianceConvention population or sample variance
     * @param featureNames optional ordered feature names
     * @return immutable fitted statistics
     */
    public static StandardizationStats fit(
            ListObjectDataset dataset,
            StandardizationMethod method,
            StandardizationScope scope,
            VarianceConvention varianceConvention,
            List<String> featureNames
    ) {
        validateConfiguration(
                dataset,
                method,
                scope,
                varianceConvention
        );

        DatasetShape shape =
                inspectDatasetShape(dataset);

        List<String> normalizedFeatureNames =
                validateAndCopyFeatureNames(
                        featureNames,
                        shape.dimensionCount,
                        scope
                );

        OnlineMoments[] moments =
                createAccumulators(
                        scope,
                        shape.dimensionCount
                );

        for (int instanceIndex = 0;
             instanceIndex < dataset.getData().size();
             instanceIndex++) {

            Object instance =
                    dataset.getData().get(instanceIndex);

            if (instance == null) {
                throw new IllegalArgumentException(
                        "Training dataset contains a null instance at index "
                                + instanceIndex
                                + "."
                );
            }

            accumulateInstance(
                    instance,
                    instanceIndex,
                    shape,
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

    private static void validateConfiguration(
            ListObjectDataset dataset,
            StandardizationMethod method,
            StandardizationScope scope,
            VarianceConvention varianceConvention
    ) {
        Objects.requireNonNull(
                dataset,
                "Training dataset cannot be null."
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

        method.requireImplemented();
        scope.requireImplemented();

        if (method == StandardizationMethod.NONE) {
            throw new IllegalArgumentException(
                    "Standardization statistics cannot be fitted for "
                            + "StandardizationMethod.NONE."
            );
        }

        if (method != StandardizationMethod.Z_SCORE) {
            throw new UnsupportedOperationException(
                    "StandardizationFitter currently supports only "
                            + "Z_SCORE, but received: "
                            + method
            );
        }

        if (dataset.getData() == null
                || dataset.getData().isEmpty()) {

            throw new IllegalArgumentException(
                    "Cannot fit standardization statistics from an "
                            + "empty dataset."
            );
        }
    }

    /**
     * Determines the numeric representation and fixed dimension count.
     *
     * Series lengths may differ, but every instance must use the same broad
     * representation family and dimension count.
     */
    private static DatasetShape inspectDatasetShape(
            ListObjectDataset dataset
    ) {
        NumericRepresentation representation =
                null;

        int dimensionCount =
                -1;

        for (int instanceIndex = 0;
             instanceIndex < dataset.getData().size();
             instanceIndex++) {

            Object instance =
                    dataset.getData().get(instanceIndex);

            if (instance == null) {
                throw new IllegalArgumentException(
                        "Training dataset contains a null instance at index "
                                + instanceIndex
                                + "."
                );
            }

            if (instance instanceof LazySeriesRef) {
                throw new UnsupportedOperationException(
                        "StandardizationFitter cannot fit statistics "
                                + "directly from LazySeriesRef objects. "
                                + "Supply precomputed statistics or use a "
                                + "future streaming lazy-data fitting pass."
                );
            }

            NumericRepresentation currentRepresentation =
                    numericRepresentationOf(
                            instance,
                            instanceIndex
                    );

            int currentDimensionCount =
                    dimensionCountOf(
                            instance,
                            currentRepresentation
                    );

            if (currentDimensionCount <= 0) {
                throw new IllegalArgumentException(
                        "Training instance "
                                + instanceIndex
                                + " contains no dimensions."
                );
            }

            if (representation == null) {
                representation =
                        currentRepresentation;

                dimensionCount =
                        currentDimensionCount;

                continue;
            }

            if (representation != currentRepresentation) {
                throw new IllegalArgumentException(
                        "Training dataset mixes numeric array "
                                + "representations. The first instance uses "
                                + representation
                                + ", while instance "
                                + instanceIndex
                                + " uses "
                                + currentRepresentation
                                + "."
                );
            }

            if (dimensionCount != currentDimensionCount) {
                throw new IllegalArgumentException(
                        "Inconsistent dimension count at training instance "
                                + instanceIndex
                                + ". Expected "
                                + dimensionCount
                                + " dimensions but found "
                                + currentDimensionCount
                                + "."
                );
            }
        }

        if (representation == null) {
            throw new IllegalArgumentException(
                    "Could not determine the training dataset's numeric "
                            + "representation."
            );
        }

        return new DatasetShape(
                representation,
                dimensionCount
        );
    }

    private static NumericRepresentation numericRepresentationOf(
            Object instance,
            int instanceIndex
    ) {
        if (instance instanceof double[]) {
            return NumericRepresentation.PRIMITIVE_UNIVARIATE;
        }

        if (instance instanceof Double[]) {
            return NumericRepresentation.BOXED_UNIVARIATE;
        }

        if (instance instanceof double[][]) {
            return NumericRepresentation.PRIMITIVE_MULTIVARIATE;
        }

        if (instance instanceof Double[][]) {
            return NumericRepresentation.BOXED_MULTIVARIATE;
        }

        throw new IllegalArgumentException(
                "Unsupported training instance type at index "
                        + instanceIndex
                        + ": "
                        + instance.getClass().getName()
                        + ". Phase 1 standardization supports double[], "
                        + "Double[], double[][], and Double[][]."
        );
    }

    private static int dimensionCountOf(
            Object instance,
            NumericRepresentation representation
    ) {
        return switch (representation) {
            case PRIMITIVE_UNIVARIATE,
                    BOXED_UNIVARIATE ->
                    1;

            case PRIMITIVE_MULTIVARIATE ->
                    ((double[][]) instance).length;

            case BOXED_MULTIVARIATE ->
                    ((Double[][]) instance).length;
        };
    }

    private static OnlineMoments[] createAccumulators(
            StandardizationScope scope,
            int dimensionCount
    ) {
        int groupCount =
                scope.statisticGroupCount(
                        dimensionCount
                );

        OnlineMoments[] moments =
                new OnlineMoments[groupCount];

        for (int groupIndex = 0;
             groupIndex < groupCount;
             groupIndex++) {

            moments[groupIndex] =
                    new OnlineMoments();
        }

        return moments;
    }

    private static void accumulateInstance(
            Object instance,
            int instanceIndex,
            DatasetShape shape,
            StandardizationScope scope,
            OnlineMoments[] moments
    ) {
        switch (shape.representation) {
            case PRIMITIVE_UNIVARIATE ->
                    accumulatePrimitiveUnivariate(
                            (double[]) instance,
                            instanceIndex,
                            moments[0]
                    );

            case BOXED_UNIVARIATE ->
                    accumulateBoxedUnivariate(
                            (Double[]) instance,
                            instanceIndex,
                            moments[0]
                    );

            case PRIMITIVE_MULTIVARIATE ->
                    accumulatePrimitiveMultivariate(
                            (double[][]) instance,
                            instanceIndex,
                            shape.dimensionCount,
                            scope,
                            moments
                    );

            case BOXED_MULTIVARIATE ->
                    accumulateBoxedMultivariate(
                            (Double[][]) instance,
                            instanceIndex,
                            shape.dimensionCount,
                            scope,
                            moments
                    );
        }
    }

    private static void accumulatePrimitiveUnivariate(
            double[] series,
            int instanceIndex,
            OnlineMoments moments
    ) {
        if (series.length == 0) {
            throw new IllegalArgumentException(
                    "Training instance "
                            + instanceIndex
                            + " is an empty univariate series."
            );
        }

        for (int timeIndex = 0;
             timeIndex < series.length;
             timeIndex++) {

            addPrimitiveValue(
                    series[timeIndex],
                    instanceIndex,
                    0,
                    timeIndex,
                    moments
            );
        }
    }

    private static void accumulateBoxedUnivariate(
            Double[] series,
            int instanceIndex,
            OnlineMoments moments
    ) {
        if (series.length == 0) {
            throw new IllegalArgumentException(
                    "Training instance "
                            + instanceIndex
                            + " is an empty univariate series."
            );
        }

        for (int timeIndex = 0;
             timeIndex < series.length;
             timeIndex++) {

            addBoxedValue(
                    series[timeIndex],
                    instanceIndex,
                    0,
                    timeIndex,
                    moments
            );
        }
    }

    private static void accumulatePrimitiveMultivariate(
            double[][] series,
            int instanceIndex,
            int expectedDimensionCount,
            StandardizationScope scope,
            OnlineMoments[] moments
    ) {
        validateMultivariateDimensionCount(
                series.length,
                expectedDimensionCount,
                instanceIndex
        );

        for (int dimensionIndex = 0;
             dimensionIndex < series.length;
             dimensionIndex++) {

            double[] dimension =
                    series[dimensionIndex];

            if (dimension == null) {
                throw new IllegalArgumentException(
                        "Training instance "
                                + instanceIndex
                                + " contains a null primitive dimension at "
                                + "index "
                                + dimensionIndex
                                + "."
                );
            }

            if (dimension.length == 0) {
                throw new IllegalArgumentException(
                        "Training instance "
                                + instanceIndex
                                + ", dimension "
                                + dimensionIndex
                                + " contains no time points."
                );
            }

            OnlineMoments accumulator =
                    accumulatorForDimension(
                            scope,
                            moments,
                            dimensionIndex
                    );

            for (int timeIndex = 0;
                 timeIndex < dimension.length;
                 timeIndex++) {

                addPrimitiveValue(
                        dimension[timeIndex],
                        instanceIndex,
                        dimensionIndex,
                        timeIndex,
                        accumulator
                );
            }
        }
    }

    private static void accumulateBoxedMultivariate(
            Double[][] series,
            int instanceIndex,
            int expectedDimensionCount,
            StandardizationScope scope,
            OnlineMoments[] moments
    ) {
        validateMultivariateDimensionCount(
                series.length,
                expectedDimensionCount,
                instanceIndex
        );

        for (int dimensionIndex = 0;
             dimensionIndex < series.length;
             dimensionIndex++) {

            Double[] dimension =
                    series[dimensionIndex];

            if (dimension == null) {
                throw new IllegalArgumentException(
                        "Training instance "
                                + instanceIndex
                                + " contains a null boxed dimension at index "
                                + dimensionIndex
                                + "."
                );
            }

            if (dimension.length == 0) {
                throw new IllegalArgumentException(
                        "Training instance "
                                + instanceIndex
                                + ", dimension "
                                + dimensionIndex
                                + " contains no time points."
                );
            }

            OnlineMoments accumulator =
                    accumulatorForDimension(
                            scope,
                            moments,
                            dimensionIndex
                    );

            for (int timeIndex = 0;
                 timeIndex < dimension.length;
                 timeIndex++) {

                addBoxedValue(
                        dimension[timeIndex],
                        instanceIndex,
                        dimensionIndex,
                        timeIndex,
                        accumulator
                );
            }
        }
    }

    private static OnlineMoments accumulatorForDimension(
            StandardizationScope scope,
            OnlineMoments[] moments,
            int dimensionIndex
    ) {
        return switch (scope) {
            case GLOBAL ->
                    moments[0];

            case PER_DIMENSION ->
                    moments[dimensionIndex];

            case PER_SERIES,
                    PER_SERIES_PER_DIMENSION ->
                    throw new UnsupportedOperationException(
                            "Scope "
                                    + scope
                                    + " is not supported by the Phase 1 "
                                    + "training-statistics fitter."
                    );
        };
    }

    /**
     * Primitive NaN values are treated as missing and skipped. Infinite
     * values are rejected because they cannot produce meaningful moments.
     */
    private static void addPrimitiveValue(
            double value,
            int instanceIndex,
            int dimensionIndex,
            int timeIndex,
            OnlineMoments moments
    ) {
        if (Double.isNaN(value)) {
            return;
        }

        if (Double.isInfinite(value)) {
            throw nonFiniteValueException(
                    value,
                    instanceIndex,
                    dimensionIndex,
                    timeIndex
            );
        }

        moments.add(value);
    }

    /**
     * Boxed null and NaN values are treated as missing and skipped.
     * Infinite values are rejected.
     */
    private static void addBoxedValue(
            Double value,
            int instanceIndex,
            int dimensionIndex,
            int timeIndex,
            OnlineMoments moments
    ) {
        if (value == null || Double.isNaN(value)) {
            return;
        }

        if (Double.isInfinite(value)) {
            throw nonFiniteValueException(
                    value,
                    instanceIndex,
                    dimensionIndex,
                    timeIndex
            );
        }

        moments.add(value);
    }

    private static IllegalArgumentException nonFiniteValueException(
            double value,
            int instanceIndex,
            int dimensionIndex,
            int timeIndex
    ) {
        return new IllegalArgumentException(
                "Encountered infinite numeric value "
                        + value
                        + " at training instance "
                        + instanceIndex
                        + ", dimension "
                        + dimensionIndex
                        + ", time index "
                        + timeIndex
                        + ". Infinite values cannot be used to fit "
                        + "standardization statistics."
        );
    }

    private static void validateMultivariateDimensionCount(
            int actualDimensionCount,
            int expectedDimensionCount,
            int instanceIndex
    ) {
        if (actualDimensionCount != expectedDimensionCount) {
            throw new IllegalArgumentException(
                    "Inconsistent dimension count at training instance "
                            + instanceIndex
                            + ". Expected "
                            + expectedDimensionCount
                            + " dimensions but found "
                            + actualDimensionCount
                            + "."
            );
        }
    }

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

        for (int groupIndex = 0;
             groupIndex < moments.length;
             groupIndex++) {

            OnlineMoments accumulator =
                    moments[groupIndex];

            if (!accumulator.hasObservations()) {
                throw new IllegalArgumentException(
                        "Standardization statistic group "
                                + groupIndex
                                + " contains no finite observations."
                );
            }

            counts[groupIndex] =
                    accumulator.getCount();

            centers[groupIndex] =
                    accumulator.getMean();

            scales[groupIndex] =
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
     * Returns a usable positive scale.
     *
     * A constant group or a group with insufficient observations for sample
     * variance receives scale 1.0. This causes centered values from that
     * group to transform to zero without division by zero.
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

        if (!Double.isFinite(standardDeviation)) {
            throw new IllegalArgumentException(
                    "Fitted standard deviation is not finite: "
                            + standardDeviation
            );
        }

        if (standardDeviation == 0.0) {
            return CONSTANT_SCALE;
        }

        return standardDeviation;
    }

    private static List<String> validateAndCopyFeatureNames(
            List<String> featureNames,
            int dimensionCount,
            StandardizationScope scope
    ) {
        if (featureNames == null || featureNames.isEmpty()) {
            return Collections.emptyList();
        }

        List<String> copied =
                new ArrayList<>(
                        featureNames.size()
                );

        for (int featureIndex = 0;
             featureIndex < featureNames.size();
             featureIndex++) {

            String featureName =
                    featureNames.get(featureIndex);

            if (featureName == null
                    || featureName.isBlank()) {

                throw new IllegalArgumentException(
                        "Feature name at index "
                                + featureIndex
                                + " cannot be null or blank."
                );
            }

            copied.add(
                    featureName.trim()
            );
        }

        /*
         * Even GLOBAL statistics benefit from retaining the complete ordered
         * feature list as schema metadata.
         */
        if (copied.size() != dimensionCount) {
            throw new IllegalArgumentException(
                    "The training data contains "
                            + dimensionCount
                            + " dimension(s), but "
                            + copied.size()
                            + " feature name(s) were supplied for "
                            + scope
                            + " standardization."
            );
        }

        return Collections.unmodifiableList(
                copied
        );
    }

    private enum NumericRepresentation {

        PRIMITIVE_UNIVARIATE,

        BOXED_UNIVARIATE,

        PRIMITIVE_MULTIVARIATE,

        BOXED_MULTIVARIATE
    }

    private static final class DatasetShape {

        private final NumericRepresentation representation;
        private final int dimensionCount;

        private DatasetShape(
                NumericRepresentation representation,
                int dimensionCount
        ) {
            this.representation =
                    Objects.requireNonNull(
                            representation
                    );

            this.dimensionCount =
                    dimensionCount;
        }
    }
}