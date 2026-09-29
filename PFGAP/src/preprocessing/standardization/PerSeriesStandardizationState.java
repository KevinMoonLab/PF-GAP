package preprocessing.standardization;

import java.io.Serial;
import java.io.Serializable;
import java.util.Arrays;
import java.util.Objects;

/**
 * Immutable affine transformation parameters calculated from one realized
 * numeric series.
 *
 * <p>This state is used only for
 * {@link StandardizationScope#PER_SERIES} and
 * {@link StandardizationScope#PER_SERIES_PER_DIMENSION}. Unlike
 * {@link StandardizationStats}, it does not represent reusable statistics
 * fitted across a training dataset. It records the exact local centers and
 * scales used to transform one series so the transformation can later be
 * inverted, including while an imputed dataset is streamed to an output
 * writer.</p>
 *
 * <p>The number of parameter groups depends on the scope:</p>
 *
 * <ul>
 *     <li>{@code PER_SERIES}: one group for the complete series.</li>
 *     <li>{@code PER_SERIES_PER_DIMENSION}: one group per realized
 *     dimension. For univariate input, this contains one group.</li>
 * </ul>
 *
 * <p>Centers and scales use the common affine representation:</p>
 *
 * <pre>
 * transformed = (value - center) / scale
 * original    = transformed * scale + center
 * </pre>
 *
 * <p>For mean centering, the scale is {@code 1.0}. Constant z-score and
 * min-max groups also use {@code 1.0}. Counts describe the accepted,
 * nonmissing observations used to calculate each group. Primitive NaN values
 * are not represented in this object and remain unchanged by transformation
 * kernels.</p>
 */
public final class PerSeriesStandardizationState
        implements Serializable {

    @Serial
    private static final long serialVersionUID = 1L;

    /**
     * Current serialized representation version.
     */
    public static final int CURRENT_FORMAT_VERSION = 1;

    private final int formatVersion;
    private final StandardizationMethod method;
    private final StandardizationScope scope;
    private final VarianceConvention varianceConvention;
    private final long[] counts;
    private final double[] centers;
    private final double[] scales;

    /**
     * Constructs validated local parameters for one realized series.
     *
     * @param method affine standardization method
     * @param scope per-series standardization scope
     * @param varianceConvention variance convention used for z-score
     * @param counts accepted observation counts per local group
     * @param centers local centers
     * @param scales finite, strictly positive local scales
     */
    public PerSeriesStandardizationState(
            StandardizationMethod method,
            StandardizationScope scope,
            VarianceConvention varianceConvention,
            long[] counts,
            double[] centers,
            double[] scales
    ) {
        this(
                CURRENT_FORMAT_VERSION,
                method,
                scope,
                varianceConvention,
                counts,
                centers,
                scales
        );
    }

    /**
     * Constructs local parameters with an explicit representation version.
     *
     * @param formatVersion serialized representation version
     * @param method affine standardization method
     * @param scope per-series standardization scope
     * @param varianceConvention variance convention used for z-score
     * @param counts accepted observation counts per local group
     * @param centers local centers
     * @param scales finite, strictly positive local scales
     */
    public PerSeriesStandardizationState(
            int formatVersion,
            StandardizationMethod method,
            StandardizationScope scope,
            VarianceConvention varianceConvention,
            long[] counts,
            double[] centers,
            double[] scales
    ) {
        validateFormatVersion(formatVersion);

        this.method = Objects.requireNonNull(
                method,
                "StandardizationMethod cannot be null."
        );
        this.scope = Objects.requireNonNull(
                scope,
                "StandardizationScope cannot be null."
        );
        this.varianceConvention = Objects.requireNonNull(
                varianceConvention,
                "VarianceConvention cannot be null."
        );
        this.formatVersion = formatVersion;
        this.counts = requireAndCopy(counts, "counts");
        this.centers = requireAndCopy(centers, "centers");
        this.scales = requireAndCopy(scales, "scales");

        validateConfiguration();
        validateArrayLengths();
        validateCounts();
        validateCenters();
        validateScales();
    }

    public int getFormatVersion() {
        return formatVersion;
    }

    public StandardizationMethod getMethod() {
        return method;
    }

    public StandardizationScope getScope() {
        return scope;
    }

    public VarianceConvention getVarianceConvention() {
        return varianceConvention;
    }

    public int getParameterGroupCount() {
        return centers.length;
    }

    public long[] getCounts() {
        return counts.clone();
    }

    public double[] getCenters() {
        return centers.clone();
    }

    public double[] getScales() {
        return scales.clone();
    }

    public long getCount(
            int groupIndex
    ) {
        validateGroupIndex(groupIndex);
        return counts[groupIndex];
    }

    public double getCenter(
            int groupIndex
    ) {
        validateGroupIndex(groupIndex);
        return centers[groupIndex];
    }

    public double getScale(
            int groupIndex
    ) {
        validateGroupIndex(groupIndex);
        return scales[groupIndex];
    }

    /**
     * Returns the local parameter group applicable to a realized dimension.
     *
     * @param dimensionIndex zero-based dimension index
     * @return zero for PER_SERIES, or the dimension index for
     *         PER_SERIES_PER_DIMENSION
     */
    public int getGroupIndexForDimension(
            int dimensionIndex
    ) {
        if (dimensionIndex < 0) {
            throw new IllegalArgumentException(
                    "Dimension index cannot be negative: "
                            + dimensionIndex
            );
        }

        return switch (scope) {
            case PER_SERIES ->
                    0;
            case PER_SERIES_PER_DIMENSION -> {
                if (dimensionIndex >= getParameterGroupCount()) {
                    throw new IndexOutOfBoundsException(
                            "Dimension index "
                                    + dimensionIndex
                                    + " is outside the local parameter range "
                                    + "[0, "
                                    + (getParameterGroupCount() - 1)
                                    + "]."
                    );
                }
                yield dimensionIndex;
            }
            case GLOBAL,
                    PER_DIMENSION ->
                    throw new IllegalStateException(
                            "PerSeriesStandardizationState cannot use scope "
                                    + scope
                                    + "."
                    );
        };
    }

    public long getCountForDimension(
            int dimensionIndex
    ) {
        return getCount(getGroupIndexForDimension(dimensionIndex));
    }

    public double getCenterForDimension(
            int dimensionIndex
    ) {
        return getCenter(getGroupIndexForDimension(dimensionIndex));
    }

    public double getScaleForDimension(
            int dimensionIndex
    ) {
        return getScale(getGroupIndexForDimension(dimensionIndex));
    }

    /**
     * Applies the checked scalar forward transformation for one finite value.
     * Hot primitive-array kernels should resolve parameters once per group
     * rather than invoking this method for every element.
     *
     * @param value finite original value
     * @param groupIndex local parameter-group index
     * @return transformed value
     */
    public double transform(
            double value,
            int groupIndex
    ) {
        requireFiniteValue(value, "transform");
        validateGroupIndex(groupIndex);
        return (value - centers[groupIndex]) / scales[groupIndex];
    }

    /**
     * Applies the checked scalar inverse transformation for one finite value.
     *
     * @param value finite transformed value
     * @param groupIndex local parameter-group index
     * @return value in the original coordinate system
     */
    public double inverseTransform(
            double value,
            int groupIndex
    ) {
        requireFiniteValue(value, "inverse-transform");
        validateGroupIndex(groupIndex);
        return value * scales[groupIndex] + centers[groupIndex];
    }

    public double transformDimensionValue(
            double value,
            int dimensionIndex
    ) {
        return transform(
                value,
                getGroupIndexForDimension(dimensionIndex)
        );
    }

    public double inverseTransformDimensionValue(
            double value,
            int dimensionIndex
    ) {
        return inverseTransform(
                value,
                getGroupIndexForDimension(dimensionIndex)
        );
    }

    private void validateConfiguration() {
        method.requireImplemented();
        scope.requireImplemented();

        if (method == StandardizationMethod.NONE) {
            throw new IllegalArgumentException(
                    "Per-series transformation state is not required for "
                            + "StandardizationMethod.NONE."
            );
        }
        if (!scope.usesPerSeriesStatistics()) {
            throw new IllegalArgumentException(
                    "PerSeriesStandardizationState requires PER_SERIES or "
                            + "PER_SERIES_PER_DIMENSION scope, but received "
                            + scope
                            + "."
            );
        }
    }

    private void validateArrayLengths() {
        if (counts.length == 0) {
            throw new IllegalArgumentException(
                    "Per-series transformation state must contain at least "
                            + "one parameter group."
            );
        }
        if (counts.length != centers.length
                || counts.length != scales.length) {
            throw new IllegalArgumentException(
                    "counts, centers, and scales must have identical "
                            + "lengths. Received counts="
                            + counts.length
                            + ", centers="
                            + centers.length
                            + ", scales="
                            + scales.length
                            + "."
            );
        }
        if (scope == StandardizationScope.PER_SERIES
                && counts.length != 1) {
            throw new IllegalArgumentException(
                    "PER_SERIES requires exactly one local parameter group, "
                            + "but received "
                            + counts.length
                            + "."
            );
        }
    }

    private void validateCounts() {
        for (int group = 0; group < counts.length; group++) {
            if (counts[group] <= 0L) {
                throw new IllegalArgumentException(
                        "Per-series observation count must be positive for "
                                + "parameter group "
                                + group
                                + ", but received "
                                + counts[group]
                                + "."
                );
            }
        }
    }

    private void validateCenters() {
        for (int group = 0; group < centers.length; group++) {
            if (!Double.isFinite(centers[group])) {
                throw new IllegalArgumentException(
                        "Per-series center must be finite for parameter group "
                                + group
                                + ", but received "
                                + centers[group]
                                + "."
                );
            }
        }
    }

    private void validateScales() {
        for (int group = 0; group < scales.length; group++) {
            double scale = scales[group];
            if (!Double.isFinite(scale) || scale <= 0.0) {
                throw new IllegalArgumentException(
                        "Per-series scale must be finite and strictly "
                                + "positive for parameter group "
                                + group
                                + ", but received "
                                + scale
                                + ". Constant groups should use scale=1.0."
                );
            }
        }
    }

    private void validateGroupIndex(
            int groupIndex
    ) {
        if (groupIndex < 0
                || groupIndex >= getParameterGroupCount()) {
            throw new IndexOutOfBoundsException(
                    "Local parameter-group index "
                            + groupIndex
                            + " is outside the valid range [0, "
                            + (getParameterGroupCount() - 1)
                            + "]."
            );
        }
    }

    private static void validateFormatVersion(
            int formatVersion
    ) {
        if (formatVersion <= 0) {
            throw new IllegalArgumentException(
                    "Per-series state format version must be positive, but "
                            + "received: "
                            + formatVersion
            );
        }
        if (formatVersion > CURRENT_FORMAT_VERSION) {
            throw new IllegalArgumentException(
                    "Unsupported per-series state format version: "
                            + formatVersion
                            + ". This version supports versions up to "
                            + CURRENT_FORMAT_VERSION
                            + "."
            );
        }
    }

    private static void requireFiniteValue(
            double value,
            String operation
    ) {
        if (!Double.isFinite(value)) {
            throw new IllegalArgumentException(
                    "Cannot "
                            + operation
                            + " a non-finite value: "
                            + value
            );
        }
    }

    private static long[] requireAndCopy(
            long[] values,
            String fieldName
    ) {
        if (values == null) {
            throw new IllegalArgumentException(
                    fieldName + " cannot be null."
            );
        }
        return values.clone();
    }

    private static double[] requireAndCopy(
            double[] values,
            String fieldName
    ) {
        if (values == null) {
            throw new IllegalArgumentException(
                    fieldName + " cannot be null."
            );
        }
        return values.clone();
    }

    @Override
    public boolean equals(
            Object other
    ) {
        if (this == other) {
            return true;
        }
        if (!(other instanceof PerSeriesStandardizationState that)) {
            return false;
        }
        return formatVersion == that.formatVersion
                && method == that.method
                && scope == that.scope
                && varianceConvention == that.varianceConvention
                && Arrays.equals(counts, that.counts)
                && Arrays.equals(centers, that.centers)
                && Arrays.equals(scales, that.scales);
    }

    @Override
    public int hashCode() {
        int result = Objects.hash(
                formatVersion,
                method,
                scope,
                varianceConvention
        );
        result = 31 * result + Arrays.hashCode(counts);
        result = 31 * result + Arrays.hashCode(centers);
        result = 31 * result + Arrays.hashCode(scales);
        return result;
    }

    @Override
    public String toString() {
        return "PerSeriesStandardizationState{"
                + "formatVersion=" + formatVersion
                + ", method=" + method
                + ", scope=" + scope
                + ", varianceConvention=" + varianceConvention
                + ", counts=" + Arrays.toString(counts)
                + ", centers=" + Arrays.toString(centers)
                + ", scales=" + Arrays.toString(scales)
                + '}';
    }
}
