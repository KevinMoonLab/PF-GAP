package preprocessing.standardization;

import java.io.Serial;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * Immutable, serializable statistics used to standardize numeric data.
 *
 * The statistics are expressed generically as centers and scales:
 *
 *     standardizedValue =
 *         (value - center) / scale
 *
 * For z-score standardization:
 *
 *     center = mean
 *     scale  = standard deviation
 *
 * The number of statistic groups depends on the configured scope:
 *
 *     GLOBAL
 *         One count, center, and scale.
 *
 *     PER_DIMENSION
 *         One count, center, and scale per dimension.
 *
 * PER_SERIES and PER_SERIES_PER_DIMENSION are not represented by this
 * training-set statistics class in the initial implementation. Those scopes
 * calculate statistics from each individual series during transformation.
 *
 * Constant dimensions and other zero-scale groups must be normalized by the
 * fitter before constructing this object. The standard Phase 1 policy is:
 *
 *     fitted scale = 0.0
 *         -> stored scale = 1.0
 *
 * This causes every value in a constant group to standardize to zero without
 * requiring special cases during transformation.
 */
public final class StandardizationStats implements Serializable {

    @Serial
    private static final long serialVersionUID = 1L;

    /**
     * Version of the serialized statistics schema.
     *
     * This is distinct from serialVersionUID. It can also be included in
     * future JSON representations to support explicit format validation.
     */
    public static final int CURRENT_FORMAT_VERSION = 1;
    // the following indicates that an externally-supplied statistics file did not
    // report the number of observations used to fit a statistic group.
    public static final long UNKNOWN_COUNT = -1L;

    private final int formatVersion;
    private final StandardizationMethod method;
    private final StandardizationScope scope;
    private final VarianceConvention varianceConvention;

    /**
     * Optional ordered feature names.
     *
     * For PER_DIMENSION statistics, the order should match the order of the
     * centers, scales, and counts arrays.
     *
     * For GLOBAL statistics, this list may contain every feature name even
     * though only one statistic group exists.
     */
    private final List<String> featureNames;

    /**
     * Number of accepted finite observations in each statistic group.
     */
    private final long[] counts;

    /**
     * Fitted center for each statistic group.
     */
    private final double[] centers;

    /**
     * Fitted scale for each statistic group.
     *
     * Every scale must be finite and strictly positive.
     */
    private final double[] scales;

    /**
     * Constructs validated fitted standardization statistics.
     *
     * @param method mathematical standardization method
     * @param scope statistics-fitting scope
     * @param varianceConvention variance denominator convention
     * @param featureNames optional ordered feature names
     * @param counts number of accepted observations per statistic group
     * @param centers fitted centers per statistic group
     * @param scales fitted positive scales per statistic group
     */
    public StandardizationStats(
            StandardizationMethod method,
            StandardizationScope scope,
            VarianceConvention varianceConvention,
            List<String> featureNames,
            long[] counts,
            double[] centers,
            double[] scales
    ) {
        this(
                CURRENT_FORMAT_VERSION,
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
     * Constructs validated statistics with an explicit format version.
     *
     * The explicit version constructor is useful for future JSON loading and
     * model compatibility checks.
     *
     * @param formatVersion statistics representation version
     * @param method mathematical standardization method
     * @param scope statistics-fitting scope
     * @param varianceConvention variance denominator convention
     * @param featureNames optional ordered feature names
     * @param counts number of accepted observations per statistic group
     * @param centers fitted centers per statistic group
     * @param scales fitted positive scales per statistic group
     */
    public StandardizationStats(
            int formatVersion,
            StandardizationMethod method,
            StandardizationScope scope,
            VarianceConvention varianceConvention,
            List<String> featureNames,
            long[] counts,
            double[] centers,
            double[] scales
    ) {
        validateFormatVersion(formatVersion);

        this.method =
                Objects.requireNonNull(
                        method,
                        "StandardizationMethod cannot be null."
                );

        this.scope =
                Objects.requireNonNull(
                        scope,
                        "StandardizationScope cannot be null."
                );

        this.varianceConvention =
                Objects.requireNonNull(
                        varianceConvention,
                        "VarianceConvention cannot be null."
                );

        this.formatVersion =
                formatVersion;

        this.featureNames =
                copyAndValidateFeatureNames(
                        featureNames
                );

        this.counts =
                requireAndCopy(
                        counts,
                        "counts"
                );

        this.centers =
                requireAndCopy(
                        centers,
                        "centers"
                );

        this.scales =
                requireAndCopy(
                        scales,
                        "scales"
                );

        validateConfiguration();
        validateArrayLengths();
        validateCounts();
        validateCenters();
        validateScales();
        validateFeatureNames();
    }

    /**
     * Returns the statistics format version.
     *
     * @return format version
     */
    public int getFormatVersion() {
        return formatVersion;
    }

    /**
     * Returns the fitted standardization method.
     *
     * @return standardization method
     */
    public StandardizationMethod getMethod() {
        return method;
    }

    /**
     * Returns the fitted standardization scope.
     *
     * @return standardization scope
     */
    public StandardizationScope getScope() {
        return scope;
    }

    /**
     * Returns the fitted variance convention.
     *
     * The convention is primarily meaningful for z-score scaling.
     *
     * @return variance convention
     */
    public VarianceConvention getVarianceConvention() {
        return varianceConvention;
    }

    /**
     * Returns an immutable view of the ordered feature names.
     *
     * @return immutable feature-name list
     */
    public List<String> getFeatureNames() {
        return featureNames;
    }

    /**
     * Returns a defensive copy of observation counts.
     *
     * @return copied observation counts
     */
    public long[] getCounts() {
        return counts.clone();
    }

    /**
     * Returns a defensive copy of fitted centers.
     *
     * @return copied centers
     */
    public double[] getCenters() {
        return centers.clone();
    }

    /**
     * Returns a defensive copy of fitted scales.
     *
     * @return copied scales
     */
    public double[] getScales() {
        return scales.clone();
    }

    /**
     * Returns the number of fitted statistic groups.
     *
     * GLOBAL statistics contain one group. PER_DIMENSION statistics contain
     * one group per dimension.
     *
     * @return statistic-group count
     */
    public int getStatisticGroupCount() {
        return centers.length;
    }

    /**
     * Returns the observation count for one statistic group.
     *
     * @param groupIndex zero-based statistic-group index
     * @return observation count
     */
    public long getCount(
            int groupIndex
    ) {
        validateGroupIndex(groupIndex);

        return counts[groupIndex];
    }

    public boolean hasKnownCount(
            int groupIndex
    ) {
        return getCount(groupIndex)
                != UNKNOWN_COUNT;
    }

    /**
     * Returns the fitted center for one statistic group.
     *
     * @param groupIndex zero-based statistic-group index
     * @return fitted center
     */
    public double getCenter(
            int groupIndex
    ) {
        validateGroupIndex(groupIndex);

        return centers[groupIndex];
    }

    /**
     * Returns the fitted scale for one statistic group.
     *
     * @param groupIndex zero-based statistic-group index
     * @return fitted scale
     */
    public double getScale(
            int groupIndex
    ) {
        validateGroupIndex(groupIndex);

        return scales[groupIndex];
    }

    /**
     * Returns the statistic-group index that should be used for a particular
     * dimension.
     *
     * GLOBAL always maps every dimension to group zero.
     *
     * PER_DIMENSION maps dimension d to group d.
     *
     * @param dimensionIndex zero-based data-dimension index
     * @return statistic-group index
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
            case GLOBAL ->
                    0;

            case PER_DIMENSION -> {
                if (dimensionIndex >= getStatisticGroupCount()) {
                    throw new IndexOutOfBoundsException(
                            "Dimension index "
                                    + dimensionIndex
                                    + " is outside the fitted range [0, "
                                    + (getStatisticGroupCount() - 1)
                                    + "]."
                    );
                }

                yield dimensionIndex;
            }

            case PER_SERIES,
                    PER_SERIES_PER_DIMENSION ->
                    throw new UnsupportedOperationException(
                            "Scope "
                                    + scope
                                    + " does not use reusable fitted "
                                    + "training-statistic groups."
                    );
        };
    }

    /**
     * Returns the center applicable to one dimension.
     *
     * @param dimensionIndex zero-based data-dimension index
     * @return fitted center
     */
    public double getCenterForDimension(
            int dimensionIndex
    ) {
        return getCenter(
                getGroupIndexForDimension(
                        dimensionIndex
                )
        );
    }

    /**
     * Returns the scale applicable to one dimension.
     *
     * @param dimensionIndex zero-based data-dimension index
     * @return fitted scale
     */
    public double getScaleForDimension(
            int dimensionIndex
    ) {
        return getScale(
                getGroupIndexForDimension(
                        dimensionIndex
                )
        );
    }

    /**
     * Returns the observation count applicable to one dimension.
     *
     * @param dimensionIndex zero-based data-dimension index
     * @return fitted observation count
     */
    public long getCountForDimension(
            int dimensionIndex
    ) {
        return getCount(
                getGroupIndexForDimension(
                        dimensionIndex
                )
        );
    }

    /**
     * Standardizes one finite numeric value using a specified statistic
     * group.
     *
     * @param value numeric value
     * @param groupIndex statistic-group index
     * @return standardized value
     */
    public double standardize(
            double value,
            int groupIndex
    ) {
        if (!Double.isFinite(value)) {
            throw new IllegalArgumentException(
                    "Cannot standardize a non-finite value: "
                            + value
            );
        }

        validateGroupIndex(groupIndex);

        return switch (method) {
            case NONE ->
                    value;

            case Z_SCORE,
                    MIN_MAX,
                    ROBUST ->
                    (value - centers[groupIndex])
                            / scales[groupIndex];
        };
    }

    /**
     * Standardizes one finite numeric value using the statistic group
     * associated with a dimension.
     *
     * @param value numeric value
     * @param dimensionIndex zero-based dimension index
     * @return standardized value
     */
    public double standardizeDimensionValue(
            double value,
            int dimensionIndex
    ) {
        return standardize(
                value,
                getGroupIndexForDimension(
                        dimensionIndex
                )
        );
    }

    /**
     * Returns whether the statistics include ordered feature names.
     *
     * @return true when feature names are available
     */
    public boolean hasFeatureNames() {
        return !featureNames.isEmpty();
    }

    /**
     * Validates that a supplied ordered feature list is compatible with
     * these fitted statistics.
     *
     * If this object has no feature names, only the expected number of
     * dimensions is validated for PER_DIMENSION statistics.
     *
     * @param suppliedFeatureNames selected feature names in reader order
     */
    public void validateFeatureCompatibility(
            List<String> suppliedFeatureNames
    ) {
        if (suppliedFeatureNames == null) {
            if (hasFeatureNames()) {
                throw new IllegalArgumentException(
                        "Standardization statistics contain feature names, "
                                + "but no feature names were supplied for "
                                + "compatibility validation."
                );
            }

            return;
        }

        List<String> supplied =
                copyAndValidateFeatureNames(
                        suppliedFeatureNames
                );

        if (scope == StandardizationScope.PER_DIMENSION
                && supplied.size() != getStatisticGroupCount()) {

            throw new IllegalArgumentException(
                    "PER_DIMENSION statistics contain "
                            + getStatisticGroupCount()
                            + " groups, but "
                            + supplied.size()
                            + " supplied features were found."
            );
        }

        if (!hasFeatureNames()) {
            return;
        }

        if (!featureNames.equals(supplied)) {
            throw new IllegalArgumentException(
                    "Feature names or feature order do not match the "
                            + "fitted standardization statistics. Expected "
                            + featureNames
                            + " but received "
                            + supplied
                            + "."
            );
        }
    }

    private void validateConfiguration() {
        method.requireImplemented();
        scope.requireImplemented();

        if (method == StandardizationMethod.NONE) {
            throw new IllegalArgumentException(
                    "StandardizationStats should not be constructed for "
                            + "StandardizationMethod.NONE because no fitted "
                            + "statistics are required."
            );
        }
    }

    private void validateArrayLengths() {
        if (counts.length == 0) {
            throw new IllegalArgumentException(
                    "Standardization statistics must contain at least one "
                            + "statistic group."
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

        if (scope == StandardizationScope.GLOBAL
                && counts.length != 1) {

            throw new IllegalArgumentException(
                    "GLOBAL standardization requires exactly one "
                            + "statistic group, but received "
                            + counts.length
                            + "."
            );
        }
    }

    private void validateCounts() {
        for (int index = 0;
             index < counts.length;
             index++) {

            long count =
                    counts[index];

            if (count == UNKNOWN_COUNT) {
                continue;
            }

            if (count <= 0L) {
                throw new IllegalArgumentException(
                        "Observation count must be positive or "
                                + "StandardizationStats.UNKNOWN_COUNT for "
                                + "statistic group "
                                + index
                                + ", but received "
                                + count
                                + "."
                );
            }
        }
    }

    private void validateCenters() {
        for (int index = 0;
             index < centers.length;
             index++) {

            if (!Double.isFinite(centers[index])) {
                throw new IllegalArgumentException(
                        "Center must be finite for statistic group "
                                + index
                                + ", but received "
                                + centers[index]
                                + "."
                );
            }
        }
    }

    private void validateScales() {
        for (int index = 0;
             index < scales.length;
             index++) {

            double scale =
                    scales[index];

            if (!Double.isFinite(scale)) {
                throw new IllegalArgumentException(
                        "Scale must be finite for statistic group "
                                + index
                                + ", but received "
                                + scale
                                + "."
                );
            }

            if (scale <= 0.0) {
                throw new IllegalArgumentException(
                        "Scale must be strictly positive for statistic "
                                + "group "
                                + index
                                + ", but received "
                                + scale
                                + ". Constant groups should use scale=1.0."
                );
            }
        }
    }

    private void validateFeatureNames() {
        if (featureNames.isEmpty()) {
            return;
        }

        if (scope == StandardizationScope.PER_DIMENSION
                && featureNames.size() != getStatisticGroupCount()) {

            throw new IllegalArgumentException(
                    "PER_DIMENSION statistics contain "
                            + getStatisticGroupCount()
                            + " statistic groups, but "
                            + featureNames.size()
                            + " feature names were supplied."
            );
        }
    }

    private void validateGroupIndex(
            int groupIndex
    ) {
        if (groupIndex < 0
                || groupIndex >= getStatisticGroupCount()) {

            throw new IndexOutOfBoundsException(
                    "Statistic-group index "
                            + groupIndex
                            + " is outside the valid range [0, "
                            + (getStatisticGroupCount() - 1)
                            + "]."
            );
        }
    }

    private static void validateFormatVersion(
            int formatVersion
    ) {
        if (formatVersion <= 0) {
            throw new IllegalArgumentException(
                    "Standardization statistics format version must be "
                            + "positive, but received: "
                            + formatVersion
            );
        }

        if (formatVersion > CURRENT_FORMAT_VERSION) {
            throw new IllegalArgumentException(
                    "Unsupported standardization statistics format version: "
                            + formatVersion
                            + ". This version of PFGAP supports versions up "
                            + "to "
                            + CURRENT_FORMAT_VERSION
                            + "."
            );
        }
    }

    private static List<String> copyAndValidateFeatureNames(
            List<String> names
    ) {
        if (names == null || names.isEmpty()) {
            return Collections.emptyList();
        }

        List<String> copied =
                new ArrayList<>(names.size());

        for (int index = 0;
             index < names.size();
             index++) {

            String name =
                    names.get(index);

            if (name == null || name.isBlank()) {
                throw new IllegalArgumentException(
                        "Feature name at index "
                                + index
                                + " cannot be null or blank."
                );
            }

            copied.add(
                    name.trim()
            );
        }

        return Collections.unmodifiableList(
                copied
        );
    }

    private static long[] requireAndCopy(
            long[] values,
            String fieldName
    ) {
        if (values == null) {
            throw new IllegalArgumentException(
                    fieldName
                            + " cannot be null."
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
                    fieldName
                            + " cannot be null."
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

        if (!(other instanceof StandardizationStats that)) {
            return false;
        }

        return formatVersion == that.formatVersion
                && method == that.method
                && scope == that.scope
                && varianceConvention == that.varianceConvention
                && featureNames.equals(that.featureNames)
                && Arrays.equals(counts, that.counts)
                && Arrays.equals(centers, that.centers)
                && Arrays.equals(scales, that.scales);
    }

    @Override
    public int hashCode() {
        int result =
                Objects.hash(
                        formatVersion,
                        method,
                        scope,
                        varianceConvention,
                        featureNames
                );

        result =
                31 * result
                        + Arrays.hashCode(counts);

        result =
                31 * result
                        + Arrays.hashCode(centers);

        result =
                31 * result
                        + Arrays.hashCode(scales);

        return result;
    }

    @Override
    public String toString() {
        return "StandardizationStats{"
                + "formatVersion=" + formatVersion
                + ", method=" + method
                + ", scope=" + scope
                + ", varianceConvention=" + varianceConvention
                + ", featureNames=" + featureNames
                + ", counts=" + Arrays.toString(counts)
                + ", centers=" + Arrays.toString(centers)
                + ", scales=" + Arrays.toString(scales)
                + '}';
    }
}