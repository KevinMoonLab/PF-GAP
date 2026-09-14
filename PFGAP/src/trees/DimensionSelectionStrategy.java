package trees;

/**
 * Strategy used to determine how many realized dimensions are randomly
 * selected at each proximity-tree split.
 *
 * <p>A realized dimension means:</p>
 *
 * <ul>
 *    <li>
 *       One tabular feature position for {@code double[]} or {@code Double[]}
 *       instances.
 *    </li>
 *    <li>
 *       One multivariate channel for {@code double[][]} or {@code Double[][]}
 *       instances.
 *    </li>
 * </ul>
 *
 * <p>The selected subset is generated once per tree node and is shared by
 * every candidate split evaluated at that node.</p>
 *
 * <p>Univariate time-point subsampling is not represented by this enum.
 * That is a separate potential feature because randomly selecting time
 * points changes the temporal structure used by time-series distances.</p>
 */
public enum DimensionSelectionStrategy {

    /**
     * Uses every available realized dimension.
     *
     * <p>No explicit selected-index array needs to be stored for this strategy.
     * A null selection can represent the all-dimensions fast path.</p>
     */
    ALL,

    /**
     * Selects approximately the square root of the available dimensions.
     *
     * <p>The intended count is:</p>
     *
     * <pre>
     * ceil(sqrt(d))
     * </pre>
     *
     * <p>where {@code d} is the number of available realized dimensions.</p>
     */
    SQRT,

    /**
     * Selects a logarithmic number of available dimensions.
     *
     * <p>The intended count is:</p>
     *
     * <pre>
     * floor(log2(d)) + 1
     * </pre>
     *
     * <p>where {@code d} is the number of available realized dimensions.</p>
     */
    LOG2,

    /**
     * Selects a configured fixed number of dimensions.
     *
     * <p>The requested count must be positive. Counts greater than the
     * available dimensionality are clamped to the available dimensionality.</p>
     */
    FIXED_COUNT,

    /**
     * Selects a configured proportion of the available dimensions.
     *
     * <p>The configured proportion must be greater than zero and no greater
     * than one. The intended count is:</p>
     *
     * <pre>
     * ceil(proportion * d)
     * </pre>
     *
     * <p>The result is clamped to the range {@code [1, d]}.</p>
     */
    PROPORTION
}