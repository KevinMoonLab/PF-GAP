package datasets;

/**
 * Controls the primitive storage type produced by numeric dataset readers.
 *
 * <p>This setting describes an input policy rather than the result type of
 * distances, statistics, proximities, predictions, or scores. Those values
 * may continue to use {@code double} regardless of feature storage.</p>
 *
 * <p>Numeric readers must produce only primitive observations:</p>
 *
 * <ul>
 *     <li>{@code float[]} or {@code float[][]} for {@link #FLOAT32}</li>
 *     <li>{@code double[]} or {@code double[][]} for {@link #FLOAT64}</li>
 * </ul>
 *
 * <p>Numeric missing values are represented by {@link Float#NaN} or
 * {@link Double#NaN}. Boxed numeric arrays are not supported.</p>
 */
public enum NumericStorageType {

    /**
     * Lets the reader choose the storage type.
     *
     * <p>Readers for typed binary formats should preserve a supported source
     * type when practical. Readers for untyped text formats should default to
     * {@link #FLOAT64} unless documented otherwise.</p>
     */
    AUTO,

    /**
     * Stores numeric observations in primitive single-precision arrays.
     */
    FLOAT32,

    /**
     * Stores numeric observations in primitive double-precision arrays.
     */
    FLOAT64;

    /**
     * Returns whether this value explicitly selects a primitive storage type.
     *
     * @return {@code true} for {@link #FLOAT32} or {@link #FLOAT64};
     *         {@code false} for {@link #AUTO}
     */
    public boolean isExplicit() {
        return this != AUTO;
    }
}
