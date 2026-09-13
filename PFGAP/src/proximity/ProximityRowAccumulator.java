package proximity;

/**
 * Mutable sink for accumulating one directed proximity-matrix row.
 *
 * <p>This abstraction allows a proximity measure to emit only target indices
 * that can receive nonzero contributions. It therefore supports sparse,
 * leaf-oriented, or source-row algorithms without requiring a scan over every
 * matrix column.</p>
 *
 * <p>Implementations may use a primitive hash table, sorted primitive arrays,
 * dense scratch storage, compressed sparse storage, or a streaming sink. A
 * proximity measure must not assume a particular representation.</p>
 *
 * <p>An accumulator instance belongs to one source row and one active task. It
 * is not required to be thread-safe. Different workers must use different
 * accumulator instances.</p>
 *
 * <p>Target indices are zero-based. Contributions to the same target may be
 * added repeatedly and must be combined by the implementation. Implementations
 * must preserve full {@code double} precision and must not apply sparsity
 * thresholds, normalization, or diagonal policies unless explicitly required
 * by their own documented contract.</p>
 */
public interface ProximityRowAccumulator {

    /**
     * Adds a contribution to one target entry.
     *
     * <p>If the target already has an accumulated value, the contribution is
     * added to that value. A zero contribution may be ignored. Non-finite
     * contributions must be rejected.</p>
     *
     * @param targetIndex zero-based target column index
     * @param contribution finite contribution to add
     * @throws IndexOutOfBoundsException when the target index is invalid
     * @throws IllegalArgumentException when the contribution is not finite
     */
    void add(
            int targetIndex,
            double contribution
    );

    /**
     * Replaces the accumulated value of one target entry.
     *
     * <p>This operation supports explicit diagonal handling and formulas that
     * finalize a target value after accumulation. Setting zero may remove the
     * target from a sparse implementation.</p>
     *
     * @param targetIndex zero-based target column index
     * @param value finite replacement value
     * @throws IndexOutOfBoundsException when the target index is invalid
     * @throws IllegalArgumentException when the value is not finite
     */
    void set(
            int targetIndex,
            double value
    );

    /**
     * Returns the currently accumulated value for a target.
     *
     * <p>An untouched target returns zero.</p>
     *
     * @param targetIndex zero-based target column index
     * @return current accumulated value
     * @throws IndexOutOfBoundsException when the target index is invalid
     */
    double get(
            int targetIndex
    );

    /**
     * Returns the number of valid target columns in this row.
     */
    int targetCount();

    /**
     * Returns the number of explicitly retained entries.
     *
     * <p>For a sparse implementation, this is normally the number of currently
     * stored nonzero targets. For a dense implementation, it may equal
     * {@link #targetCount()} even when some values are zero.</p>
     */
    int retainedEntryCount();

    /**
     * Multiplies all retained values by a finite scalar.
     *
     * <p>This is intended for row-level normalization after contribution
     * accumulation. For example, an RF-GAP row may accumulate contributions
     * from all trees in the source-specific set and then scale once by the
     * reciprocal of that set's size.</p>
     *
     * @param factor finite scaling factor
     * @throws IllegalArgumentException when the factor is not finite
     */
    void scale(
            double factor
    );

    /**
     * Removes all accumulated values while retaining reusable storage when
     * possible.
     *
     * <p>This method permits one worker-local accumulator to be reused across
     * multiple source rows without repeated large allocations.</p>
     */
    void clear();

    /**
     * Validates a target index for default or implementing methods.
     *
     * @throws IndexOutOfBoundsException when the target index is outside
     *                                   {@code [0, targetCount())}
     */
    default void checkTargetIndex(
            int targetIndex
    ) {
        int count =
                targetCount();

        if (targetIndex < 0 || targetIndex >= count) {
            throw new IndexOutOfBoundsException(
                    "Target index "
                            + targetIndex
                            + " is outside [0, "
                            + count
                            + ")."
            );
        }
    }

    /**
     * Validates a value that will be stored or accumulated.
     *
     * @throws IllegalArgumentException when the value is NaN or infinite
     */
    default void checkFinite(
            double value,
            String valueName
    ) {
        if (!Double.isFinite(value)) {
            throw new IllegalArgumentException(
                    valueName
                            + " must be finite, but received "
                            + value
                            + "."
            );
        }
    }
}
