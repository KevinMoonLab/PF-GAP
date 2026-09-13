package proximity;

import java.util.Arrays;

/**
 * Immutable compressed sparse row representation of a proximity matrix.
 *
 * <p>The matrix stores only explicitly retained entries:</p>
 *
 * <pre>
 * rowOffsets[row] ... rowOffsets[row + 1] - 1
 * </pre>
 *
 * <p>identify the segment of {@code columnIndices} and {@code values} belonging
 * to one row. Column indices within every row must be strictly increasing.
 * Missing entries have value zero.</p>
 *
 * <p>This initial implementation uses ordinary Java primitive arrays and thus
 * supports at most {@link Integer#MAX_VALUE} retained entries. Construction
 * code should track totals with {@code long} and fail before narrowing when
 * that limit would be exceeded. The public access API does not require callers
 * to depend on the backing-array layout, leaving room for a segmented CSR
 * implementation later.</p>
 *
 * <p>The package-private ownership constructor does not copy its arrays. The
 * builder transferring those arrays must never mutate them afterward. Public
 * array-export methods return defensive copies.</p>
 */
public final class CompressedSparseProximityMatrix {

    private final int rowCount;
    private final int columnCount;
    private final int[] rowOffsets;
    private final int[] columnIndices;
    private final double[] values;

    /**
     * Creates a validated CSR matrix and assumes ownership of its arrays.
     *
     * @param rowCount number of matrix rows
     * @param columnCount number of matrix columns
     * @param rowOffsets CSR row offsets of length {@code rowCount + 1}
     * @param columnIndices sorted column indices for retained entries
     * @param values retained finite proximity values
     */
    CompressedSparseProximityMatrix(
            int rowCount,
            int columnCount,
            int[] rowOffsets,
            int[] columnIndices,
            double[] values
    ) {
        validateDimensions(rowCount, columnCount);
        validateStorage(
                rowCount,
                columnCount,
                rowOffsets,
                columnIndices,
                values
        );

        this.rowCount = rowCount;
        this.columnCount = columnCount;
        this.rowOffsets = rowOffsets;
        this.columnIndices = columnIndices;
        this.values = values;
    }

    /**
     * Creates an empty sparse matrix with the requested dimensions.
     */
    public static CompressedSparseProximityMatrix empty(
            int rowCount,
            int columnCount
    ) {
        validateDimensions(rowCount, columnCount);

        return new CompressedSparseProximityMatrix(
                rowCount,
                columnCount,
                new int[rowCount + 1],
                new int[0],
                new double[0]
        );
    }

    /** Returns the number of matrix rows. */
    public int rowCount() {
        return rowCount;
    }

    /** Returns the number of matrix columns. */
    public int columnCount() {
        return columnCount;
    }

    /** Returns the number of explicitly retained entries. */
    public int nonZeroCount() {
        return values.length;
    }

    /**
     * Returns the retained-entry count as a long for representation-neutral
     * client code.
     */
    public long nonZeroCountLong() {
        return values.length;
    }

    /**
     * Returns the number of retained entries in one row.
     */
    public int rowEntryCount(
            int rowIndex
    ) {
        checkRowIndex(rowIndex);
        return rowOffsets[rowIndex + 1] - rowOffsets[rowIndex];
    }

    /**
     * Returns one matrix value, or zero when the entry is not retained.
     *
     * <p>Lookup uses binary search because each row's columns are strictly
     * increasing.</p>
     */
    public double get(
            int rowIndex,
            int columnIndex
    ) {
        checkRowIndex(rowIndex);
        checkColumnIndex(columnIndex);

        int position = findPosition(rowIndex, columnIndex);
        return position >= 0 ? values[position] : 0.0;
    }

    /**
     * Returns whether an entry is explicitly retained.
     *
     * <p>This differs from {@code get(row, column) != 0.0} only if a producer
     * deliberately retains exact zeros. The standard matrix builder should not
     * retain them.</p>
     */
    public boolean contains(
            int rowIndex,
            int columnIndex
    ) {
        checkRowIndex(rowIndex);
        checkColumnIndex(columnIndex);
        return findPosition(rowIndex, columnIndex) >= 0;
    }

    /**
     * Returns the column index at a row-local retained-entry offset.
     */
    public int columnIndexAt(
            int rowIndex,
            int rowEntryOffset
    ) {
        return columnIndices[absolutePosition(rowIndex, rowEntryOffset)];
    }

    /**
     * Returns the value at a row-local retained-entry offset.
     */
    public double valueAt(
            int rowIndex,
            int rowEntryOffset
    ) {
        return values[absolutePosition(rowIndex, rowEntryOffset)];
    }

    /**
     * Iterates one sparse row without allocation.
     *
     * <p>The consumer is invoked in ascending column order.</p>
     */
    public void forEachInRow(
            int rowIndex,
            EntryConsumer consumer
    ) {
        checkRowIndex(rowIndex);
        if (consumer == null) {
            throw new IllegalArgumentException("EntryConsumer cannot be null.");
        }

        int start = rowOffsets[rowIndex];
        int end = rowOffsets[rowIndex + 1];

        for (int position = start; position < end; position++) {
            consumer.accept(
                    columnIndices[position],
                    values[position]
            );
        }
    }

    /**
     * Computes the sum of explicitly retained values in one row.
     */
    public double rowSum(
            int rowIndex
    ) {
        checkRowIndex(rowIndex);

        double sum = 0.0;
        int start = rowOffsets[rowIndex];
        int end = rowOffsets[rowIndex + 1];

        for (int position = start; position < end; position++) {
            sum += values[position];
        }

        return sum;
    }

    /** Returns a defensive copy of the CSR row offsets. */
    public int[] copyRowOffsets() {
        return rowOffsets.clone();
    }

    /** Returns a defensive copy of retained column indices. */
    public int[] copyColumnIndices() {
        return columnIndices.clone();
    }

    /** Returns a defensive copy of retained values. */
    public double[] copyValues() {
        return values.clone();
    }

    /**
     * Returns the absolute retained-entry start offset of one row.
     *
     * <p>Package-private for matrix builders and compatibility adapters in the
     * proximity package.</p>
     */
    int rowStartOffset(
            int rowIndex
    ) {
        checkRowIndex(rowIndex);
        return rowOffsets[rowIndex];
    }

    /**
     * Returns the absolute retained-entry end offset of one row.
     *
     * <p>Package-private for matrix builders and compatibility adapters.</p>
     */
    int rowEndOffset(
            int rowIndex
    ) {
        checkRowIndex(rowIndex);
        return rowOffsets[rowIndex + 1];
    }

    /** Package-private allocation-free absolute column access. */
    int columnIndexAtAbsolutePosition(
            int position
    ) {
        checkAbsolutePosition(position);
        return columnIndices[position];
    }

    /** Package-private allocation-free absolute value access. */
    double valueAtAbsolutePosition(
            int position
    ) {
        checkAbsolutePosition(position);
        return values[position];
    }

    private int findPosition(
            int rowIndex,
            int columnIndex
    ) {
        return Arrays.binarySearch(
                columnIndices,
                rowOffsets[rowIndex],
                rowOffsets[rowIndex + 1],
                columnIndex
        );
    }

    private int absolutePosition(
            int rowIndex,
            int rowEntryOffset
    ) {
        checkRowIndex(rowIndex);

        int count = rowOffsets[rowIndex + 1] - rowOffsets[rowIndex];
        if (rowEntryOffset < 0 || rowEntryOffset >= count) {
            throw new IndexOutOfBoundsException(
                    "Row entry offset " + rowEntryOffset
                            + " is outside [0, " + count + ") for row "
                            + rowIndex + "."
            );
        }

        return rowOffsets[rowIndex] + rowEntryOffset;
    }

    private void checkRowIndex(
            int rowIndex
    ) {
        if (rowIndex < 0 || rowIndex >= rowCount) {
            throw new IndexOutOfBoundsException(
                    "rowIndex " + rowIndex + " is outside [0, "
                            + rowCount + ")."
            );
        }
    }

    private void checkColumnIndex(
            int columnIndex
    ) {
        if (columnIndex < 0 || columnIndex >= columnCount) {
            throw new IndexOutOfBoundsException(
                    "columnIndex " + columnIndex + " is outside [0, "
                            + columnCount + ")."
            );
        }
    }

    private void checkAbsolutePosition(
            int position
    ) {
        if (position < 0 || position >= values.length) {
            throw new IndexOutOfBoundsException(
                    "Retained-entry position " + position
                            + " is outside [0, " + values.length + ")."
            );
        }
    }

    private static void validateDimensions(
            int rowCount,
            int columnCount
    ) {
        if (rowCount < 0 || columnCount < 0) {
            throw new IllegalArgumentException(
                    "Matrix dimensions cannot be negative. Received "
                            + rowCount + " x " + columnCount + "."
            );
        }
    }

    private static void validateStorage(
            int rowCount,
            int columnCount,
            int[] rowOffsets,
            int[] columnIndices,
            double[] values
    ) {
        if (rowOffsets == null
                || columnIndices == null
                || values == null) {

            throw new IllegalArgumentException(
                    "CSR storage arrays cannot be null."
            );
        }

        if (rowOffsets.length != rowCount + 1) {
            throw new IllegalArgumentException(
                    "rowOffsets must have length rowCount + 1. Expected "
                            + (rowCount + 1) + " but received "
                            + rowOffsets.length + "."
            );
        }

        if (columnIndices.length != values.length) {
            throw new IllegalArgumentException(
                    "columnIndices and values must have equal length."
            );
        }

        if (rowOffsets[0] != 0) {
            throw new IllegalArgumentException(
                    "The first CSR row offset must be zero."
            );
        }

        if (rowOffsets[rowCount] != values.length) {
            throw new IllegalArgumentException(
                    "The final CSR row offset must equal the retained-entry "
                            + "count."
            );
        }

        int priorOffset = 0;
        for (int rowIndex = 0; rowIndex < rowCount; rowIndex++) {
            int start = rowOffsets[rowIndex];
            int end = rowOffsets[rowIndex + 1];

            if (start < priorOffset || end < start || end > values.length) {
                throw new IllegalArgumentException(
                        "CSR row offsets are invalid at row " + rowIndex + "."
                );
            }

            int priorColumn = -1;
            for (int position = start; position < end; position++) {
                int columnIndex = columnIndices[position];
                double value = values[position];

                if (columnIndex < 0 || columnIndex >= columnCount) {
                    throw new IllegalArgumentException(
                            "Column index " + columnIndex
                                    + " is invalid at retained position "
                                    + position + "."
                    );
                }

                if (columnIndex <= priorColumn) {
                    throw new IllegalArgumentException(
                            "Column indices must be strictly increasing within "
                                    + "row " + rowIndex + "."
                    );
                }

                if (!Double.isFinite(value)) {
                    throw new IllegalArgumentException(
                            "Non-finite proximity value " + value
                                    + " at retained position " + position + "."
                    );
                }

                priorColumn = columnIndex;
            }

            priorOffset = end;
        }
    }

    /**
     * Allocation-free consumer for one retained matrix entry.
     */
    @FunctionalInterface
    public interface EntryConsumer {

        /**
         * Accepts one retained entry.
         *
         * @param columnIndex zero-based column index
         * @param value retained finite value
         */
        void accept(
                int columnIndex,
                double value
        );
    }

    @Override
    public String toString() {
        return "CompressedSparseProximityMatrix{"
                + "rows=" + rowCount
                + ", columns=" + columnCount
                + ", retainedEntries=" + values.length
                + '}';
    }
}
