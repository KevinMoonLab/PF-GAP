package imputation.results;

import datasets.NumericStorageType;
import output.MatrixMarketWriter;

import java.io.IOException;
import java.io.Serial;
import java.io.Serializable;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Objects;

/**
 * Primitive CSR result containing only values at originally missing positions.
 *
 * <p>Rows correspond to dataset instances. For one-dimensional instances,
 * columns are time/feature positions. For dimension-major two-dimensional
 * instances, columns use the global flattening rule:</p>
 *
 * <pre>
 * column = dimension * maximumTimeLength + position
 * </pre>
 *
 * <p>This global rule supports unequal instance lengths and unequal dimension
 * counts while producing one valid rectangular matrix. Actual source shape is
 * retained by {@code instanceDimensionOffsets} and {@code dimensionLengths}.
 * Exact zero values are retained because CSR presence means the source cell was
 * originally missing, not merely numerically nonzero.</p>
 */
public final class ImputedValuesCSR implements Serializable {

    @Serial
    private static final long serialVersionUID = 1L;

    private final int rowCount;
    private final int columnCount;
    private final int[] rowOffsets;
    private final int[] columnIndices;
    private final Object values;
    private final NumericStorageType storageType;
    private final boolean twoDimensionalSource;
    private final int maximumDimensionCount;
    private final int maximumTimeLength;
    private final int[] instanceDimensionOffsets;
    private final int[] dimensionLengths;

    private ImputedValuesCSR(
            int rowCount,
            int columnCount,
            int[] rowOffsets,
            int[] columnIndices,
            Object values,
            NumericStorageType storageType,
            boolean twoDimensionalSource,
            int maximumDimensionCount,
            int maximumTimeLength,
            int[] instanceDimensionOffsets,
            int[] dimensionLengths
    ) {
        this.rowCount = rowCount;
        this.columnCount = columnCount;
        this.rowOffsets = rowOffsets.clone();
        this.columnIndices = columnIndices.clone();
        this.values = cloneValues(values, storageType);
        this.storageType = storageType;
        this.twoDimensionalSource = twoDimensionalSource;
        this.maximumDimensionCount = maximumDimensionCount;
        this.maximumTimeLength = maximumTimeLength;
        this.instanceDimensionOffsets = instanceDimensionOffsets.clone();
        this.dimensionLengths = dimensionLengths.clone();
        validate();
    }

    public static ImputedValuesCSR ofDouble(
            int rowCount,
            int columnCount,
            int[] rowOffsets,
            int[] columnIndices,
            double[] values,
            boolean twoDimensionalSource,
            int maximumDimensionCount,
            int maximumTimeLength,
            int[] instanceDimensionOffsets,
            int[] dimensionLengths
    ) {
        return new ImputedValuesCSR(
                rowCount, columnCount, rowOffsets, columnIndices, values,
                NumericStorageType.FLOAT64, twoDimensionalSource,
                maximumDimensionCount, maximumTimeLength,
                instanceDimensionOffsets, dimensionLengths
        );
    }

    public static ImputedValuesCSR ofFloat(
            int rowCount,
            int columnCount,
            int[] rowOffsets,
            int[] columnIndices,
            float[] values,
            boolean twoDimensionalSource,
            int maximumDimensionCount,
            int maximumTimeLength,
            int[] instanceDimensionOffsets,
            int[] dimensionLengths
    ) {
        return new ImputedValuesCSR(
                rowCount, columnCount, rowOffsets, columnIndices, values,
                NumericStorageType.FLOAT32, twoDimensionalSource,
                maximumDimensionCount, maximumTimeLength,
                instanceDimensionOffsets, dimensionLengths
        );
    }

    public int rowCount() { return rowCount; }
    public int columnCount() { return columnCount; }
    public int entryCount() { return columnIndices.length; }
    public long entryCountLong() { return columnIndices.length; }
    public NumericStorageType storageType() { return storageType; }
    public boolean isTwoDimensionalSource() { return twoDimensionalSource; }
    public int maximumDimensionCount() { return maximumDimensionCount; }
    public int maximumTimeLength() { return maximumTimeLength; }
    public int rowStart(int row) { checkRow(row); return rowOffsets[row]; }
    public int rowEnd(int row) { checkRow(row); return rowOffsets[row + 1]; }
    public int rowEntryCount(int row) { return rowEnd(row) - rowStart(row); }
    public int columnIndexAtOffset(int offset) { return columnIndices[offset]; }

    public double valueAtOffset(int offset) {
        if (storageType == NumericStorageType.FLOAT32) {
            return ((float[]) values)[offset];
        }
        return ((double[]) values)[offset];
    }

    public int dimensionCount(int instance) {
        checkRow(instance);
        return instanceDimensionOffsets[instance + 1]
                - instanceDimensionOffsets[instance];
    }

    public int dimensionLength(int instance, int dimension) {
        checkRow(instance);
        int start = instanceDimensionOffsets[instance];
        int count = instanceDimensionOffsets[instance + 1] - start;
        if (dimension < 0 || dimension >= count) {
            throw new IndexOutOfBoundsException(
                    "Dimension " + dimension + " outside [0, " + count + ")."
            );
        }
        return dimensionLengths[start + dimension];
    }

    public int flattenColumn(int dimension, int position) {
        if (dimension < 0 || dimension >= maximumDimensionCount
                || position < 0 || position >= maximumTimeLength) {
            throw new IndexOutOfBoundsException(
                    "Flattened coordinate outside declared source envelope."
            );
        }
        return Math.addExact(
                Math.multiplyExact(dimension, maximumTimeLength),
                position
        );
    }

    public int[] copyRowOffsets() { return rowOffsets.clone(); }
    public int[] copyColumnIndices() { return columnIndices.clone(); }
    public int[] copyInstanceDimensionOffsets() {
        return instanceDimensionOffsets.clone();
    }
    public int[] copyDimensionLengths() { return dimensionLengths.clone(); }

    public double[] copyDoubleValues() {
        if (storageType != NumericStorageType.FLOAT64) {
            throw new IllegalStateException("CSR values are FLOAT32.");
        }
        return ((double[]) values).clone();
    }

    public float[] copyFloatValues() {
        if (storageType != NumericStorageType.FLOAT32) {
            throw new IllegalStateException("CSR values are FLOAT64.");
        }
        return ((float[]) values).clone();
    }

    /** Streams this CSR directly to Matrix Market coordinate format. */
    public Path writeMatrixMarket(
            Path path,
            String description
    ) throws IOException {
        return MatrixMarketWriter.writeCoordinate(
                path,
                rowCount,
                columnCount,
                entryCountLong(),
                description,
                sink -> {
                    for (int row = 0; row < rowCount; row++) {
                        for (int offset = rowOffsets[row];
                             offset < rowOffsets[row + 1];
                             offset++) {
                            sink.write(
                                    row,
                                    columnIndices[offset],
                                    valueAtOffset(offset)
                            );
                        }
                    }
                }
        );
    }

    private void validate() {
        if (rowCount < 0 || columnCount < 0
                || maximumDimensionCount < 0 || maximumTimeLength < 0) {
            throw new IllegalArgumentException("CSR dimensions cannot be negative.");
        }
        if (rowOffsets.length != rowCount + 1 || rowOffsets[0] != 0
                || rowOffsets[rowCount] != columnIndices.length) {
            throw new IllegalArgumentException("Invalid CSR row offsets.");
        }
        int valueCount = storageType == NumericStorageType.FLOAT32
                ? ((float[]) values).length
                : ((double[]) values).length;
        if (valueCount != columnIndices.length) {
            throw new IllegalArgumentException(
                    "CSR value and column-index counts differ."
            );
        }
        if (instanceDimensionOffsets.length != rowCount + 1
                || instanceDimensionOffsets[0] != 0
                || instanceDimensionOffsets[rowCount] != dimensionLengths.length) {
            throw new IllegalArgumentException("Invalid source-shape offsets.");
        }
        for (int row = 0; row < rowCount; row++) {
            if (rowOffsets[row] > rowOffsets[row + 1]) {
                throw new IllegalArgumentException("CSR row offsets decrease.");
            }
            int previous = -1;
            for (int offset = rowOffsets[row];
                 offset < rowOffsets[row + 1];
                 offset++) {
                int column = columnIndices[offset];
                if (column < 0 || column >= columnCount || column <= previous) {
                    throw new IllegalArgumentException(
                            "CSR columns must be strictly increasing per row."
                    );
                }
                if (!Double.isFinite(valueAtOffset(offset))) {
                    throw new IllegalArgumentException(
                            "Imputed CSR values must be finite."
                    );
                }
                previous = column;
            }
        }
    }

    private void checkRow(int row) {
        if (row < 0 || row >= rowCount) {
            throw new IndexOutOfBoundsException(
                    "Row " + row + " outside [0, " + rowCount + ")."
            );
        }
    }

    private static Object cloneValues(
            Object values,
            NumericStorageType storageType
    ) {
        Objects.requireNonNull(storageType, "Storage type cannot be null.");
        if (storageType == NumericStorageType.FLOAT32) {
            return ((float[]) Objects.requireNonNull(values)).clone();
        }
        if (storageType == NumericStorageType.FLOAT64) {
            return ((double[]) Objects.requireNonNull(values)).clone();
        }
        throw new IllegalArgumentException("AUTO is not a materialized CSR dtype.");
    }

    @Override
    public String toString() {
        return "ImputedValuesCSR{"
                + "rows=" + rowCount
                + ", columns=" + columnCount
                + ", entries=" + entryCount()
                + ", storageType=" + storageType
                + ", twoDimensionalSource=" + twoDimensionalSource
                + ", maximumDimensionCount=" + maximumDimensionCount
                + ", maximumTimeLength=" + maximumTimeLength
                + '}';
    }
}
