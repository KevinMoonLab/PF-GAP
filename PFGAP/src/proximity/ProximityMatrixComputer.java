package proximity;

import core.parallel.ParallelRuntime;
import trees.ProximityForest;

import java.util.Arrays;
import java.util.Objects;
import java.util.OptionalDouble;

/**
 * Computes dense or compressed-sparse proximity matrices.
 *
 * <p>This class owns matrix-level scheduling. Formula implementations remain
 * stateless and do not create executors. Sequential and parallel execution use
 * the same row and cell calculations; only the caller-owned
 * {@link ParallelRuntime} determines whether row blocks execute concurrently.</p>
 *
 * <p>Dense train/train computation exploits explicitly declared symmetry and
 * constant diagonal values. Sparse computation prefers direct row accumulation
 * when supported by the selected measure; otherwise it falls back to scalar
 * target scans. Sparse rows are prepared in bounded row blocks rather than one
 * temporary object per matrix row.</p>
 */
public final class ProximityMatrixComputer {

    private static final int MINIMUM_DENSE_ROW_RANGE_SIZE = 4;
    private static final int SPARSE_ROWS_PER_BLOCK = 256;
    private static final int DEFAULT_EXPECTED_SPARSE_ROW_ENTRIES = 64;

    private ProximityMatrixComputer() {
    }

    /** Computes an exact dense train/train matrix. */
    public static double[][] computeTrainDense(
            ProximityForest forest,
            int trainingSize,
            ProximityMeasure measure,
            ParallelRuntime runtime
    ) throws Exception {
        validateCommon(forest, trainingSize, measure, runtime);

        ProximityContext context = ProximityContextBuilder.buildTrain(
                forest,
                trainingSize,
                measure,
                ProximityContextLayout.DENSE_INDEXED,
                runtime
        );

        double[][] matrix = new double[trainingSize][trainingSize];
        ProximityProperties properties = measure.properties();
        OptionalDouble constantDiagonal = properties.constantTrainDiagonal();

        runRowRanges(
                trainingSize,
                MINIMUM_DENSE_ROW_RANGE_SIZE,
                runtime,
                (start, end) -> {
                    for (int row = start; row < end; row++) {
                        int firstColumn = properties.trainSymmetric()
                                ? row
                                : 0;

                        for (int column = firstColumn;
                             column < trainingSize;
                             column++) {

                            double value;
                            if (row == column && constantDiagonal.isPresent()) {
                                value = constantDiagonal.getAsDouble();
                            } else {
                                value = measure.computeTrainTrain(
                                        row,
                                        column,
                                        context
                                );
                            }

                            requireFinite(value, row, column);
                            matrix[row][column] = value;

                            if (properties.trainSymmetric() && row != column) {
                                matrix[column][row] = value;
                            }
                        }
                    }
                }
        );

        return matrix;
    }

    /** Computes an exact dense test/train matrix. */
    public static double[][] computeTestTrainDense(
            ProximityForest forest,
            int testingSize,
            int trainingSize,
            ProximityMeasure measure,
            ParallelRuntime runtime
    ) throws Exception {
        validateCommon(forest, trainingSize, measure, runtime);
        if (testingSize < 0) {
            throw new IllegalArgumentException("testingSize cannot be negative.");
        }

        ProximityContext context = ProximityContextBuilder.buildTestTrain(
                forest,
                trainingSize,
                testingSize,
                measure,
                ProximityContextLayout.DENSE_INDEXED,
                runtime
        );

        double[][] matrix = new double[testingSize][trainingSize];

        runRowRanges(
                testingSize,
                MINIMUM_DENSE_ROW_RANGE_SIZE,
                runtime,
                (start, end) -> {
                    for (int row = start; row < end; row++) {
                        for (int column = 0;
                             column < trainingSize;
                             column++) {

                            double value = measure.computeTestTrain(
                                    row,
                                    column,
                                    context
                            );

                            requireFinite(value, row, column);
                            matrix[row][column] = value;
                        }
                    }
                }
        );

        return matrix;
    }

    /**
     * Computes a train/train matrix in compressed sparse row form.
     *
     * @param retentionThreshold retain values whose absolute value is strictly
     *                           greater than this nonnegative threshold
     */
    public static CompressedSparseProximityMatrix computeTrainSparse(
            ProximityForest forest,
            int trainingSize,
            ProximityMeasure measure,
            double retentionThreshold,
            ParallelRuntime runtime
    ) throws Exception {
        validateCommon(forest, trainingSize, measure, runtime);
        validateThreshold(retentionThreshold);
        warnIfStructurallyDense(measure);

        ProximityContext context = ProximityContextBuilder.buildTrain(
                forest,
                trainingSize,
                measure,
                ProximityContextLayout.SPARSE_TREE,
                runtime
        );

        return computeSparse(
                trainingSize,
                trainingSize,
                retentionThreshold,
                runtime,
                (row, accumulator) -> {
                    if (measure.supportsTrainRowAccumulation()) {
                        measure.accumulateTrainRow(row, context, accumulator);
                    } else {
                        OptionalDouble diagonal =
                                measure.properties().constantTrainDiagonal();

                        for (int column = 0;
                             column < trainingSize;
                             column++) {

                            double value = row == column && diagonal.isPresent()
                                    ? diagonal.getAsDouble()
                                    : measure.computeTrainTrain(
                                            row,
                                            column,
                                            context
                                    );

                            requireFinite(value, row, column);
                            accumulator.set(column, value);
                        }
                    }
                }
        );
    }

    /**
     * Computes a test/train matrix in compressed sparse row form.
     *
     * @param retentionThreshold retain values whose absolute value is strictly
     *                           greater than this nonnegative threshold
     */
    public static CompressedSparseProximityMatrix computeTestTrainSparse(
            ProximityForest forest,
            int testingSize,
            int trainingSize,
            ProximityMeasure measure,
            double retentionThreshold,
            ParallelRuntime runtime
    ) throws Exception {
        validateCommon(forest, trainingSize, measure, runtime);
        if (testingSize < 0) {
            throw new IllegalArgumentException("testingSize cannot be negative.");
        }
        validateThreshold(retentionThreshold);
        warnIfStructurallyDense(measure);

        ProximityContext context = ProximityContextBuilder.buildTestTrain(
                forest,
                trainingSize,
                testingSize,
                measure,
                ProximityContextLayout.SPARSE_TREE,
                runtime
        );

        return computeSparse(
                testingSize,
                trainingSize,
                retentionThreshold,
                runtime,
                (row, accumulator) -> {
                    if (measure.supportsTestTrainRowAccumulation()) {
                        measure.accumulateTestTrainRow(row, context, accumulator);
                    } else {
                        for (int column = 0;
                             column < trainingSize;
                             column++) {

                            double value = measure.computeTestTrain(
                                    row,
                                    column,
                                    context
                            );

                            requireFinite(value, row, column);
                            accumulator.set(column, value);
                        }
                    }
                }
        );
    }

    private static CompressedSparseProximityMatrix computeSparse(
            int rowCount,
            int columnCount,
            double retentionThreshold,
            ParallelRuntime runtime,
            SparseRowComputer rowComputer
    ) throws Exception {
        if (rowCount == 0 || columnCount == 0) {
            return CompressedSparseProximityMatrix.empty(
                    rowCount,
                    columnCount
            );
        }

        int blockCount = divideCeiling(rowCount, SPARSE_ROWS_PER_BLOCK);
        SparseBlock[] blocks = new SparseBlock[blockCount];

        runIndexRange(
                blockCount,
                runtime,
                blockIndex -> {
                    int firstRow = blockIndex * SPARSE_ROWS_PER_BLOCK;
                    int endRow = Math.min(
                            rowCount,
                            firstRow + SPARSE_ROWS_PER_BLOCK
                    );

                    blocks[blockIndex] = buildSparseBlock(
                            firstRow,
                            endRow,
                            columnCount,
                            retentionThreshold,
                            rowComputer
                    );
                }
        );

        long totalEntries = 0L;
        for (SparseBlock block : blocks) {
            if (block == null) {
                throw new IllegalStateException(
                        "Sparse proximity block was not produced."
                );
            }
            totalEntries += block.values.length;
            if (totalEntries > Integer.MAX_VALUE) {
                throw new IllegalStateException(
                        "Sparse proximity matrix contains more than "
                                + Integer.MAX_VALUE
                                + " retained entries. Segmented CSR storage "
                                + "is required for this result."
                );
            }
        }

        int[] rowOffsets = new int[rowCount + 1];
        int[] columnIndices = new int[(int) totalEntries];
        double[] values = new double[(int) totalEntries];

        int outputPosition = 0;
        for (SparseBlock block : blocks) {
            for (int localRow = 0;
                 localRow < block.rowCount();
                 localRow++) {

                int globalRow = block.firstRow + localRow;
                int localStart = block.rowOffsets[localRow];
                int localEnd = block.rowOffsets[localRow + 1];
                int count = localEnd - localStart;

                System.arraycopy(
                        block.columnIndices,
                        localStart,
                        columnIndices,
                        outputPosition,
                        count
                );
                System.arraycopy(
                        block.values,
                        localStart,
                        values,
                        outputPosition,
                        count
                );

                outputPosition += count;
                rowOffsets[globalRow + 1] = outputPosition;
            }
        }

        return new CompressedSparseProximityMatrix(
                rowCount,
                columnCount,
                rowOffsets,
                columnIndices,
                values
        );
    }

    private static SparseBlock buildSparseBlock(
            int firstRow,
            int endRow,
            int columnCount,
            double retentionThreshold,
            SparseRowComputer rowComputer
    ) throws Exception {
        int rowCount = endRow - firstRow;
        int[] rowOffsets = new int[rowCount + 1];
        IntBuffer columns = new IntBuffer();
        DoubleBuffer values = new DoubleBuffer();

        PrimitiveProximityRowAccumulator accumulator =
                new PrimitiveProximityRowAccumulator(
                        columnCount,
                        Math.min(
                                columnCount,
                                DEFAULT_EXPECTED_SPARSE_ROW_ENTRIES
                        )
                );

        for (int row = firstRow; row < endRow; row++) {
            accumulator.clear();
            rowComputer.compute(row, accumulator);

            int retainedCount = accumulator.retainedEntryCount();
            int[] rowColumns = new int[retainedCount];
            double[] rowValues = new double[retainedCount];
            int accepted = 0;

            for (int entry = 0; entry < retainedCount; entry++) {
                int column = accumulator.retainedTargetAt(entry);
                double value = accumulator.retainedValueAt(entry);
                requireFinite(value, row, column);

                if (Math.abs(value) > retentionThreshold) {
                    rowColumns[accepted] = column;
                    rowValues[accepted] = value;
                    accepted++;
                }
            }

            sortByColumn(rowColumns, rowValues, accepted);
            columns.append(rowColumns, accepted);
            values.append(rowValues, accepted);
            rowOffsets[row - firstRow + 1] = columns.size;
        }

        return new SparseBlock(
                firstRow,
                rowOffsets,
                columns.toArray(),
                values.toArray()
        );
    }

    private static void sortByColumn(
            int[] columns,
            double[] values,
            int length
    ) {
        if (length < 2) {
            return;
        }
        quickSort(columns, values, 0, length - 1);
    }

    private static void quickSort(
            int[] columns,
            double[] values,
            int low,
            int high
    ) {
        int left = low;
        int right = high;
        int pivot = columns[(low + high) >>> 1];

        while (left <= right) {
            while (columns[left] < pivot) {
                left++;
            }
            while (columns[right] > pivot) {
                right--;
            }
            if (left <= right) {
                int column = columns[left];
                columns[left] = columns[right];
                columns[right] = column;

                double value = values[left];
                values[left] = values[right];
                values[right] = value;

                left++;
                right--;
            }
        }

        if (low < right) {
            quickSort(columns, values, low, right);
        }
        if (left < high) {
            quickSort(columns, values, left, high);
        }
    }

    private static void runRowRanges(
            int rowCount,
            int minimumRangeSize,
            ParallelRuntime runtime,
            RowRangeAction action
    ) throws Exception {
        if (rowCount == 0) {
            return;
        }
        if (runtime.isParallel() && rowCount >= minimumRangeSize) {
            runtime.forRanges(
                    0,
                    rowCount,
                    minimumRangeSize,
                    action::run
            );
            return;
        }
        action.run(0, rowCount);
    }

    private static void runIndexRange(
            int count,
            ParallelRuntime runtime,
            IndexAction action
    ) throws Exception {
        if (count == 0) {
            return;
        }
        if (runtime.isParallel() && count > 1) {
            runtime.forRange(
                    0,
                    count,
                    1,
                    action::run
            );
            return;
        }
        for (int index = 0; index < count; index++) {
            action.run(index);
        }
    }

    private static void validateCommon(
            ProximityForest forest,
            int trainingSize,
            ProximityMeasure measure,
            ParallelRuntime runtime
    ) {
        Objects.requireNonNull(forest, "ProximityForest cannot be null.");
        Objects.requireNonNull(measure, "ProximityMeasure cannot be null.");
        Objects.requireNonNull(runtime, "ParallelRuntime cannot be null.");

        if (trainingSize < 0) {
            throw new IllegalArgumentException("trainingSize cannot be negative.");
        }

        measure.validateContract();
    }

    private static void validateThreshold(
            double retentionThreshold
    ) {
        if (!Double.isFinite(retentionThreshold)
                || retentionThreshold < 0.0) {

            throw new IllegalArgumentException(
                    "retentionThreshold must be finite and nonnegative."
            );
        }
    }

    private static void requireFinite(
            double value,
            int row,
            int column
    ) {
        if (!Double.isFinite(value)) {
            throw new IllegalStateException(
                    "Proximity formula produced non-finite value "
                            + value + " at row " + row
                            + ", column " + column + "."
            );
        }
    }

    private static void warnIfStructurallyDense(
            ProximityMeasure measure
    ) {
        if ("DEPTH_WEIGHTED".equals(measure.id())) {
            System.err.println(
                    "Warning: exact DEPTH_WEIGHTED proximity is generally "
                            + "structurally dense. Sparse construction remains "
                            + "enabled, but may retain nearly every matrix entry."
            );
        }
    }

    private static int divideCeiling(
            int value,
            int divisor
    ) {
        return value / divisor + (value % divisor == 0 ? 0 : 1);
    }

    @FunctionalInterface
    private interface RowRangeAction {
        void run(int startInclusive, int endExclusive) throws Exception;
    }

    @FunctionalInterface
    private interface IndexAction {
        void run(int index) throws Exception;
    }

    @FunctionalInterface
    private interface SparseRowComputer {
        void compute(
                int rowIndex,
                PrimitiveProximityRowAccumulator accumulator
        ) throws Exception;
    }

    private record SparseBlock(
            int firstRow,
            int[] rowOffsets,
            int[] columnIndices,
            double[] values
    ) {
        private int rowCount() {
            return rowOffsets.length - 1;
        }
    }

    private static final class IntBuffer {
        private int[] values = new int[256];
        private int size;

        private void append(int[] source, int length) {
            ensureCapacity(size + length);
            System.arraycopy(source, 0, values, size, length);
            size += length;
        }

        private void ensureCapacity(int required) {
            if (required <= values.length) {
                return;
            }
            int capacity = values.length;
            while (capacity < required) {
                int grown = capacity + (capacity >>> 1) + 1;
                if (grown < 0 || grown > Integer.MAX_VALUE - 8) {
                    capacity = required;
                    break;
                }
                capacity = grown;
            }
            values = Arrays.copyOf(values, capacity);
        }

        private int[] toArray() {
            return Arrays.copyOf(values, size);
        }
    }

    private static final class DoubleBuffer {
        private double[] values = new double[256];
        private int size;

        private void append(double[] source, int length) {
            ensureCapacity(size + length);
            System.arraycopy(source, 0, values, size, length);
            size += length;
        }

        private void ensureCapacity(int required) {
            if (required <= values.length) {
                return;
            }
            int capacity = values.length;
            while (capacity < required) {
                int grown = capacity + (capacity >>> 1) + 1;
                if (grown < 0 || grown > Integer.MAX_VALUE - 8) {
                    capacity = required;
                    break;
                }
                capacity = grown;
            }
            values = Arrays.copyOf(values, capacity);
        }

        private double[] toArray() {
            return Arrays.copyOf(values, size);
        }
    }
}
