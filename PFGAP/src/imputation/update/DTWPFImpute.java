package imputation.update;

import core.parallel.ParallelRuntime;
import datasets.ListObjectDataset;
import distance.elastic.DTWWithPath;
import distance.missing.DTWAROW;
import distance.missing.DTWAROW_D;
import distance.multiTS.DTW_D;
import imputation.util.MissingIndices;
import proximity.CompressedSparseProximityMatrix;
import proximity.ProximityMatrixResult;
import util.Pair;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/**
 * Performs DTW-aligned proximity-weighted numeric imputation.
 *
 * <p>For a missing value at target time {@code t}, each proximity neighbor
 * contributes the average of its observed values at times aligned to
 * {@code t}. That neighbor-level average is weighted once by the proximity,
 * and the result is normalized over neighbors that provide a usable aligned
 * value.</p>
 *
 * <p>This implementation consumes an explicit {@link ProximityMatrixResult},
 * traverses dense or CSR rows directly, and schedules independent target-row
 * updates through the caller-owned {@link ParallelRuntime}. It creates no
 * executors, uses no parallel streams, reads no global proximity state, and
 * performs no dense-to-map conversion or epsilon-based sparsification.</p>
 *
 * <p>Alignment paths are scoped to one target row and one update pass. Each
 * target-neighbor path is computed once, reused for all missing values in that
 * target, and released when the row is complete. This avoids the previous
 * static global cache and prevents path state from leaking across datasets,
 * repetitions, or concurrent operations.</p>
 *
 * <p>Numeric updates use Jacobi-style semantics and preserve primitive feature
 * storage. Supported observations are {@code double[]}, {@code float[]},
 * {@code double[][]}, and {@code float[][]}. Weighted calculations use double
 * precision; float results are narrowed once when written.</p>
 */
public final class DTWPFImpute {

    private static final int MINIMUM_PARALLEL_ROW_RANGE_SIZE = 16;

    private DTWPFImpute() {
    }

    /** Updates missing training values using DTW-aligned train/train weights. */
    public static void trainNumericImpute(
            ListObjectDataset data,
            ProximityMatrixResult proximities,
            ParallelRuntime runtime,
            int windowSize
    ) throws Exception {
        Objects.requireNonNull(data, "Training data cannot be null.");
        validateRuntimeAndMatrix(runtime, proximities);

        List<Object> rawData = data.getData();
        MissingIndices missing = data.getMissingIndices();

        if (rawData == null || rawData.isEmpty() || missing == null) {
            return;
        }

        validateShape(
                proximities,
                rawData.size(),
                rawData.size(),
                "training"
        );

        NeighborRows neighbors = NeighborRows.from(proximities);

        if (missing.is2D()) {
            impute2D(
                    data,
                    rawData,
                    rawData,
                    missing,
                    missing,
                    neighbors,
                    true,
                    runtime,
                    windowSize
            );
        } else {
            impute1D(
                    data,
                    rawData,
                    rawData,
                    missing,
                    missing,
                    neighbors,
                    true,
                    runtime,
                    windowSize
            );
        }
    }

    /** Updates missing test values using DTW-aligned test/train weights. */
    public static void testNumericImpute(
            ListObjectDataset testData,
            ListObjectDataset trainData,
            ProximityMatrixResult proximities,
            ParallelRuntime runtime,
            int windowSize
    ) throws Exception {
        Objects.requireNonNull(testData, "Testing data cannot be null.");
        Objects.requireNonNull(trainData, "Training data cannot be null.");
        validateRuntimeAndMatrix(runtime, proximities);

        List<Object> testRaw = testData.getData();
        MissingIndices testMissing = testData.getMissingIndices();

        if (testRaw == null || testRaw.isEmpty() || testMissing == null) {
            return;
        }

        List<Object> trainRaw = Objects.requireNonNull(
                trainData.getData(),
                "Training data storage cannot be null."
        );

        validateShape(
                proximities,
                testRaw.size(),
                trainRaw.size(),
                "test/train"
        );

        NeighborRows neighbors = NeighborRows.from(proximities);
        MissingIndices trainMissing = trainData.getMissingIndices();

        if (testMissing.is2D()) {
            impute2D(
                    testData,
                    testRaw,
                    trainRaw,
                    testMissing,
                    trainMissing,
                    neighbors,
                    false,
                    runtime,
                    windowSize
            );
        } else {
            impute1D(
                    testData,
                    testRaw,
                    trainRaw,
                    testMissing,
                    trainMissing,
                    neighbors,
                    false,
                    runtime,
                    windowSize
            );
        }
    }

    private static void impute1D(
            ListObjectDataset dataToUpdate,
            List<Object> targetData,
            List<Object> neighborData,
            MissingIndices targetMissing,
            MissingIndices neighborMissing,
            NeighborRows neighborRows,
            boolean excludeSelf,
            ParallelRuntime runtime,
            int windowSize
    ) throws Exception {
        MissingIndices retainedMissing = dataToUpdate.getMissingIndices();
        double[] fallbackMeans = computeObservedMeans1D(
                neighborData,
                neighborMissing
        );
        Object[] updated = new Object[targetData.size()];

        forRanges(targetData.size(), runtime, (start, end) -> {
            for (int target = start; target < end; target++) {
                Object targetSeries = targetData.get(target);
                Object replacement = copySeries1D(targetSeries);
                int missingStart = targetMissing.start1D(target);
                int missingEnd = targetMissing.end1D(target);

                List<AlignedNeighbor> alignedNeighbors = buildAlignedNeighbors(
                        target,
                        targetSeries,
                        neighborData,
                        neighborRows,
                        excludeSelf,
                        false,
                        windowSize
                );

                for (int missingOffset = missingStart;
                        missingOffset < missingEnd;
                        missingOffset++) {
                    int targetTime = targetMissing.positionAt(missingOffset);
                    requireIndex(length1D(replacement), targetTime, target, -1);
                    WeightedAverage average = new WeightedAverage();

                    for (AlignedNeighbor neighbor : alignedNeighbors) {
                        AlignedAverage aligned = alignedAverage1D(
                                neighborData.get(neighbor.index()),
                                neighbor.path(),
                                targetTime
                        );

                        if (aligned.available()) {
                            average.add(neighbor.weight(), aligned.value());
                        }
                    }

                    setNumericValue1D(
                            replacement,
                            targetTime,
                            average.available()
                                    ? average.value()
                                    : fallbackValue1D(
                                            replacement,
                                            targetTime,
                                            fallbackMeans
                                    )
                    );
                }

                updated[target] = replacement;
            }
        });

        dataToUpdate.setData(asObjectList(updated));
        dataToUpdate.setMissingIndices(retainedMissing);
    }

    private static void impute2D(
            ListObjectDataset dataToUpdate,
            List<Object> targetData,
            List<Object> neighborData,
            MissingIndices targetMissing,
            MissingIndices neighborMissing,
            NeighborRows neighborRows,
            boolean excludeSelf,
            ParallelRuntime runtime,
            int windowSize
    ) throws Exception {
        MissingIndices retainedMissing = dataToUpdate.getMissingIndices();
        double[][] fallbackMeans = computeObservedMeans2D(
                neighborData,
                neighborMissing
        );
        Object[] updated = new Object[targetData.size()];

        forRanges(targetData.size(), runtime, (start, end) -> {
            for (int target = start; target < end; target++) {
                Object targetSeries = targetData.get(target);
                Object replacement = copySeries2D(targetSeries);
                int missingDimensions = targetMissing.dimensionCount(target);

                List<AlignedNeighbor> alignedNeighbors = buildAlignedNeighbors(
                        target,
                        targetSeries,
                        neighborData,
                        neighborRows,
                        excludeSelf,
                        true,
                        windowSize
                );

                for (int dimension = 0;
                     dimension < missingDimensions;
                     dimension++) {
                    if (dimension >= dimensions2D(replacement)) {
                        continue;
                    }

                    int missingStart = targetMissing.start2D(
                            target,
                            dimension
                    );
                    int missingEnd = targetMissing.end2D(
                            target,
                            dimension
                    );
                    for (int missingOffset = missingStart;
                            missingOffset < missingEnd;
                            missingOffset++) {
                        int targetTime = targetMissing.positionAt(missingOffset);
                        requireIndex(
                                length2D(replacement, dimension),
                                targetTime,
                                target,
                                dimension
                        );

                        WeightedAverage average = new WeightedAverage();

                        for (AlignedNeighbor neighbor : alignedNeighbors) {
                            AlignedAverage aligned = alignedAverage2D(
                                    neighborData.get(neighbor.index()),
                                    neighbor.path(),
                                    dimension,
                                    targetTime
                            );

                            if (aligned.available()) {
                                average.add(neighbor.weight(), aligned.value());
                            }
                        }

                        setNumericValue2D(
                                replacement,
                                dimension,
                                targetTime,
                                average.available()
                                        ? average.value()
                                        : fallbackValue2D(
                                                replacement,
                                                dimension,
                                                targetTime,
                                                fallbackMeans
                                        )
                        );
                    }
                }

                updated[target] = replacement;
            }
        });

        dataToUpdate.setData(asObjectList(updated));
        dataToUpdate.setMissingIndices(retainedMissing);
    }

    private static List<AlignedNeighbor> buildAlignedNeighbors(
            int target,
            Object targetSeries,
            List<Object> neighborData,
            NeighborRows neighborRows,
            boolean excludeSelf,
            boolean twoDimensional,
            int windowSize
    ) throws Exception {
        List<AlignedNeighbor> aligned = new ArrayList<>();

        neighborRows.forEach(target, (neighbor, weight) -> {
            if (weight <= 0.0 || (excludeSelf && neighbor == target)) {
                return;
            }

            Object neighborSeries = neighborData.get(neighbor);
            List<Pair<Integer, Integer>> path = computeAlignmentPath(
                    targetSeries,
                    neighborSeries,
                    twoDimensional,
                    windowSize
            );

            if (path != null && !path.isEmpty()) {
                aligned.add(
                        new AlignedNeighbor(
                                neighbor,
                                weight,
                                path
                        )
                );
            }
        });

        return aligned;
    }

    private static List<Pair<Integer, Integer>> computeAlignmentPath(
            Object first,
            Object second,
            boolean twoDimensional,
            int windowSize
    ) {
        boolean missing = containsMissing(first) || containsMissing(second);

        if (twoDimensional) {
            if (missing) {
                return new DTWAROW_D().getAlignmentPath(
                        first,
                        second,
                        windowSize
                );
            }

            return new DTW_D().getAlignmentPath(
                    toDoubleSeries2D(first),
                    toDoubleSeries2D(second),
                    windowSize
            );
        }

        if (missing) {
            return new DTWAROW().getAlignmentPath(
                    first,
                    second,
                    windowSize
            );
        }

        return new DTWWithPath().getAlignmentPath(
                toDoubleSeries1D(first),
                toDoubleSeries1D(second),
                windowSize
        );
    }

    private static AlignedAverage alignedAverage1D(
            Object neighborSeries,
            List<Pair<Integer, Integer>> path,
            int targetTime
    ) {
        double sum = 0.0;
        int count = 0;

        for (Pair<Integer, Integer> pair : path) {
            if (pair.getKey() != targetTime) {
                continue;
            }

            int alignedTime = pair.getValue();
            if (!hasIndex1D(neighborSeries, alignedTime)) {
                continue;
            }

            double value = getNumericValue1D(neighborSeries, alignedTime);
            if (!Double.isNaN(value)) {
                sum += value;
                count++;
            }
        }

        return count == 0
                ? AlignedAverage.UNAVAILABLE
                : new AlignedAverage(true, sum / count);
    }

    private static AlignedAverage alignedAverage2D(
            Object neighborSeries,
            List<Pair<Integer, Integer>> path,
            int dimension,
            int targetTime
    ) {
        double sum = 0.0;
        int count = 0;

        for (Pair<Integer, Integer> pair : path) {
            if (pair.getKey() != targetTime) {
                continue;
            }

            int alignedTime = pair.getValue();
            if (!hasIndex2D(neighborSeries, dimension, alignedTime)) {
                continue;
            }

            double value = getNumericValue2D(
                    neighborSeries,
                    dimension,
                    alignedTime
            );
            if (!Double.isNaN(value)) {
                sum += value;
                count++;
            }
        }

        return count == 0
                ? AlignedAverage.UNAVAILABLE
                : new AlignedAverage(true, sum / count);
    }

    private static void validateRuntimeAndMatrix(
            ParallelRuntime runtime,
            ProximityMatrixResult proximities
    ) {
        Objects.requireNonNull(runtime, "ParallelRuntime cannot be null.");
        Objects.requireNonNull(
                proximities,
                "ProximityMatrixResult cannot be null."
        );
    }

    private static void validateShape(
            ProximityMatrixResult matrix,
            int expectedRows,
            int expectedColumns,
            String description
    ) {
        if (matrix.rowCount() != expectedRows
                || matrix.columnCount() != expectedColumns) {
            throw new IllegalArgumentException(
                    description
                            + " proximity matrix shape must be "
                            + expectedRows
                            + " x "
                            + expectedColumns
                            + ", but received "
                            + matrix.rowCount()
                            + " x "
                            + matrix.columnCount()
                            + "."
            );
        }
    }

    private static void forRanges(
            int count,
            ParallelRuntime runtime,
            RangeAction action
    ) throws Exception {
        if (runtime.isParallel()
                && count >= MINIMUM_PARALLEL_ROW_RANGE_SIZE) {
            runtime.forRanges(
                    0,
                    count,
                    MINIMUM_PARALLEL_ROW_RANGE_SIZE,
                    action::run
            );
            return;
        }

        action.run(0, count);
    }

    private static boolean containsMissing(
            Object series
    ) {
        if (series instanceof double[] row) {
            for (double value : row) {
                if (Double.isNaN(value)) {
                    return true;
                }
            }
            return false;
        }

        if (series instanceof double[][] matrix) {
            for (double[] row : matrix) {
                for (double value : row) {
                    if (Double.isNaN(value)) {
                        return true;
                    }
                }
            }
            return false;
        }
        if (series instanceof float[] row) {
            for (float value : row) {
                if (Float.isNaN(value)) {
                    return true;
                }
            }
            return false;
        }
        if (series instanceof float[][] matrix) {
            for (float[] row : matrix) {
                for (float value : row) {
                    if (Float.isNaN(value)) {
                        return true;
                    }
                }
            }
            return false;
        }

        if (series instanceof Object[] row) {
            for (Object value : row) {
                if (isMissingObject(value)) {
                    return true;
                }
            }
            return false;
        }

        if (series instanceof Object[][] matrix) {
            for (Object[] row : matrix) {
                for (Object value : row) {
                    if (isMissingObject(value)) {
                        return true;
                    }
                }
            }
        }

        return false;
    }

    private static boolean isMissingObject(
            Object value
    ) {
        return value == null
                || value instanceof Number number
                && Double.isNaN(number.doubleValue());
    }

    private static boolean hasIndex1D(
            Object series,
            int index
    ) {
        if (series instanceof double[] row) {
            return index >= 0 && index < row.length;
        }
        if (series instanceof float[] row) {
            return index >= 0 && index < row.length;
        }
        if (series instanceof Object[] row) {
            return index >= 0 && index < row.length;
        }
        return false;
    }

    private static boolean hasIndex2D(
            Object series,
            int dimension,
            int index
    ) {
        if (series instanceof double[][] matrix) {
            return dimension >= 0
                    && dimension < matrix.length
                    && index >= 0
                    && index < matrix[dimension].length;
        }
        if (series instanceof float[][] matrix) {
            return dimension >= 0
                    && dimension < matrix.length
                    && index >= 0
                    && index < matrix[dimension].length;
        }
        if (series instanceof Object[][] matrix) {
            return dimension >= 0
                    && dimension < matrix.length
                    && index >= 0
                    && index < matrix[dimension].length;
        }
        return false;
    }

    private static double getNumericValue1D(
            Object series,
            int index
    ) {
        if (series instanceof double[] row) {
            return row[index];
        }
        if (series instanceof float[] row) {
            return row[index];
        }
        if (series instanceof Object[] row) {
            return objectToDouble(row[index]);
        }
        throw unsupportedSeries(series, "1D");
    }

    private static double getNumericValue2D(
            Object series,
            int dimension,
            int index
    ) {
        if (series instanceof double[][] matrix) {
            return matrix[dimension][index];
        }
        if (series instanceof float[][] matrix) {
            return matrix[dimension][index];
        }
        if (series instanceof Object[][] matrix) {
            return objectToDouble(matrix[dimension][index]);
        }
        throw unsupportedSeries(series, "2D");
    }

    private static double objectToDouble(
            Object value
    ) {
        if (value == null) {
            return Double.NaN;
        }
        if (!(value instanceof Number number)) {
            throw new IllegalArgumentException(
                    "Expected numeric value but found "
                            + value.getClass().getName()
                            + "."
            );
        }
        return number.doubleValue();
    }

    private static Object copySeries1D(Object series) {
        if (series instanceof double[] row) {
            return row.clone();
        }
        if (series instanceof float[] row) {
            return row.clone();
        }
        throw unsupportedSeries(series, "1D");
    }

    private static Object copySeries2D(Object series) {
        if (series instanceof double[][] matrix) {
            double[][] copied = new double[matrix.length][];
            for (int dimension = 0; dimension < matrix.length; dimension++) {
                copied[dimension] = matrix[dimension].clone();
            }
            return copied;
        }
        if (series instanceof float[][] matrix) {
            float[][] copied = new float[matrix.length][];
            for (int dimension = 0; dimension < matrix.length; dimension++) {
                copied[dimension] = matrix[dimension].clone();
            }
            return copied;
        }
        throw unsupportedSeries(series, "2D");
    }

    private static double[] toDoubleSeries1D(Object series) {
        if (series instanceof double[] row) {
            return row;
        }
        if (series instanceof float[] row) {
            double[] converted = new double[row.length];
            for (int index = 0; index < row.length; index++) {
                converted[index] = row[index];
            }
            return converted;
        }
        throw unsupportedSeries(series, "1D");
    }

    private static double[][] toDoubleSeries2D(Object series) {
        if (series instanceof double[][] matrix) {
            return matrix;
        }
        if (series instanceof float[][] matrix) {
            double[][] converted = new double[matrix.length][];
            for (int dimension = 0; dimension < matrix.length; dimension++) {
                converted[dimension] = new double[matrix[dimension].length];
                for (int index = 0; index < matrix[dimension].length; index++) {
                    converted[dimension][index] = matrix[dimension][index];
                }
            }
            return converted;
        }
        throw unsupportedSeries(series, "2D");
    }

    private static IllegalArgumentException unsupportedSeries(
            Object series,
            String description
    ) {
        return new IllegalArgumentException(
                "Unsupported "
                        + description
                        + " numeric series type: "
                        + (series == null
                        ? "null"
                        : series.getClass().getName())
                        + "."
        );
    }

    private static void requireIndex(
            int length,
            int index,
            int instance,
            int dimension
    ) {
        if (index < 0 || index >= length) {
            throw new IndexOutOfBoundsException(
                    "Missing time index "
                            + index
                            + " is invalid for instance "
                            + instance
                            + (dimension < 0
                            ? ""
                            : ", dimension " + dimension)
                            + " with length "
                            + length
                            + "."
            );
        }
    }

    private static double fallbackValue1D(
            Object row,
            int index,
            double[] means
    ) {
        double current = getNumericValue1D(row, index);
        if (!Double.isNaN(current)) {
            return current;
        }
        if (index < means.length && !Double.isNaN(means[index])) {
            return means[index];
        }
        return 0.0;
    }

    private static double fallbackValue2D(
            Object matrix,
            int dimension,
            int index,
            double[][] means
    ) {
        double current = getNumericValue2D(matrix, dimension, index);
        if (!Double.isNaN(current)) {
            return current;
        }
        if (dimension < means.length
                && index < means[dimension].length
                && !Double.isNaN(means[dimension][index])) {
            return means[dimension][index];
        }
        return 0.0;
    }

    private static void setNumericValue1D(
            Object series,
            int index,
            double value
    ) {
        if (series instanceof double[] row) {
            row[index] = value;
            return;
        }
        if (series instanceof float[] row) {
            row[index] = (float) value;
            return;
        }
        throw unsupportedSeries(series, "1D");
    }

    private static void setNumericValue2D(
            Object series,
            int dimension,
            int index,
            double value
    ) {
        if (series instanceof double[][] matrix) {
            matrix[dimension][index] = value;
            return;
        }
        if (series instanceof float[][] matrix) {
            matrix[dimension][index] = (float) value;
            return;
        }
        throw unsupportedSeries(series, "2D");
    }

    private static double[] computeObservedMeans1D(
            List<Object> data,
            MissingIndices missing
    ) {
        int maximumLength = 0;
        for (Object series : data) {
            maximumLength = Math.max(maximumLength, length1D(series));
        }

        double[] sums = new double[maximumLength];
        int[] counts = new int[maximumLength];

        for (int instance = 0; instance < data.size(); instance++) {
            Object series = data.get(instance);
            int length = length1D(series);
            for (int index = 0; index < length; index++) {
                double value = getNumericValue1D(series, index);
                if (!Double.isNaN(value)) {
                    sums[index] += value;
                    counts[index]++;
                }
            }
        }

        double[] means = new double[maximumLength];
        for (int index = 0; index < maximumLength; index++) {
            means[index] = counts[index] == 0
                    ? Double.NaN
                    : sums[index] / counts[index];
        }
        return means;
    }

    private static double[][] computeObservedMeans2D(
            List<Object> data,
            MissingIndices missing
    ) {
        int maximumDimensions = 0;
        for (Object series : data) {
            maximumDimensions = Math.max(
                    maximumDimensions,
                    dimensions2D(series)
            );
        }

        int[] maximumLengths = new int[maximumDimensions];
        for (Object series : data) {
            for (int dimension = 0;
                 dimension < dimensions2D(series);
                 dimension++) {
                maximumLengths[dimension] = Math.max(
                        maximumLengths[dimension],
                        length2D(series, dimension)
                );
            }
        }

        double[][] sums = new double[maximumDimensions][];
        int[][] counts = new int[maximumDimensions][];
        for (int dimension = 0;
             dimension < maximumDimensions;
             dimension++) {
            sums[dimension] = new double[maximumLengths[dimension]];
            counts[dimension] = new int[maximumLengths[dimension]];
        }

        for (int instance = 0; instance < data.size(); instance++) {
            Object series = data.get(instance);
            for (int dimension = 0;
                 dimension < dimensions2D(series);
                 dimension++) {
                for (int index = 0;
                     index < length2D(series, dimension);
                     index++) {

                    double value = getNumericValue2D(
                            series,
                            dimension,
                            index
                    );
                    if (!Double.isNaN(value)) {
                        sums[dimension][index] += value;
                        counts[dimension][index]++;
                    }
                }
            }
        }

        double[][] means = new double[maximumDimensions][];
        for (int dimension = 0;
             dimension < maximumDimensions;
             dimension++) {
            means[dimension] = new double[maximumLengths[dimension]];
            for (int index = 0;
                 index < maximumLengths[dimension];
                 index++) {
                means[dimension][index] = counts[dimension][index] == 0
                        ? Double.NaN
                        : sums[dimension][index]
                        / counts[dimension][index];
            }
        }
        return means;
    }

    private static boolean isMissing1D(
            MissingIndices missing,
            int instance,
            int index
    ) {
        if (missing == null) {
            return false;
        }
        if (!missing.is1D()
                || instance < 0
                || instance >= missing.instanceCount()) {
            return true;
        }
        return missing.contains1D(instance, index);
    }

    private static boolean isMissing2D(
            MissingIndices missing,
            int instance,
            int dimension,
            int index
    ) {
        if (missing == null) {
            return false;
        }
        if (!missing.is2D()
                || instance < 0
                || instance >= missing.instanceCount()
                || dimension < 0
                || dimension >= missing.dimensionCount(instance)) {
            return true;
        }
        return missing.contains2D(instance, dimension, index);
    }

    private static int length1D(
            Object series
    ) {
        if (series instanceof double[] row) {
            return row.length;
        }
        if (series instanceof float[] row) {
            return row.length;
        }
        if (series instanceof Object[] row) {
            return row.length;
        }
        return 0;
    }

    private static int dimensions2D(
            Object series
    ) {
        if (series instanceof double[][] matrix) {
            return matrix.length;
        }
        if (series instanceof float[][] matrix) {
            return matrix.length;
        }
        if (series instanceof Object[][] matrix) {
            return matrix.length;
        }
        return 0;
    }

    private static int length2D(
            Object series,
            int dimension
    ) {
        if (series instanceof double[][] matrix) {
            return dimension >= 0 && dimension < matrix.length
                    ? matrix[dimension].length
                    : 0;
        }
        if (series instanceof float[][] matrix) {
            return dimension >= 0 && dimension < matrix.length
                    ? matrix[dimension].length
                    : 0;
        }
        if (series instanceof Object[][] matrix) {
            return dimension >= 0 && dimension < matrix.length
                    ? matrix[dimension].length
                    : 0;
        }
        return 0;
    }

    private static List<Object> asObjectList(
            Object[] values
    ) {
        return new ArrayList<>(Arrays.asList(values));
    }

    @FunctionalInterface
    private interface RangeAction {
        void run(int startInclusive, int endExclusive) throws Exception;
    }

    @FunctionalInterface
    private interface NeighborConsumer {
        void accept(int neighborIndex, double weight) throws Exception;
    }

    private interface NeighborRows {
        void forEach(int rowIndex, NeighborConsumer consumer) throws Exception;

        static NeighborRows from(ProximityMatrixResult result) {
            if (result instanceof ProximityMatrixResult.Dense dense) {
                return new DenseNeighborRows(dense.values());
            }
            if (result instanceof ProximityMatrixResult.Sparse sparse) {
                return new SparseNeighborRows(sparse.values());
            }
            throw new IllegalArgumentException(
                    "Unsupported proximity result implementation: "
                            + result.getClass().getName()
                            + "."
            );
        }
    }

    private record DenseNeighborRows(double[][] matrix)
            implements NeighborRows {
        @Override
        public void forEach(
                int rowIndex,
                NeighborConsumer consumer
        ) throws Exception {
            double[] row = matrix[rowIndex];
            for (int column = 0; column < row.length; column++) {
                double weight = row[column];
                if (weight != 0.0) {
                    consumer.accept(column, weight);
                }
            }
        }
    }

    private record SparseNeighborRows(
            CompressedSparseProximityMatrix matrix
    ) implements NeighborRows {
        @Override
        public void forEach(
                int rowIndex,
                NeighborConsumer consumer
        ) throws Exception {
            int count = matrix.rowEntryCount(rowIndex);
            for (int offset = 0; offset < count; offset++) {
                consumer.accept(
                        matrix.columnIndexAt(rowIndex, offset),
                        matrix.valueAt(rowIndex, offset)
                );
            }
        }
    }

    private record AlignedNeighbor(
            int index,
            double weight,
            List<Pair<Integer, Integer>> path
    ) {
    }

    private record AlignedAverage(
            boolean available,
            double value
    ) {
        private static final AlignedAverage UNAVAILABLE =
                new AlignedAverage(false, Double.NaN);
    }

    private static final class WeightedAverage {
        private double weightedSum;
        private double totalWeight;

        private void add(double weight, double value) {
            weightedSum += weight * value;
            totalWeight += weight;
        }

        private boolean available() {
            return totalWeight > 0.0;
        }

        private double value() {
            return weightedSum / totalWeight;
        }
    }
}
