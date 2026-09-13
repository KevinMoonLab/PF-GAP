package imputation.update;

import core.parallel.ParallelRuntime;
import datasets.ListObjectDataset;
import imputation.util.MissingIndices;
import proximity.CompressedSparseProximityMatrix;
import proximity.ProximityMatrixResult;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Performs standard proximity-weighted imputation updates.
 *
 * <p>This class consumes an explicit {@link ProximityMatrixResult}; it does not
 * read proximity matrices from global application state and does not convert
 * dense matrices to boxed sparse maps. Dense rows are scanned directly, while
 * sparse rows visit only retained CSR entries.</p>
 *
 * <p>Numeric updates use Jacobi-style iteration semantics: every proposed row
 * is computed from the unchanged input dataset and stored separately, then the
 * complete replacement dataset is published after all workers finish. This
 * makes source-row parallelism deterministic and prevents one row from reading
 * another row's partially updated values.</p>
 *
 * <p>This integration version preserves support for primitive {@code double[]}
 * and {@code double[][]} as well as boxed numeric arrays. Updated numeric rows
 * are emitted as primitive arrays. A later representation overhaul can remove
 * the boxed compatibility paths and optimize missing-position lookup.</p>
 */
public final class PFImpute {

    private static final int MINIMUM_PARALLEL_ROW_RANGE_SIZE = 32;

    private PFImpute() {
    }

    /** Updates missing numeric values in a training dataset. */
    public static void trainNumericImpute(
            ListObjectDataset dataToUpdate,
            ProximityMatrixResult proximities,
            ParallelRuntime runtime
    ) throws Exception {
        Objects.requireNonNull(dataToUpdate, "Training data cannot be null.");
        validateRuntimeAndMatrix(runtime, proximities);

        List<Object> rawData = dataToUpdate.getData();
        MissingIndices missingIndices = dataToUpdate.getMissingIndices();

        if (rawData == null || rawData.isEmpty() || missingIndices == null) {
            return;
        }

        validateShape(
                proximities,
                rawData.size(),
                rawData.size(),
                "training"
        );

        NeighborRows neighborRows = NeighborRows.from(proximities);

        if (missingIndices.is2D()) {
            trainNumericImpute2D(
                    dataToUpdate,
                    neighborRows,
                    runtime
            );
        } else {
            trainNumericImpute1D(
                    dataToUpdate,
                    neighborRows,
                    runtime
            );
        }
    }

    /** Updates missing numeric values in a test dataset from training values. */
    public static void testNumericImpute(
            ListObjectDataset testData,
            ListObjectDataset trainData,
            ProximityMatrixResult proximities,
            ParallelRuntime runtime
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

        NeighborRows neighborRows = NeighborRows.from(proximities);

        if (testMissing.is2D()) {
            testNumericImpute2D(
                    testData,
                    trainData,
                    neighborRows,
                    runtime
            );
        } else {
            testNumericImpute1D(
                    testData,
                    trainData,
                    neighborRows,
                    runtime
            );
        }
    }

    /** Updates missing categorical values in a training dataset. */
    public static void trainCategoricalImpute(
            ListObjectDataset data,
            ProximityMatrixResult proximities,
            ParallelRuntime runtime
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

        updateCategorical(
                data,
                data,
                missing,
                NeighborRows.from(proximities),
                true,
                runtime
        );
    }

    /** Updates missing categorical test values from training values. */
    public static void testCategoricalImpute(
            ListObjectDataset testData,
            ListObjectDataset trainData,
            ProximityMatrixResult proximities,
            ParallelRuntime runtime
    ) throws Exception {
        Objects.requireNonNull(testData, "Testing data cannot be null.");
        Objects.requireNonNull(trainData, "Training data cannot be null.");
        validateRuntimeAndMatrix(runtime, proximities);

        List<Object> testRaw = testData.getData();
        MissingIndices missing = testData.getMissingIndices();

        if (testRaw == null || testRaw.isEmpty() || missing == null) {
            return;
        }

        validateShape(
                proximities,
                testRaw.size(),
                trainData.size(),
                "test/train"
        );

        updateCategorical(
                testData,
                trainData,
                missing,
                NeighborRows.from(proximities),
                false,
                runtime
        );
    }

    private static void trainNumericImpute1D(
            ListObjectDataset data,
            NeighborRows neighbors,
            ParallelRuntime runtime
    ) throws Exception {
        List<Object> rawData = data.getData();
        List<List<Integer>> missing = data.getMissingIndices().indices1D;
        double[] fallbackMeans = computeObservedMeans1D(rawData, missing);
        Object[] updated = new Object[rawData.size()];

        forRanges(rawData.size(), runtime, (start, end) -> {
            for (int target = start; target < end; target++) {
                double[] row = copySeries1DToPrimitive(rawData.get(target));
                List<Integer> targetMissing = missing.get(target);

                for (int feature : targetMissing) {
                    ensureFeatureIndex(row.length, feature, target, -1);
                    NumericAccumulator accumulator = new NumericAccumulator();

                    int finalTarget = target;
                    neighbors.forEach(target, (neighbor, weight) -> {
                        if (neighbor == finalTarget || weight <= 0.0) {
                            return;
                        }
                        if (isMissing1D(missing, neighbor, feature)) {
                            return;
                        }

                        Object neighborSeries = rawData.get(neighbor);
                        if (!hasIndex1D(neighborSeries, feature)) {
                            return;
                        }

                        double value = getNumericValue1D(neighborSeries, feature);
                        if (!Double.isNaN(value)) {
                            accumulator.add(weight, value);
                        }
                    });

                    row[feature] = accumulator.hasWeight()
                            ? accumulator.mean()
                            : fallbackValue1D(row, feature, fallbackMeans);
                }

                updated[target] = row;
            }
        });

        data.setData(asObjectList(updated));
    }

    private static void trainNumericImpute2D(
            ListObjectDataset data,
            NeighborRows neighbors,
            ParallelRuntime runtime
    ) throws Exception {
        List<Object> rawData = data.getData();
        List<List<List<Integer>>> missing = data.getMissingIndices().indices2D;
        double[][] fallbackMeans = computeObservedMeans2D(rawData, missing);
        Object[] updated = new Object[rawData.size()];

        forRanges(rawData.size(), runtime, (start, end) -> {
            for (int target = start; target < end; target++) {
                double[][] matrix = copySeries2DToPrimitive(rawData.get(target));
                List<List<Integer>> targetMissing = missing.get(target);

                for (int dimension = 0;
                     dimension < targetMissing.size();
                     dimension++) {
                    if (dimension >= matrix.length) {
                        continue;
                    }

                    final int selectedDimension = dimension;
                    for (int feature : targetMissing.get(dimension)) {
                        ensureFeatureIndex(
                                matrix[dimension].length,
                                feature,
                                target,
                                dimension
                        );

                        NumericAccumulator accumulator = new NumericAccumulator();

                        int finalTarget = target;
                        neighbors.forEach(target, (neighbor, weight) -> {
                            if (neighbor == finalTarget || weight <= 0.0) {
                                return;
                            }
                            if (isMissing2D(
                                    missing,
                                    neighbor,
                                    selectedDimension,
                                    feature
                            )) {
                                return;
                            }

                            Object neighborSeries = rawData.get(neighbor);
                            if (!hasIndex2D(
                                    neighborSeries,
                                    selectedDimension,
                                    feature
                            )) {
                                return;
                            }

                            double value = getNumericValue2D(
                                    neighborSeries,
                                    selectedDimension,
                                    feature
                            );
                            if (!Double.isNaN(value)) {
                                accumulator.add(weight, value);
                            }
                        });

                        matrix[dimension][feature] = accumulator.hasWeight()
                                ? accumulator.mean()
                                : fallbackValue2D(
                                matrix,
                                dimension,
                                feature,
                                fallbackMeans
                        );
                    }
                }

                updated[target] = matrix;
            }
        });

        data.setData(asObjectList(updated));
    }

    private static void testNumericImpute1D(
            ListObjectDataset testData,
            ListObjectDataset trainData,
            NeighborRows neighbors,
            ParallelRuntime runtime
    ) throws Exception {
        List<Object> testRaw = testData.getData();
        List<Object> trainRaw = trainData.getData();
        List<List<Integer>> testMissing =
                testData.getMissingIndices().indices1D;
        List<List<Integer>> trainMissing = trainData.getMissingIndices() == null
                ? null
                : trainData.getMissingIndices().indices1D;

        double[] fallbackMeans = computeObservedMeans1D(
                trainRaw,
                trainMissing
        );
        Object[] updated = new Object[testRaw.size()];

        forRanges(testRaw.size(), runtime, (start, end) -> {
            for (int testIndex = start; testIndex < end; testIndex++) {
                double[] row = copySeries1DToPrimitive(testRaw.get(testIndex));

                for (int feature : testMissing.get(testIndex)) {
                    ensureFeatureIndex(row.length, feature, testIndex, -1);
                    NumericAccumulator accumulator = new NumericAccumulator();

                    neighbors.forEach(testIndex, (trainIndex, weight) -> {
                        if (weight <= 0.0
                                || isMissing1D(
                                trainMissing,
                                trainIndex,
                                feature
                        )) {
                            return;
                        }

                        Object trainSeries = trainRaw.get(trainIndex);
                        if (!hasIndex1D(trainSeries, feature)) {
                            return;
                        }

                        double value = getNumericValue1D(trainSeries, feature);
                        if (!Double.isNaN(value)) {
                            accumulator.add(weight, value);
                        }
                    });

                    row[feature] = accumulator.hasWeight()
                            ? accumulator.mean()
                            : fallbackValue1D(row, feature, fallbackMeans);
                }

                updated[testIndex] = row;
            }
        });

        testData.setData(asObjectList(updated));
    }

    private static void testNumericImpute2D(
            ListObjectDataset testData,
            ListObjectDataset trainData,
            NeighborRows neighbors,
            ParallelRuntime runtime
    ) throws Exception {
        List<Object> testRaw = testData.getData();
        List<Object> trainRaw = trainData.getData();
        List<List<List<Integer>>> testMissing =
                testData.getMissingIndices().indices2D;
        List<List<List<Integer>>> trainMissing =
                trainData.getMissingIndices() == null
                        ? null
                        : trainData.getMissingIndices().indices2D;

        double[][] fallbackMeans = computeObservedMeans2D(
                trainRaw,
                trainMissing
        );
        Object[] updated = new Object[testRaw.size()];

        forRanges(testRaw.size(), runtime, (start, end) -> {
            for (int testIndex = start; testIndex < end; testIndex++) {
                double[][] matrix = copySeries2DToPrimitive(
                        testRaw.get(testIndex)
                );
                List<List<Integer>> targetMissing = testMissing.get(testIndex);

                for (int dimension = 0;
                     dimension < targetMissing.size();
                     dimension++) {
                    if (dimension >= matrix.length) {
                        continue;
                    }

                    final int selectedDimension = dimension;
                    for (int feature : targetMissing.get(dimension)) {
                        ensureFeatureIndex(
                                matrix[dimension].length,
                                feature,
                                testIndex,
                                dimension
                        );

                        NumericAccumulator accumulator = new NumericAccumulator();

                        neighbors.forEach(testIndex, (trainIndex, weight) -> {
                            if (weight <= 0.0
                                    || isMissing2D(
                                    trainMissing,
                                    trainIndex,
                                    selectedDimension,
                                    feature
                            )) {
                                return;
                            }

                            Object trainSeries = trainRaw.get(trainIndex);
                            if (!hasIndex2D(
                                    trainSeries,
                                    selectedDimension,
                                    feature
                            )) {
                                return;
                            }

                            double value = getNumericValue2D(
                                    trainSeries,
                                    selectedDimension,
                                    feature
                            );
                            if (!Double.isNaN(value)) {
                                accumulator.add(weight, value);
                            }
                        });

                        matrix[dimension][feature] = accumulator.hasWeight()
                                ? accumulator.mean()
                                : fallbackValue2D(
                                matrix,
                                dimension,
                                feature,
                                fallbackMeans
                        );
                    }
                }

                updated[testIndex] = matrix;
            }
        });

        testData.setData(asObjectList(updated));
    }

    private static void updateCategorical(
            ListObjectDataset targets,
            ListObjectDataset neighborsData,
            MissingIndices missing,
            NeighborRows neighbors,
            boolean excludeSelf,
            ParallelRuntime runtime
    ) throws Exception {
        List<Object> targetData = targets.getData();
        List<Object> sourceData = neighborsData.getData();
        Object[] updated = new Object[targetData.size()];

        forRanges(targetData.size(), runtime, (start, end) -> {
            for (int target = start; target < end; target++) {
                if (missing.is2D()) {
                    Object[][] matrix = copyObjectMatrix(targetData.get(target));
                    List<List<Integer>> targetMissing = missing.indices2D.get(target);

                    for (int dimension = 0;
                         dimension < targetMissing.size();
                         dimension++) {
                        if (dimension >= matrix.length) {
                            continue;
                        }

                        final int selectedDimension = dimension;
                        for (int feature : targetMissing.get(dimension)) {
                            ensureFeatureIndex(
                                    matrix[dimension].length,
                                    feature,
                                    target,
                                    dimension
                            );

                            matrix[dimension][feature] = weightedMode(
                                    target,
                                    selectedDimension,
                                    feature,
                                    true,
                                    sourceData,
                                    neighbors,
                                    excludeSelf
                            );
                        }
                    }

                    updated[target] = matrix;
                } else {
                    Object[] row = copyObjectRow(targetData.get(target));

                    for (int feature : missing.indices1D.get(target)) {
                        ensureFeatureIndex(row.length, feature, target, -1);
                        row[feature] = weightedMode(
                                target,
                                -1,
                                feature,
                                false,
                                sourceData,
                                neighbors,
                                excludeSelf
                        );
                    }

                    updated[target] = row;
                }
            }
        });

        targets.setData(asObjectList(updated));
    }

    private static Object weightedMode(
            int target,
            int dimension,
            int feature,
            boolean twoDimensional,
            List<Object> sourceData,
            NeighborRows neighbors,
            boolean excludeSelf
    ) throws Exception {
        Map<Object, Double> frequencies = new HashMap<>();

        neighbors.forEach(target, (neighbor, weight) -> {
            if (weight <= 0.0 || (excludeSelf && neighbor == target)) {
                return;
            }

            Object value = categoricalValue(
                    sourceData.get(neighbor),
                    dimension,
                    feature,
                    twoDimensional
            );

            if (value != null) {
                frequencies.merge(value, weight, Double::sum);
            }
        });

        Object bestValue = null;
        double bestWeight = Double.NEGATIVE_INFINITY;

        for (Map.Entry<Object, Double> entry : frequencies.entrySet()) {
            if (entry.getValue() > bestWeight) {
                bestWeight = entry.getValue();
                bestValue = entry.getKey();
            }
        }

        return bestValue;
    }

    private static Object categoricalValue(
            Object series,
            int dimension,
            int feature,
            boolean twoDimensional
    ) {
        if (twoDimensional) {
            if (!(series instanceof Object[][] matrix)
                    || dimension < 0
                    || dimension >= matrix.length
                    || feature < 0
                    || feature >= matrix[dimension].length) {
                return null;
            }
            return matrix[dimension][feature];
        }

        if (!(series instanceof Object[] row)
                || feature < 0
                || feature >= row.length) {
            return null;
        }

        return row[feature];
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

    private static List<Object> asObjectList(
            Object[] values
    ) {
        return new ArrayList<>(Arrays.asList(values));
    }

    private static boolean isMissing1D(
            List<List<Integer>> missingIndices,
            int seriesIndex,
            int featureIndex
    ) {
        if (missingIndices == null) {
            return false;
        }
        if (seriesIndex < 0 || seriesIndex >= missingIndices.size()) {
            return true;
        }

        List<Integer> missing = missingIndices.get(seriesIndex);
        return missing != null && missing.contains(featureIndex);
    }

    private static boolean isMissing2D(
            List<List<List<Integer>>> missingIndices,
            int seriesIndex,
            int dimension,
            int featureIndex
    ) {
        if (missingIndices == null) {
            return false;
        }
        if (seriesIndex < 0 || seriesIndex >= missingIndices.size()) {
            return true;
        }

        List<List<Integer>> instanceMissing = missingIndices.get(seriesIndex);
        if (instanceMissing == null
                || dimension < 0
                || dimension >= instanceMissing.size()) {
            return true;
        }

        List<Integer> missing = instanceMissing.get(dimension);
        return missing != null && missing.contains(featureIndex);
    }

    private static boolean hasIndex1D(
            Object series,
            int featureIndex
    ) {
        if (series instanceof double[] values) {
            return featureIndex >= 0 && featureIndex < values.length;
        }
        if (series instanceof Object[] values) {
            return featureIndex >= 0 && featureIndex < values.length;
        }
        return false;
    }

    private static boolean hasIndex2D(
            Object series,
            int dimension,
            int featureIndex
    ) {
        if (series instanceof double[][] values) {
            return dimension >= 0
                    && dimension < values.length
                    && featureIndex >= 0
                    && featureIndex < values[dimension].length;
        }
        if (series instanceof Object[][] values) {
            return dimension >= 0
                    && dimension < values.length
                    && featureIndex >= 0
                    && featureIndex < values[dimension].length;
        }
        return false;
    }

    private static double getNumericValue1D(
            Object series,
            int featureIndex
    ) {
        if (series instanceof double[] values) {
            return values[featureIndex];
        }
        if (series instanceof Object[] values) {
            return objectToDouble(values[featureIndex]);
        }
        throw unsupportedSeries(series, "1D");
    }

    private static double getNumericValue2D(
            Object series,
            int dimension,
            int featureIndex
    ) {
        if (series instanceof double[][] values) {
            return values[dimension][featureIndex];
        }
        if (series instanceof Object[][] values) {
            return objectToDouble(values[dimension][featureIndex]);
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

    private static double[] copySeries1DToPrimitive(
            Object series
    ) {
        if (series instanceof double[] values) {
            return values.clone();
        }
        if (series instanceof Object[] values) {
            double[] copied = new double[values.length];
            for (int index = 0; index < values.length; index++) {
                copied[index] = objectToDouble(values[index]);
            }
            return copied;
        }
        throw unsupportedSeries(series, "1D");
    }

    private static double[][] copySeries2DToPrimitive(
            Object series
    ) {
        if (series instanceof double[][] values) {
            double[][] copied = new double[values.length][];
            for (int dimension = 0;
                 dimension < values.length;
                 dimension++) {
                copied[dimension] = values[dimension].clone();
            }
            return copied;
        }
        if (series instanceof Object[][] values) {
            double[][] copied = new double[values.length][];
            for (int dimension = 0;
                 dimension < values.length;
                 dimension++) {
                copied[dimension] = new double[values[dimension].length];
                for (int feature = 0;
                     feature < values[dimension].length;
                     feature++) {
                    copied[dimension][feature] = objectToDouble(
                            values[dimension][feature]
                    );
                }
            }
            return copied;
        }
        throw unsupportedSeries(series, "2D");
    }

    private static Object[] copyObjectRow(
            Object series
    ) {
        if (!(series instanceof Object[] row)) {
            throw unsupportedSeries(series, "categorical 1D");
        }
        return row.clone();
    }

    private static Object[][] copyObjectMatrix(
            Object series
    ) {
        if (!(series instanceof Object[][] matrix)) {
            throw unsupportedSeries(series, "categorical 2D");
        }

        Object[][] copied = new Object[matrix.length][];
        for (int dimension = 0;
             dimension < matrix.length;
             dimension++) {
            copied[dimension] = matrix[dimension].clone();
        }
        return copied;
    }

    private static IllegalArgumentException unsupportedSeries(
            Object series,
            String description
    ) {
        return new IllegalArgumentException(
                "Unsupported "
                        + description
                        + " series type: "
                        + (series == null
                        ? "null"
                        : series.getClass().getName())
                        + "."
        );
    }

    private static void ensureFeatureIndex(
            int length,
            int feature,
            int instance,
            int dimension
    ) {
        if (feature < 0 || feature >= length) {
            throw new IndexOutOfBoundsException(
                    "Missing feature index "
                            + feature
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
            double[] row,
            int feature,
            double[] fallbackMeans
    ) {
        if (!Double.isNaN(row[feature])) {
            return row[feature];
        }
        if (feature < fallbackMeans.length
                && !Double.isNaN(fallbackMeans[feature])) {
            return fallbackMeans[feature];
        }
        return 0.0;
    }

    private static double fallbackValue2D(
            double[][] matrix,
            int dimension,
            int feature,
            double[][] fallbackMeans
    ) {
        if (!Double.isNaN(matrix[dimension][feature])) {
            return matrix[dimension][feature];
        }
        if (dimension < fallbackMeans.length
                && feature < fallbackMeans[dimension].length
                && !Double.isNaN(fallbackMeans[dimension][feature])) {
            return fallbackMeans[dimension][feature];
        }
        return 0.0;
    }

    private static double[] computeObservedMeans1D(
            List<Object> data,
            List<List<Integer>> missingIndices
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

            for (int feature = 0; feature < length; feature++) {
                if (isMissing1D(missingIndices, instance, feature)) {
                    continue;
                }

                double value = getNumericValue1D(series, feature);
                if (!Double.isNaN(value)) {
                    sums[feature] += value;
                    counts[feature]++;
                }
            }
        }

        double[] means = new double[maximumLength];
        for (int feature = 0; feature < maximumLength; feature++) {
            means[feature] = counts[feature] == 0
                    ? Double.NaN
                    : sums[feature] / counts[feature];
        }
        return means;
    }

    private static double[][] computeObservedMeans2D(
            List<Object> data,
            List<List<List<Integer>>> missingIndices
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
                for (int feature = 0;
                     feature < length2D(series, dimension);
                     feature++) {
                    if (isMissing2D(
                            missingIndices,
                            instance,
                            dimension,
                            feature
                    )) {
                        continue;
                    }

                    double value = getNumericValue2D(
                            series,
                            dimension,
                            feature
                    );
                    if (!Double.isNaN(value)) {
                        sums[dimension][feature] += value;
                        counts[dimension][feature]++;
                    }
                }
            }
        }

        double[][] means = new double[maximumDimensions][];
        for (int dimension = 0;
             dimension < maximumDimensions;
             dimension++) {
            means[dimension] = new double[maximumLengths[dimension]];
            for (int feature = 0;
                 feature < maximumLengths[dimension];
                 feature++) {
                means[dimension][feature] = counts[dimension][feature] == 0
                        ? Double.NaN
                        : sums[dimension][feature]
                        / counts[dimension][feature];
            }
        }
        return means;
    }

    private static int length1D(
            Object series
    ) {
        if (series instanceof double[] values) {
            return values.length;
        }
        if (series instanceof Object[] values) {
            return values.length;
        }
        return 0;
    }

    private static int dimensions2D(
            Object series
    ) {
        if (series instanceof double[][] values) {
            return values.length;
        }
        if (series instanceof Object[][] values) {
            return values.length;
        }
        return 0;
    }

    private static int length2D(
            Object series,
            int dimension
    ) {
        if (series instanceof double[][] values) {
            return dimension >= 0 && dimension < values.length
                    ? values[dimension].length
                    : 0;
        }
        if (series instanceof Object[][] values) {
            return dimension >= 0 && dimension < values.length
                    ? values[dimension].length
                    : 0;
        }
        return 0;
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

    private static final class NumericAccumulator {
        private double weightedSum;
        private double totalWeight;

        private void add(double weight, double value) {
            weightedSum += weight * value;
            totalWeight += weight;
        }

        private boolean hasWeight() {
            return totalWeight > 0.0;
        }

        private double mean() {
            return weightedSum / totalWeight;
        }
    }
}
