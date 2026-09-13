package outlier;

import core.parallel.ParallelRuntime;
import proximity.CompressedSparseProximityMatrix;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Computes classification outlier scores from proximity matrices.
 *
 * <p>This scorer is distinct from isolation-forest depth scoring. It applies
 * the random-forest proximity outlier definition within each class:</p>
 *
 * <pre>
 * raw(i) = N / sum over j in class(i) of proximity(i, j)^2
 * </pre>
 *
 * <p>Raw scores are then normalized within class by their class median and
 * mean absolute deviation from that median. This class does not route samples,
 * inspect tree depth, or implement isolation-mode scoring.</p>
 *
 * <p>Dense and CSR inputs are handled directly. Sparse rows are traversed in
 * time proportional to retained entries when symmetrization is disabled.
 * Parallel work is scheduled through the caller-owned
 * {@link ParallelRuntime}; no parallel streams or executors are created.</p>
 */
public final class OutlierScorer {

    private static final double ZERO_SUM_FLOOR = 1.0e-6;
    private static final double ZERO_DEVIATION_FLOOR = 1.0e-6;
    private static final int MINIMUM_PARALLEL_SCORE_RANGE_SIZE = 64;

    private OutlierScorer() {
    }

    /**
     * Computes proximity-based classification outlier scores from a dense
     * square training matrix.
     *
     * @param labels one class label per training observation
     * @param proximities square dense train/train proximity matrix
     * @param symmetrize whether to use (P(i,j) + P(j,i)) / 2
     * @param runtime caller-owned parallel runtime
     */
    public static double[] scoreDense(
            Object[] labels,
            double[][] proximities,
            boolean symmetrize,
            ParallelRuntime runtime
    ) throws Exception {
        LabelGroups groups = validateDenseInputs(
                labels,
                proximities,
                runtime
        );

        int observationCount = labels.length;
        double[] rawScores = new double[observationCount];

        forRanges(
                observationCount,
                runtime,
                (start, end) -> {
                    for (int source = start; source < end; source++) {
                        int[] sameClass = groups.indicesForObservation(source);
                        double sumOfSquares = 0.0;

                        for (int target : sameClass) {
                            double proximity = symmetrize
                                    ? 0.5 * (
                                    proximities[source][target]
                                            + proximities[target][source]
                            )
                                    : proximities[source][target];

                            sumOfSquares += proximity * proximity;
                        }

                        rawScores[source] = rawScore(
                                observationCount,
                                sumOfSquares
                        );
                    }
                }
        );

        return normalizeWithinClasses(
                rawScores,
                groups,
                runtime
        );
    }

    /**
     * Computes proximity-based classification outlier scores from a CSR square
     * training matrix.
     *
     * <p>Without symmetrization, each source row visits only retained entries
     * whose target has the same class label. Missing CSR entries contribute
     * zero and therefore require no work.</p>
     *
     * <p>With symmetrization, an auxiliary compressed-column index is built
     * once so each source can merge its outgoing and incoming nonzero entries
     * without scanning every same-class target.</p>
     */
    public static double[] scoreSparse(
            Object[] labels,
            CompressedSparseProximityMatrix proximities,
            boolean symmetrize,
            ParallelRuntime runtime
    ) throws Exception {
        LabelGroups groups = validateSparseInputs(
                labels,
                proximities,
                runtime
        );

        int observationCount = labels.length;
        double[] rawScores = new double[observationCount];
        CompressedColumns columns = symmetrize
                ? CompressedColumns.from(proximities)
                : null;

        forRanges(
                observationCount,
                runtime,
                (start, end) -> {
                    for (int source = start; source < end; source++) {
                        double sumOfSquares = symmetrize
                                ? symmetricSparseSumOfSquares(
                                source,
                                labels,
                                proximities,
                                columns
                        )
                                : directedSparseSumOfSquares(
                                source,
                                labels,
                                proximities
                        );

                        rawScores[source] = rawScore(
                                observationCount,
                                sumOfSquares
                        );
                    }
                }
        );

        return normalizeWithinClasses(
                rawScores,
                groups,
                runtime
        );
    }

    private static double directedSparseSumOfSquares(
            int source,
            Object[] labels,
            CompressedSparseProximityMatrix matrix
    ) {
        Object sourceLabel = labels[source];
        double sum = 0.0;
        int entryCount = matrix.rowEntryCount(source);

        for (int offset = 0; offset < entryCount; offset++) {
            int target = matrix.columnIndexAt(source, offset);
            if (!Objects.equals(sourceLabel, labels[target])) {
                continue;
            }

            double value = matrix.valueAt(source, offset);
            sum += value * value;
        }

        return sum;
    }

    /**
     * Merges one CSR row with the corresponding compressed column. Each target
     * in the union contributes exactly once.
     */
    private static double symmetricSparseSumOfSquares(
            int source,
            Object[] labels,
            CompressedSparseProximityMatrix rows,
            CompressedColumns columns
    ) {
        Object sourceLabel = labels[source];
        int rowOffset = 0;
        int rowCount = rows.rowEntryCount(source);
        int columnPosition = columns.offsets[source];
        int columnEnd = columns.offsets[source + 1];
        double sum = 0.0;

        while (rowOffset < rowCount || columnPosition < columnEnd) {
            int outgoingTarget = rowOffset < rowCount
                    ? rows.columnIndexAt(source, rowOffset)
                    : Integer.MAX_VALUE;

            int incomingSource = columnPosition < columnEnd
                    ? columns.rowIndices[columnPosition]
                    : Integer.MAX_VALUE;

            int target = Math.min(outgoingTarget, incomingSource);
            double outgoing = 0.0;
            double incoming = 0.0;

            if (outgoingTarget == target) {
                outgoing = rows.valueAt(source, rowOffset);
                rowOffset++;
            }

            if (incomingSource == target) {
                incoming = columns.values[columnPosition];
                columnPosition++;
            }

            if (Objects.equals(sourceLabel, labels[target])) {
                double symmetricValue = 0.5 * (outgoing + incoming);
                sum += symmetricValue * symmetricValue;
            }
        }

        return sum;
    }

    private static double rawScore(
            int observationCount,
            double sumOfSquares
    ) {
        if (!Double.isFinite(sumOfSquares) || sumOfSquares < 0.0) {
            throw new IllegalStateException(
                    "Within-class squared proximity sum must be finite and "
                            + "nonnegative, but received "
                            + sumOfSquares
                            + "."
            );
        }

        double denominator = sumOfSquares == 0.0
                ? ZERO_SUM_FLOOR
                : sumOfSquares;

        return observationCount / denominator;
    }

    private static double[] normalizeWithinClasses(
            double[] rawScores,
            LabelGroups groups,
            ParallelRuntime runtime
    ) throws Exception {
        int classCount = groups.classCount();
        double[] medians = new double[classCount];
        double[] meanAbsoluteDeviations = new double[classCount];

        for (int classIndex = 0; classIndex < classCount; classIndex++) {
            int[] indices = groups.indicesByClass[classIndex];
            double[] classScores = new double[indices.length];

            for (int position = 0; position < indices.length; position++) {
                classScores[position] = rawScores[indices[position]];
            }

            Arrays.sort(classScores);
            double median = medianOfSorted(classScores);
            double absoluteDeviationSum = 0.0;

            for (double score : classScores) {
                absoluteDeviationSum += Math.abs(score - median);
            }

            double meanAbsoluteDeviation = absoluteDeviationSum / classScores.length;
            medians[classIndex] = median;
            meanAbsoluteDeviations[classIndex] = meanAbsoluteDeviation == 0.0
                    ? ZERO_DEVIATION_FLOOR
                    : meanAbsoluteDeviation;
        }

        double[] normalized = new double[rawScores.length];

        forRanges(
                rawScores.length,
                runtime,
                (start, end) -> {
                    for (int index = start; index < end; index++) {
                        int classIndex = groups.classByObservation[index];
                        normalized[index] = Math.abs(
                                rawScores[index] - medians[classIndex]
                        ) / meanAbsoluteDeviations[classIndex];
                    }
                }
        );

        return normalized;
    }

    private static double medianOfSorted(
            double[] sortedValues
    ) {
        int count = sortedValues.length;
        int middle = count >>> 1;

        if ((count & 1) == 0) {
            return 0.5 * (
                    sortedValues[middle - 1]
                            + sortedValues[middle]
            );
        }

        return sortedValues[middle];
    }

    private static LabelGroups validateDenseInputs(
            Object[] labels,
            double[][] matrix,
            ParallelRuntime runtime
    ) {
        LabelGroups groups = validateLabelsAndRuntime(labels, runtime);
        Objects.requireNonNull(matrix, "Dense proximity matrix cannot be null.");

        if (matrix.length != labels.length) {
            throw new IllegalArgumentException(
                    "Dense proximity row count must match label count."
            );
        }

        for (int row = 0; row < matrix.length; row++) {
            if (matrix[row] == null || matrix[row].length != labels.length) {
                throw new IllegalArgumentException(
                        "Dense training proximity matrix must be square. "
                                + "Invalid row: "
                                + row
                                + "."
                );
            }

            for (int column = 0; column < matrix[row].length; column++) {
                if (!Double.isFinite(matrix[row][column])) {
                    throw new IllegalArgumentException(
                            "Dense proximity matrix contains non-finite value at "
                                    + row
                                    + ", "
                                    + column
                                    + "."
                    );
                }
            }
        }

        return groups;
    }

    private static LabelGroups validateSparseInputs(
            Object[] labels,
            CompressedSparseProximityMatrix matrix,
            ParallelRuntime runtime
    ) {
        LabelGroups groups = validateLabelsAndRuntime(labels, runtime);
        Objects.requireNonNull(matrix, "Sparse proximity matrix cannot be null.");

        if (matrix.rowCount() != labels.length
                || matrix.columnCount() != labels.length) {

            throw new IllegalArgumentException(
                    "Sparse training proximity matrix must be square and match "
                            + "the label count."
            );
        }

        return groups;
    }

    private static LabelGroups validateLabelsAndRuntime(
            Object[] labels,
            ParallelRuntime runtime
    ) {
        Objects.requireNonNull(labels, "Training labels cannot be null.");
        Objects.requireNonNull(runtime, "ParallelRuntime cannot be null.");

        if (labels.length == 0) {
            throw new IllegalArgumentException(
                    "At least one training label is required."
            );
        }

        return LabelGroups.from(labels);
    }

    private static void forRanges(
            int count,
            ParallelRuntime runtime,
            RangeAction action
    ) throws Exception {
        if (runtime.isParallel()
                && count >= MINIMUM_PARALLEL_SCORE_RANGE_SIZE) {

            runtime.forRanges(
                    0,
                    count,
                    MINIMUM_PARALLEL_SCORE_RANGE_SIZE,
                    action::run
            );
            return;
        }

        action.run(0, count);
    }

    @FunctionalInterface
    private interface RangeAction {
        void run(int startInclusive, int endExclusive) throws Exception;
    }

    private static final class LabelGroups {

        private final int[][] indicesByClass;
        private final int[] classByObservation;

        private LabelGroups(
                int[][] indicesByClass,
                int[] classByObservation
        ) {
            this.indicesByClass = indicesByClass;
            this.classByObservation = classByObservation;
        }

        private static LabelGroups from(
                Object[] labels
        ) {
            Map<Object, Integer> classIds = new HashMap<>();
            List<List<Integer>> groupedIndices = new ArrayList<>();
            int[] classByObservation = new int[labels.length];

            for (int index = 0; index < labels.length; index++) {
                Object label = labels[index];
                Integer classIndex = classIds.get(label);

                if (classIndex == null && !classIds.containsKey(label)) {
                    classIndex = groupedIndices.size();
                    classIds.put(label, classIndex);
                    groupedIndices.add(new ArrayList<>());
                }

                classByObservation[index] = classIndex;
                groupedIndices.get(classIndex).add(index);
            }

            int[][] indicesByClass = new int[groupedIndices.size()][];

            for (int classIndex = 0;
                 classIndex < groupedIndices.size();
                 classIndex++) {

                List<Integer> boxedIndices = groupedIndices.get(classIndex);
                int[] indices = new int[boxedIndices.size()];

                for (int position = 0; position < indices.length; position++) {
                    indices[position] = boxedIndices.get(position);
                }

                indicesByClass[classIndex] = indices;
            }

            return new LabelGroups(
                    indicesByClass,
                    classByObservation
            );
        }

        private int classCount() {
            return indicesByClass.length;
        }

        private int[] indicesForObservation(
                int observationIndex
        ) {
            return indicesByClass[classByObservation[observationIndex]];
        }
    }

    /**
     * Compressed-column mirror used only for exact sparse symmetrization.
     */
    private static final class CompressedColumns {

        private final int[] offsets;
        private final int[] rowIndices;
        private final double[] values;

        private CompressedColumns(
                int[] offsets,
                int[] rowIndices,
                double[] values
        ) {
            this.offsets = offsets;
            this.rowIndices = rowIndices;
            this.values = values;
        }

        private static CompressedColumns from(
                CompressedSparseProximityMatrix matrix
        ) {
            int size = matrix.columnCount();
            int nonzeroCount = matrix.nonZeroCount();
            int[] counts = new int[size];

            for (int row = 0; row < matrix.rowCount(); row++) {
                int entryCount = matrix.rowEntryCount(row);
                for (int offset = 0; offset < entryCount; offset++) {
                    counts[matrix.columnIndexAt(row, offset)]++;
                }
            }

            int[] offsets = new int[size + 1];
            for (int index = 0; index < size; index++) {
                offsets[index + 1] = Math.addExact(offsets[index], counts[index]);
            }

            int[] rowIndices = new int[nonzeroCount];
            double[] values = new double[nonzeroCount];
            int[] cursors = Arrays.copyOf(offsets, size);

            for (int row = 0; row < matrix.rowCount(); row++) {
                int entryCount = matrix.rowEntryCount(row);
                for (int offset = 0; offset < entryCount; offset++) {
                    int column = matrix.columnIndexAt(row, offset);
                    int destination = cursors[column]++;
                    rowIndices[destination] = row;
                    values[destination] = matrix.valueAt(row, offset);
                }
            }

            return new CompressedColumns(
                    offsets,
                    rowIndices,
                    values
            );
        }
    }
}
