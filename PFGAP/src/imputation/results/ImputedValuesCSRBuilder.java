package imputation.results;

import datasets.ListObjectDataset;
import datasets.NumericStorageType;
import imputation.util.MissingIndices;
import preprocessing.standardization.PerSeriesStandardizationState;
import preprocessing.standardization.StandardizationScope;
import preprocessing.standardization.StandardizationStats;

import java.util.List;
import java.util.Objects;

/** Builds an imputed-only CSR from final data and original missing metadata. */
public final class ImputedValuesCSRBuilder {

    private ImputedValuesCSRBuilder() {
    }

    public static ImputedValuesCSR build(
            ListObjectDataset dataset,
            MissingIndices missingIndices,
            StandardizationStats reusableStatistics,
            List<PerSeriesStandardizationState> perSeriesStates
    ) {
        Objects.requireNonNull(dataset, "Dataset cannot be null.");
        Objects.requireNonNull(missingIndices, "MissingIndices cannot be null.");
        List<Object> data = Objects.requireNonNull(
                dataset.getData(), "Dataset data cannot be null."
        );
        List<PerSeriesStandardizationState> states = perSeriesStates == null
                ? List.of() : perSeriesStates;
        validateSources(data, missingIndices, reusableStatistics, states);

        Shape shape = inspectShape(data, missingIndices.is2D());
        int[] rowOffsets = buildRowOffsets(missingIndices);
        int entryCount = rowOffsets[data.size()];
        int[] columnIndices = new int[entryCount];
        NumericStorageType storageType = detectStorageType(data);
        int reusableGroupCount = missingIndices.is2D()
                ? shape.maximumDimensionCount()
                : shape.maximumTimeLength();
        PreparedReusable reusable = PreparedReusable.from(
                reusableStatistics,
                reusableGroupCount
        );

        if (storageType == NumericStorageType.FLOAT32) {
            float[] values = new float[entryCount];
            fill(data, missingIndices, shape.maximumTimeLength(), states,
                    reusable, columnIndices, values, null);
            return ImputedValuesCSR.ofFloat(
                    data.size(), shape.columnCount(), rowOffsets, columnIndices,
                    values, missingIndices.is2D(),
                    shape.maximumDimensionCount(), shape.maximumTimeLength(),
                    shape.instanceDimensionOffsets(), shape.dimensionLengths()
            );
        }

        double[] values = new double[entryCount];
        fill(data, missingIndices, shape.maximumTimeLength(), states,
                reusable, columnIndices, null, values);
        return ImputedValuesCSR.ofDouble(
                data.size(), shape.columnCount(), rowOffsets, columnIndices,
                values, missingIndices.is2D(),
                shape.maximumDimensionCount(), shape.maximumTimeLength(),
                shape.instanceDimensionOffsets(), shape.dimensionLengths()
        );
    }

    private static void fill(
            List<Object> data,
            MissingIndices missing,
            int maximumTimeLength,
            List<PerSeriesStandardizationState> states,
            PreparedReusable reusable,
            int[] columns,
            float[] floatValues,
            double[] doubleValues
    ) {
        int output = 0;
        for (int instance = 0; instance < data.size(); instance++) {
            Object item = data.get(instance);
            PerSeriesStandardizationState state = states.isEmpty()
                    ? null : states.get(instance);
            if (missing.is1D()) {
                int start = missing.start1D(instance);
                int end = missing.end1D(instance);
                for (int offset = start; offset < end; offset++) {
                    int position = missing.positionAt(offset);
                    double value = value1D(item, position, instance);
                    columns[output] = position;
                    writeValue(
                            output++,
                            inverse(value, position, 0, reusable, state),
                            floatValues,
                            doubleValues
                    );
                }
            } else {
                int dimensions = missing.dimensionCount(instance);
                for (int dimension = 0; dimension < dimensions; dimension++) {
                    int start = missing.start2D(instance, dimension);
                    int end = missing.end2D(instance, dimension);
                    for (int offset = start; offset < end; offset++) {
                        int position = missing.positionAt(offset);
                        double value = value2D(
                                item, dimension, position, instance);
                        columns[output] = Math.addExact(
                                Math.multiplyExact(dimension, maximumTimeLength),
                                position
                        );
                        writeValue(
                                output++,
                                inverse(
                                        value, dimension, dimension,
                                        reusable, state
                                ),
                                floatValues,
                                doubleValues
                        );
                    }
                }
            }
        }
        if (output != columns.length) {
            throw new IllegalStateException(
                    "Missing-position traversal produced " + output
                            + " values; expected " + columns.length + "."
            );
        }
    }

    private static void writeValue(
            int offset,
            double value,
            float[] floatValues,
            double[] doubleValues
    ) {
        if (!Double.isFinite(value)) {
            throw new IllegalStateException(
                    "Imputed-only output contains a non-finite value. "
                            + "Imputation may be incomplete."
            );
        }
        if (floatValues != null) {
            float narrowed = (float) value;
            if (!Float.isFinite(narrowed)) {
                throw new IllegalStateException(
                        "Inverse-transformed value cannot be represented as float."
                );
            }
            floatValues[offset] = narrowed;
        } else {
            doubleValues[offset] = value;
        }
    }

    private static double inverse(
            double value,
            int reusableGroup,
            int localGroup,
            PreparedReusable reusable,
            PerSeriesStandardizationState state
    ) {
        if (state != null) {
            int group = state.getParameterGroupCount() == 1 ? 0 : localGroup;
            return value * state.getScale(group) + state.getCenter(group);
        }
        if (reusable != null) {
            int group = reusable.scope() == StandardizationScope.GLOBAL
                    ? 0 : reusableGroup;
            return value * reusable.scales()[group]
                    + reusable.centers()[group];
        }
        return value;
    }

    private static int[] buildRowOffsets(MissingIndices missing) {
        int[] offsets = new int[missing.instanceCount() + 1];
        long total = 0;
        for (int instance = 0; instance < missing.instanceCount(); instance++) {
            int count = 0;
            if (missing.is1D()) {
                count = missing.missingCount1D(instance);
            } else {
                for (int dimension = 0;
                     dimension < missing.dimensionCount(instance);
                     dimension++) {
                    count = Math.addExact(
                            count,
                            missing.missingCount2D(instance, dimension)
                    );
                }
            }
            total = Math.addExact(total, count);
            if (total > Integer.MAX_VALUE) {
                throw new IllegalArgumentException(
                        "Imputed CSR exceeds Java array capacity."
                );
            }
            offsets[instance + 1] = (int) total;
        }
        return offsets;
    }

    private static Shape inspectShape(
            List<Object> data,
            boolean twoDimensional
    ) {
        int[] instanceOffsets = new int[data.size() + 1];
        int maximumDimensions = twoDimensional ? 0 : 1;
        int maximumTime = 0;
        int groupCount = twoDimensional ? 0 : data.size();

        if (twoDimensional) {
            for (int instance = 0; instance < data.size(); instance++) {
                int dimensions = dimensionCount(data.get(instance), instance);
                maximumDimensions = Math.max(maximumDimensions, dimensions);
                groupCount = Math.addExact(groupCount, dimensions);
                instanceOffsets[instance + 1] = groupCount;
            }
        } else {
            for (int instance = 0; instance < data.size(); instance++) {
                instanceOffsets[instance + 1] = instance + 1;
            }
        }

        int[] lengths = new int[groupCount];
        int group = 0;
        for (int instance = 0; instance < data.size(); instance++) {
            Object item = data.get(instance);
            if (twoDimensional) {
                int dimensions = dimensionCount(item, instance);
                for (int dimension = 0; dimension < dimensions; dimension++) {
                    int length = dimensionLength(item, dimension, instance);
                    lengths[group++] = length;
                    maximumTime = Math.max(maximumTime, length);
                }
            } else {
                int length = vectorLength(item, instance);
                lengths[group++] = length;
                maximumTime = Math.max(maximumTime, length);
            }
        }
        int columns = Math.multiplyExact(maximumDimensions, maximumTime);
        return new Shape(
                columns, maximumDimensions, maximumTime,
                instanceOffsets, lengths
        );
    }

    private static NumericStorageType detectStorageType(List<Object> data) {
        Object first = data.get(0);
        boolean floatBacked = first instanceof float[]
                || first instanceof float[][];
        Class<?> type = first.getClass();
        for (int index = 0; index < data.size(); index++) {
            Object value = Objects.requireNonNull(data.get(index));
            if (value.getClass() != type) {
                throw new IllegalArgumentException(
                        "Imputed CSR requires homogeneous instance types."
                );
            }
        }
        return floatBacked
                ? NumericStorageType.FLOAT32
                : NumericStorageType.FLOAT64;
    }

    private static void validateSources(
            List<Object> data,
            MissingIndices missing,
            StandardizationStats reusable,
            List<PerSeriesStandardizationState> states
    ) {
        if (data.size() != missing.instanceCount()) {
            throw new IllegalArgumentException(
                    "Dataset size and MissingIndices instance count differ."
            );
        }
        if (data.isEmpty()) {
            throw new IllegalArgumentException(
                    "Cannot build imputed CSR from an empty dataset."
            );
        }
        if (reusable != null && !states.isEmpty()) {
            throw new IllegalArgumentException(
                    "Supply reusable statistics or per-series states, not both."
            );
        }
        if (!states.isEmpty() && states.size() != data.size()) {
            throw new IllegalArgumentException(
                    "Per-series state count does not match dataset size."
            );
        }
        Object first = data.get(0);
        boolean item2D = first instanceof double[][]
                || first instanceof float[][]
                || first instanceof Object[][];
        if (item2D != missing.is2D()) {
            throw new IllegalArgumentException(
                    "Dataset dimensionality and MissingIndices dimensionality differ."
            );
        }
    }

    private static double value1D(Object item, int position, int instance) {
        if (item instanceof double[] values) return values[position];
        if (item instanceof float[] values) return values[position];
        if (item instanceof Object[] values) {
            Object value = values[position];
            if (value instanceof Number number) return number.doubleValue();
            throw nonnumeric(instance, 0, position, value);
        }
        throw new IllegalArgumentException("Unsupported 1D dataset representation.");
    }

    private static double value2D(
            Object item, int dimension, int position, int instance
    ) {
        if (item instanceof double[][] values) return values[dimension][position];
        if (item instanceof float[][] values) return values[dimension][position];
        if (item instanceof Object[][] values) {
            Object value = values[dimension][position];
            if (value instanceof Number number) return number.doubleValue();
            throw nonnumeric(instance, dimension, position, value);
        }
        throw new IllegalArgumentException("Unsupported 2D dataset representation.");
    }

    private static IllegalArgumentException nonnumeric(
            int instance, int dimension, int position, Object value
    ) {
        return new IllegalArgumentException(
                "Imputed cell is not numeric at instance " + instance
                        + ", dimension " + dimension
                        + ", position " + position
                        + ": " + (value == null ? "null" : value.getClass().getTypeName())
                        + ". Matrix Market output supports numeric imputations only."
        );
    }

    private static int vectorLength(Object item, int instance) {
        if (item instanceof double[] values) return values.length;
        if (item instanceof float[] values) return values.length;
        if (item instanceof Object[] values) return values.length;
        throw new IllegalArgumentException(
                "Unsupported vector type at instance " + instance + "."
        );
    }

    private static int dimensionCount(Object item, int instance) {
        if (item instanceof double[][] values) return values.length;
        if (item instanceof float[][] values) return values.length;
        if (item instanceof Object[][] values) return values.length;
        throw new IllegalArgumentException(
                "Unsupported matrix type at instance " + instance + "."
        );
    }

    private static int dimensionLength(
            Object item, int dimension, int instance
    ) {
        if (item instanceof double[][] values) return values[dimension].length;
        if (item instanceof float[][] values) return values[dimension].length;
        if (item instanceof Object[][] values) return values[dimension].length;
        throw new IllegalArgumentException(
                "Unsupported matrix type at instance " + instance + "."
        );
    }

    private record Shape(
            int columnCount,
            int maximumDimensionCount,
            int maximumTimeLength,
            int[] instanceDimensionOffsets,
            int[] dimensionLengths
    ) {
    }

    private record PreparedReusable(
            StandardizationScope scope,
            double[] centers,
            double[] scales
    ) {
        private static PreparedReusable from(
                StandardizationStats stats,
                int dimensionCount
        ) {
            if (stats == null) return null;
            StandardizationScope scope = stats.getScope();
            if (!scope.usesTrainingStatistics()) {
                throw new IllegalArgumentException(
                        "Reusable statistics must use GLOBAL or PER_DIMENSION."
                );
            }
            double[] centers = stats.getCenters();
            double[] scales = stats.getScales();
            int expected = scope == StandardizationScope.GLOBAL
                    ? 1 : dimensionCount;
            if (centers.length != expected || scales.length != expected) {
                throw new IllegalArgumentException(
                        "Reusable parameter count does not match CSR dimensions."
                );
            }
            return new PreparedReusable(scope, centers, scales);
        }
    }
}
