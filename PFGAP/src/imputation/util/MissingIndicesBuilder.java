package imputation.util;

import java.util.List;
import java.util.Objects;

/**
 * Builds final CSR missing-position metadata directly from eager datasets.
 *
 * <p>The builder performs two sequential passes. The first validates the
 * homogeneous representation and computes exact CSR offsets. The second writes
 * positions directly into the final flat array. No boxed position objects,
 * nested position arrays, growable buffers, or flattening copies are created.</p>
 */
public final class MissingIndicesBuilder {

    private MissingIndicesBuilder() {
        // Utility class.
    }

    public static MissingIndices buildFromDataset(List<Object> dataset) {
        if (dataset == null || dataset.isEmpty()) {
            throw new IllegalArgumentException(
                    "Dataset cannot be null or empty."
            );
        }

        Object first = requireInstance(dataset, 0);
        if (first instanceof double[]) {
            return buildDouble1D(dataset);
        }
        if (first instanceof float[]) {
            return buildFloat1D(dataset);
        }
        if (first instanceof double[][]) {
            return buildDouble2D(dataset);
        }
        if (first instanceof float[][]) {
            return buildFloat2D(dataset);
        }
        if (first instanceof Object[][]) {
            rejectBoxedNumericArray(first);
            return buildObject2D(dataset);
        }
        if (first instanceof Object[]) {
            rejectBoxedNumericArray(first);
            return buildObject1D(dataset);
        }
        throw unsupportedType(first, 0);
    }

    private static MissingIndices buildDouble1D(List<Object> dataset) {
        int[] offsets = new int[checkedOffsetLength(dataset.size())];
        long total = 0L;

        for (int instance = 0; instance < dataset.size(); instance++) {
            Object value = requireInstance(dataset, instance);
            if (!(value instanceof double[] row)) {
                throw inconsistentType(value, instance, "double[]");
            }
            total = checkedAdd(total, countMissing(row));
            offsets[instance + 1] = checkedCount(total);
        }

        int[] positions = new int[(int) total];
        for (int instance = 0; instance < dataset.size(); instance++) {
            writeMissing(
                    (double[]) dataset.get(instance),
                    positions,
                    offsets[instance]
            );
        }
        return MissingIndices.takeOwnership1D(offsets, positions);
    }

    private static MissingIndices buildFloat1D(List<Object> dataset) {
        int[] offsets = new int[checkedOffsetLength(dataset.size())];
        long total = 0L;

        for (int instance = 0; instance < dataset.size(); instance++) {
            Object value = requireInstance(dataset, instance);
            if (!(value instanceof float[] row)) {
                throw inconsistentType(value, instance, "float[]");
            }
            total = checkedAdd(total, countMissing(row));
            offsets[instance + 1] = checkedCount(total);
        }

        int[] positions = new int[(int) total];
        for (int instance = 0; instance < dataset.size(); instance++) {
            writeMissing(
                    (float[]) dataset.get(instance),
                    positions,
                    offsets[instance]
            );
        }
        return MissingIndices.takeOwnership1D(offsets, positions);
    }

    private static MissingIndices buildObject1D(List<Object> dataset) {
        int[] offsets = new int[checkedOffsetLength(dataset.size())];
        long total = 0L;

        for (int instance = 0; instance < dataset.size(); instance++) {
            Object value = requireInstance(dataset, instance);
            rejectBoxedNumericArray(value);
            if (!(value instanceof Object[] row)
                    || value instanceof Object[][]) {
                throw inconsistentType(value, instance, "Object[]");
            }
            total = checkedAdd(total, countMissing(row));
            offsets[instance + 1] = checkedCount(total);
        }

        int[] positions = new int[(int) total];
        for (int instance = 0; instance < dataset.size(); instance++) {
            writeMissing(
                    (Object[]) dataset.get(instance),
                    positions,
                    offsets[instance]
            );
        }
        return MissingIndices.takeOwnership1D(offsets, positions);
    }

    private static MissingIndices buildDouble2D(List<Object> dataset) {
        TwoDimensionalLayout layout = countDouble2D(dataset);
        int[] positions = new int[layout.positionCount()];
        int group = 0;

        for (Object value : dataset) {
            double[][] matrix = (double[][]) value;
            for (double[] row : matrix) {
                writeMissing(
                        row,
                        positions,
                        layout.groupOffsets()[group++]
                );
            }
        }
        return MissingIndices.takeOwnership2D(
                layout.instanceOffsets(),
                layout.groupOffsets(),
                positions
        );
    }

    private static MissingIndices buildFloat2D(List<Object> dataset) {
        TwoDimensionalLayout layout = countFloat2D(dataset);
        int[] positions = new int[layout.positionCount()];
        int group = 0;

        for (Object value : dataset) {
            float[][] matrix = (float[][]) value;
            for (float[] row : matrix) {
                writeMissing(
                        row,
                        positions,
                        layout.groupOffsets()[group++]
                );
            }
        }
        return MissingIndices.takeOwnership2D(
                layout.instanceOffsets(),
                layout.groupOffsets(),
                positions
        );
    }

    private static MissingIndices buildObject2D(List<Object> dataset) {
        TwoDimensionalLayout layout = countObject2D(dataset);
        int[] positions = new int[layout.positionCount()];
        int group = 0;

        for (Object value : dataset) {
            Object[][] matrix = (Object[][]) value;
            for (Object[] row : matrix) {
                writeMissing(
                        row,
                        positions,
                        layout.groupOffsets()[group++]
                );
            }
        }
        return MissingIndices.takeOwnership2D(
                layout.instanceOffsets(),
                layout.groupOffsets(),
                positions
        );
    }

    private static TwoDimensionalLayout countDouble2D(
            List<Object> dataset
    ) {
        int[] instanceOffsets = countDimensionGroups(
                dataset,
                double[][].class
        );
        int[] groupOffsets = new int[
                checkedOffsetLength(instanceOffsets[dataset.size()])
        ];
        long total = 0L;
        int group = 0;

        for (int instance = 0; instance < dataset.size(); instance++) {
            Object value = requireInstance(dataset, instance);
            if (!(value instanceof double[][] matrix)) {
                throw inconsistentType(value, instance, "double[][]");
            }
            for (int dimension = 0;
                    dimension < matrix.length;
                    dimension++) {
                double[] row = requireRow(
                        matrix[dimension],
                        instance,
                        dimension,
                        "double[]"
                );
                total = checkedAdd(total, countMissing(row));
                groupOffsets[++group] = checkedCount(total);
            }
        }
        return new TwoDimensionalLayout(
                instanceOffsets,
                groupOffsets,
                (int) total
        );
    }

    private static TwoDimensionalLayout countFloat2D(
            List<Object> dataset
    ) {
        int[] instanceOffsets = countDimensionGroups(
                dataset,
                float[][].class
        );
        int[] groupOffsets = new int[
                checkedOffsetLength(instanceOffsets[dataset.size()])
        ];
        long total = 0L;
        int group = 0;

        for (int instance = 0; instance < dataset.size(); instance++) {
            Object value = requireInstance(dataset, instance);
            if (!(value instanceof float[][] matrix)) {
                throw inconsistentType(value, instance, "float[][]");
            }
            for (int dimension = 0;
                    dimension < matrix.length;
                    dimension++) {
                float[] row = requireRow(
                        matrix[dimension],
                        instance,
                        dimension,
                        "float[]"
                );
                total = checkedAdd(total, countMissing(row));
                groupOffsets[++group] = checkedCount(total);
            }
        }
        return new TwoDimensionalLayout(
                instanceOffsets,
                groupOffsets,
                (int) total
        );
    }

    private static TwoDimensionalLayout countObject2D(
            List<Object> dataset
    ) {
        int[] instanceOffsets = countDimensionGroups(
                dataset,
                Object[][].class
        );
        int[] groupOffsets = new int[
                checkedOffsetLength(instanceOffsets[dataset.size()])
        ];
        long total = 0L;
        int group = 0;

        for (int instance = 0; instance < dataset.size(); instance++) {
            Object value = requireInstance(dataset, instance);
            rejectBoxedNumericArray(value);
            if (!(value instanceof Object[][] matrix)) {
                throw inconsistentType(value, instance, "Object[][]");
            }
            for (int dimension = 0;
                    dimension < matrix.length;
                    dimension++) {
                Object[] row = requireRow(
                        matrix[dimension],
                        instance,
                        dimension,
                        "Object[]"
                );
                rejectBoxedNumericArray(row);
                total = checkedAdd(total, countMissing(row));
                groupOffsets[++group] = checkedCount(total);
            }
        }
        return new TwoDimensionalLayout(
                instanceOffsets,
                groupOffsets,
                (int) total
        );
    }

    private static int[] countDimensionGroups(
            List<Object> dataset,
            Class<?> expectedType
    ) {
        int[] offsets = new int[checkedOffsetLength(dataset.size())];
        long groups = 0L;

        for (int instance = 0; instance < dataset.size(); instance++) {
            Object value = requireInstance(dataset, instance);
            if (!expectedType.isInstance(value)) {
                throw inconsistentType(
                        value,
                        instance,
                        expectedType.getTypeName()
                );
            }
            int dimensions = value instanceof double[][] matrix
                    ? matrix.length
                    : value instanceof float[][] matrix
                    ? matrix.length
                    : ((Object[][]) value).length;
            groups = checkedAdd(groups, dimensions);
            offsets[instance + 1] = checkedCount(groups);
        }
        return offsets;
    }

    private static int countMissing(double[] row) {
        int count = 0;
        for (double value : row) {
            if (Double.isNaN(value)) {
                count++;
            }
        }
        return count;
    }

    private static int countMissing(float[] row) {
        int count = 0;
        for (float value : row) {
            if (Float.isNaN(value)) {
                count++;
            }
        }
        return count;
    }

    private static int countMissing(Object[] row) {
        int count = 0;
        for (Object value : row) {
            if (value == null) {
                count++;
            }
        }
        return count;
    }

    private static void writeMissing(
            double[] row,
            int[] positions,
            int output
    ) {
        for (int position = 0; position < row.length; position++) {
            if (Double.isNaN(row[position])) {
                positions[output++] = position;
            }
        }
    }

    private static void writeMissing(
            float[] row,
            int[] positions,
            int output
    ) {
        for (int position = 0; position < row.length; position++) {
            if (Float.isNaN(row[position])) {
                positions[output++] = position;
            }
        }
    }

    private static void writeMissing(
            Object[] row,
            int[] positions,
            int output
    ) {
        for (int position = 0; position < row.length; position++) {
            if (row[position] == null) {
                positions[output++] = position;
            }
        }
    }

    private static Object requireInstance(
            List<Object> dataset,
            int instance
    ) {
        return Objects.requireNonNull(
                dataset.get(instance),
                "Dataset instance " + instance + " cannot be null."
        );
    }

    private static <T> T requireRow(
            T row,
            int instance,
            int dimension,
            String expectedType
    ) {
        if (row == null) {
            throw new IllegalArgumentException(
                    "Dataset instance "
                            + instance
                            + ", dimension "
                            + dimension
                            + " is null. Expected "
                            + expectedType
                            + "."
            );
        }
        return row;
    }

    private static void rejectBoxedNumericArray(Object value) {
        Class<?> type = value.getClass();
        Class<?> component = type.getComponentType();
        while (component != null && component.isArray()) {
            component = component.getComponentType();
        }
        if (component != null
                && Number.class.isAssignableFrom(component)) {
            throw new IllegalArgumentException(
                    "Boxed numeric arrays are not supported: "
                            + type.getTypeName()
                            + ". Use primitive float/double arrays and NaN."
            );
        }
    }

    private static int checkedOffsetLength(int count) {
        if (count == Integer.MAX_VALUE) {
            throw new IllegalArgumentException(
                    "Too many CSR groups for an offset array."
            );
        }
        return count + 1;
    }

    private static long checkedAdd(long current, int addition) {
        long result = current + addition;
        if (result > Integer.MAX_VALUE) {
            throw new IllegalArgumentException(
                    "Missing-index metadata exceeds Java array limits."
            );
        }
        return result;
    }

    private static int checkedCount(long count) {
        if (count > Integer.MAX_VALUE) {
            throw new IllegalArgumentException(
                    "Missing-index metadata exceeds Java array limits."
            );
        }
        return (int) count;
    }

    private static IllegalArgumentException inconsistentType(
            Object value,
            int instance,
            String expectedType
    ) {
        return new IllegalArgumentException(
                "Dataset instance "
                        + instance
                        + " has type "
                        + value.getClass().getTypeName()
                        + ", expected "
                        + expectedType
                        + "."
        );
    }

    private static IllegalArgumentException unsupportedType(
            Object value,
            int instance
    ) {
        return new IllegalArgumentException(
                "Unsupported dataset instance type at index "
                        + instance
                        + ": "
                        + value.getClass().getTypeName()
                        + "."
        );
    }

    private record TwoDimensionalLayout(
            int[] instanceOffsets,
            int[] groupOffsets,
            int positionCount
    ) {
    }
}
