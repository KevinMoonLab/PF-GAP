package datasets.writers;

import datasets.ListObjectDataset;
import de.siegmar.fastcsv.writer.CsvWriter;
import preprocessing.standardization.PerSeriesStandardizationState;
import preprocessing.standardization.StandardizationScope;
import preprocessing.standardization.StandardizationStats;

import java.io.IOException;
import java.lang.reflect.Array;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.List;
import java.util.Objects;

/**
 * FastCSV-backed mirror writer for {@code DelimitedFileReader}.
 *
 * <p>Supported materialized instance shapes are:</p>
 *
 * <ol>
 *     <li>{@code double[]}, {@code float[]}, and generic
 *     {@code Object[]} vectors.</li>
 *     <li>{@code double[][]}, {@code float[][]}, and generic
 *     {@code Object[][]} dimension-major matrices.</li>
 * </ol>
 *
 * <p>For one-dimensional data, FastCSV writes one field per array entry using
 * {@link DatasetWriteOptions#getEntrySeparatorCharacter()} as the field
 * separator. For row-encoded two-dimensional data, FastCSV uses the configured
 * array separator between dimensions. Each dimension is assembled as one field
 * whose values are separated by the entry separator. FastCSV then quotes that
 * field when required.</p>
 *
 * <p>Null generic entries are written as empty fields or empty inner
 * tokens, matching the generic reader's missing-value interpretation.</p>
 *
 * <p>Numeric inverse transformation is fused with textual record construction.
 * Non-numeric generic values are emitted unchanged. The source dataset is never
 * mutated.</p>
 */
public final class DelimitedFileWriter implements DatasetWriter {

    @Override
    public Path write(
            ListObjectDataset dataset,
            DatasetWriteOptions options
    ) throws IOException {
        Objects.requireNonNull(dataset, "Dataset cannot be null.");
        Objects.requireNonNull(options, "DatasetWriteOptions cannot be null.");

        List<Object> data = Objects.requireNonNull(
                dataset.getData(),
                "Dataset data cannot be null."
        );
        if (data.isEmpty()) {
            throw new IllegalArgumentException(
                    "Cannot write an empty generic delimited dataset."
            );
        }
        if (!supports(dataset, options)) {
            Object first = data.get(0);
            throw new IllegalArgumentException(
                    "DELIMITED output does not support materialized instance "
                            + "type "
                            + (first == null
                            ? "null"
                            : first.getClass().getTypeName())
                            + " with layout "
                            + options.getDataLayout()
                            + "."
            );
        }

        boolean matrix = isMatrix(data.get(0));
        DatasetWriteOptions.DataLayout layout = resolveLayout(
                data,
                options,
                matrix
        );
        Class<?> instanceType = data.get(0).getClass();
        PreparedReusableParameters reusable =
                PreparedReusableParameters.from(
                        options.getReusableStatistics()
                );
        List<PerSeriesStandardizationState> states =
                options.getPerSeriesStates();

        validateStateCount(states, data.size());
        validateLabelCount(options, data.size());
        if (matrix && options.shouldEmbedLabels()) {
            throw new IllegalArgumentException(
                    "The mirror DelimitedFileReader does not parse embedded "
                            + "labels for row-encoded two-dimensional data. "
                            + "Use a separate label file."
            );
        }
        if (!matrix
                && data.get(0) instanceof Object[]
                && options.shouldEmbedLabels()) {
            throw new IllegalArgumentException(
                    "The mirror DelimitedFileReader does not parse embedded "
                            + "labels from generic Object[] rows. Use a "
                            + "separate label file."
            );
        }
        validateDataset(
                data,
                instanceType,
                layout,
                options,
                reusable
        );
        createParentDirectories(options.getOutputPath());

        char outerSeparator = matrix
                ? options.getArraySeparatorCharacter()
                : options.getEntrySeparatorCharacter();

        try (CsvWriter writer = CsvWriter.builder()
                .fieldSeparator(outerSeparator)
                .build(
                        options.getOutputPath(),
                        StandardCharsets.UTF_8,
                        StandardOpenOption.CREATE,
                        StandardOpenOption.TRUNCATE_EXISTING,
                        StandardOpenOption.WRITE
                )) {
            String[] record = null;

            if (options.shouldIncludeHeader()) {
                int headerFields = headerFieldCount(
                        data.get(0),
                        layout,
                        options
                );
                record = new String[
                        headerFields
                                + (options.shouldEmbedLabels() ? 1 : 0)
                ];
                fillHeader(record, options, headerFields);
                writer.writeRecord(record);
            }

            for (int instanceIndex = 0;
                 instanceIndex < data.size();
                 instanceIndex++) {
                Object instance = data.get(instanceIndex);
                int dataFieldCount = matrix
                        ? matrixDimensionCount(instance)
                        : vectorLength(instance);
                int requiredFields = dataFieldCount
                        + (options.shouldEmbedLabels() ? 1 : 0);
                if (record == null || record.length != requiredFields) {
                    record = new String[requiredFields];
                }

                PerSeriesStandardizationState state = states.isEmpty()
                        ? null
                        : states.get(instanceIndex);
                validateInverseCompatibility(
                        instance,
                        layout,
                        reusable,
                        state,
                        instanceIndex
                );
                fillRecord(
                        record,
                        instance,
                        instanceIndex,
                        layout,
                        options,
                        reusable,
                        state
                );
                writer.writeRecord(record);
            }
        }

        return options.getOutputPath();
    }

    @Override
    public boolean supports(
            ListObjectDataset dataset,
            DatasetWriteOptions options
    ) {
        if (dataset == null || options == null || dataset.getData() == null
                || dataset.getData().isEmpty()) {
            return false;
        }
        Object first = dataset.getData().get(0);
        if (!isSupportedInstance(first)) {
            return false;
        }
        if (options.getDataLayout() == DatasetWriteOptions.DataLayout.LONG_FORM) {
            return false;
        }
        boolean expectedMatrix = isMatrix(first);
        Class<?> expectedType = first.getClass();
        for (Object instance : dataset.getData()) {
            if (instance == null
                    || instance.getClass() != expectedType
                    || isMatrix(instance) != expectedMatrix) {
                return false;
            }
        }
        return true;
    }

    @Override
    public String formatName() {
        return "DELIMITED";
    }

    private static DatasetWriteOptions.DataLayout resolveLayout(
            List<Object> data,
            DatasetWriteOptions options,
            boolean matrix
    ) {
        DatasetWriteOptions.DataLayout configured = options.getDataLayout();
        if (configured != DatasetWriteOptions.DataLayout.AUTO) {
            if (matrix
                    && configured
                    != DatasetWriteOptions.DataLayout.MULTIVARIATE_SERIES) {
                throw new IllegalArgumentException(
                        "Two-dimensional delimited instances require "
                                + "MULTIVARIATE_SERIES layout."
                );
            }
            if (!matrix
                    && configured
                    == DatasetWriteOptions.DataLayout.MULTIVARIATE_SERIES) {
                throw new IllegalArgumentException(
                        "One-dimensional delimited instances cannot use "
                                + "MULTIVARIATE_SERIES layout."
                );
            }
            return configured;
        }

        if (matrix) {
            return DatasetWriteOptions.DataLayout.MULTIVARIATE_SERIES;
        }
        if (options.shouldIncludeHeader()
                || options.hasFeatureNames()
                || (options.hasReusableStatistics()
                && options.getReusableStatistics().getScope()
                == StandardizationScope.PER_DIMENSION)) {
            return DatasetWriteOptions.DataLayout.TABULAR;
        }

        int firstLength = vectorLength(data.get(0));
        for (int index = 1; index < data.size(); index++) {
            if (vectorLength(data.get(index)) != firstLength) {
                return DatasetWriteOptions.DataLayout.UNIVARIATE_SERIES;
            }
        }
        return DatasetWriteOptions.DataLayout.TABULAR;
    }

    private static void fillHeader(
            String[] record,
            DatasetWriteOptions options,
            int dataFieldCount
    ) {
        int offset = options.shouldEmbedLabels() && options.isTargetFirst()
                ? 1
                : 0;
        for (int field = 0; field < dataFieldCount; field++) {
            record[offset + field] = options.getFeatureNames().get(field);
        }
        if (options.shouldEmbedLabels()) {
            record[options.isTargetFirst() ? 0 : record.length - 1] =
                    options.getTargetName();
        }
    }

    private static void fillRecord(
            String[] record,
            Object instance,
            int instanceIndex,
            DatasetWriteOptions.DataLayout layout,
            DatasetWriteOptions options,
            PreparedReusableParameters reusable,
            PerSeriesStandardizationState state
    ) {
        int offset = options.shouldEmbedLabels() && options.isTargetFirst()
                ? 1
                : 0;

        if (instance instanceof double[] values) {
            fillDoubleVector(record, offset, values, layout, reusable, state);
        } else if (instance instanceof float[] values) {
            fillFloatVector(record, offset, values, layout, reusable, state);
        } else if (instance instanceof Object[] values
                && !(instance instanceof Object[][])) {
            fillObjectVector(record, offset, values, layout, reusable, state);
        } else if (instance instanceof double[][] values) {
            fillDoubleMatrix(
                    record, offset, values, options.getEntrySeparator(),
                    reusable, state
            );
        } else if (instance instanceof float[][] values) {
            fillFloatMatrix(
                    record, offset, values, options.getEntrySeparator(),
                    reusable, state
            );
        } else if (instance instanceof Object[][] values) {
            fillObjectMatrix(
                    record, offset, values, options.getEntrySeparator(),
                    reusable, state
            );
        } else {
            throw new IllegalArgumentException(
                    "Unsupported delimited instance type at index "
                            + instanceIndex
                            + ": "
                            + instance.getClass().getTypeName()
            );
        }

        if (options.shouldEmbedLabels()) {
            Object label = options.getLabels().get(instanceIndex);
            if (label == null) {
                throw new IllegalArgumentException(
                        "Embedded label cannot be null at instance "
                                + instanceIndex
                                + "."
                );
            }
            record[options.isTargetFirst() ? 0 : record.length - 1] =
                    label.toString();
        }
    }

    private static void fillDoubleVector(
            String[] record,
            int offset,
            double[] values,
            DatasetWriteOptions.DataLayout layout,
            PreparedReusableParameters reusable,
            PerSeriesStandardizationState state
    ) {
        for (int position = 0; position < values.length; position++) {
            record[offset + position] = Double.toString(
                    inverseVector(
                            values[position], position, layout, reusable, state
                    )
            );
        }
    }

    private static void fillFloatVector(
            String[] record,
            int offset,
            float[] values,
            DatasetWriteOptions.DataLayout layout,
            PreparedReusableParameters reusable,
            PerSeriesStandardizationState state
    ) {
        for (int position = 0; position < values.length; position++) {
            record[offset + position] = Float.toString((float) inverseVector(
                    values[position], position, layout, reusable, state
            ));
        }
    }

    private static void fillObjectVector(
            String[] record,
            int offset,
            Object[] values,
            DatasetWriteOptions.DataLayout layout,
            PreparedReusableParameters reusable,
            PerSeriesStandardizationState state
    ) {
        for (int position = 0; position < values.length; position++) {
            record[offset + position] = formatObjectValue(
                    values[position], position, layout, reusable, state
            );
        }
    }

    private static void fillDoubleMatrix(
            String[] record,
            int offset,
            double[][] matrix,
            String separator,
            PreparedReusableParameters reusable,
            PerSeriesStandardizationState state
    ) {
        for (int dimension = 0; dimension < matrix.length; dimension++) {
            double[] values = matrix[dimension];
            StringBuilder builder = new StringBuilder(
                    Math.max(16, values.length * 12)
            );
            for (int time = 0; time < values.length; time++) {
                if (time > 0) builder.append(separator);
                builder.append(inverseMatrix(
                        values[time], dimension, reusable, state
                ));
            }
            record[offset + dimension] = builder.toString();
        }
    }

    private static void fillFloatMatrix(
            String[] record,
            int offset,
            float[][] matrix,
            String separator,
            PreparedReusableParameters reusable,
            PerSeriesStandardizationState state
    ) {
        for (int dimension = 0; dimension < matrix.length; dimension++) {
            float[] values = matrix[dimension];
            StringBuilder builder = new StringBuilder(
                    Math.max(16, values.length * 8)
            );
            for (int time = 0; time < values.length; time++) {
                if (time > 0) builder.append(separator);
                builder.append((float) inverseMatrix(
                        values[time], dimension, reusable, state
                ));
            }
            record[offset + dimension] = builder.toString();
        }
    }

    private static void fillObjectMatrix(
            String[] record,
            int offset,
            Object[][] matrix,
            String separator,
            PreparedReusableParameters reusable,
            PerSeriesStandardizationState state
    ) {
        for (int dimension = 0; dimension < matrix.length; dimension++) {
            Object[] values = matrix[dimension];
            StringBuilder builder = new StringBuilder(
                    Math.max(16, values.length * 8)
            );
            for (int time = 0; time < values.length; time++) {
                if (time > 0) builder.append(separator);
                Object value = values[time];
                if (value != null) {
                    if (value instanceof Number number) {
                        double transformed = inverseMatrix(
                                number.doubleValue(), dimension, reusable, state
                        );
                        builder.append(value instanceof Float
                                ? Float.toString((float) transformed)
                                : Double.toString(transformed));
                    } else {
                        builder.append(value);
                    }
                }
            }
            record[offset + dimension] = builder.toString();
        }
    }

    private static double inverseVector(
            double value,
            int position,
            DatasetWriteOptions.DataLayout layout,
            PreparedReusableParameters reusable,
            PerSeriesStandardizationState state
    ) {
        value = requireWritableNumeric(value);
        if (Double.isNaN(value)) return value;
        if (state != null) {
            return value * state.getScale(0) + state.getCenter(0);
        }
        if (reusable != null) {
            int group = reusable.scope() == StandardizationScope.GLOBAL
                    ? 0 : position;
            return value * reusable.scales()[group]
                    + reusable.centers()[group];
        }
        return value;
    }

    private static double inverseMatrix(
            double value,
            int dimension,
            PreparedReusableParameters reusable,
            PerSeriesStandardizationState state
    ) {
        value = requireWritableNumeric(value);
        if (Double.isNaN(value)) return value;
        if (state != null) {
            int group = state.getParameterGroupCount() == 1 ? 0 : dimension;
            return value * state.getScale(group) + state.getCenter(group);
        }
        if (reusable != null) {
            int group = reusable.scope() == StandardizationScope.GLOBAL
                    ? 0 : dimension;
            return value * reusable.scales()[group]
                    + reusable.centers()[group];
        }
        return value;
    }

    private static String formatObjectValue(
            Object value,
            int position,
            DatasetWriteOptions.DataLayout layout,
            PreparedReusableParameters reusable,
            PerSeriesStandardizationState state
    ) {
        if (value == null) return "";
        if (!(value instanceof Number number)) return value.toString();
        double output = inverseVector(
                number.doubleValue(), position, layout, reusable, state
        );
        return value instanceof Float
                ? Float.toString((float) output)
                : Double.toString(output);
    }

    private static double requireWritableNumeric(double value) {
        if (Double.isInfinite(value)) {
            throw new IllegalArgumentException(
                    "Cannot write an infinite numeric value."
            );
        }
        return value;
    }

    private static void validateDataset(
            List<Object> data,
            Class<?> expectedType,
            DatasetWriteOptions.DataLayout layout,
            DatasetWriteOptions options,
            PreparedReusableParameters reusable
    ) {
        boolean expectedMatrix = isMatrix(data.get(0));
        int firstVectorLength = expectedMatrix
                ? -1
                : vectorLength(data.get(0));
        int firstDimensionCount = expectedMatrix
                ? matrixDimensionCount(data.get(0))
                : -1;

        for (int instanceIndex = 0;
             instanceIndex < data.size();
             instanceIndex++) {
            Object instance = Objects.requireNonNull(
                    data.get(instanceIndex),
                    "Delimited instance cannot be null at index "
                            + instanceIndex
                            + "."
            );
            if (instance.getClass() != expectedType) {
                throw new IllegalArgumentException(
                        "Delimited output cannot mix instance types. First "
                                + "type="
                                + expectedType.getTypeName()
                                + ", instance "
                                + instanceIndex
                                + " type="
                                + instance.getClass().getTypeName()
                                + "."
                );
            }

            if (expectedMatrix) {
                validateMatrix(
                        instance,
                        instanceIndex,
                        firstDimensionCount,
                        options,
                        reusable
                );
            } else {
                int length = vectorLength(instance);
                boolean fixedWidth =
                        layout == DatasetWriteOptions.DataLayout.TABULAR
                                || options.shouldIncludeHeader()
                                || (reusable != null
                                && reusable.scope()
                                == StandardizationScope.PER_DIMENSION);
                if (fixedWidth && length != firstVectorLength) {
                    throw new IllegalArgumentException(
                            "Fixed-width delimited output expected "
                                    + firstVectorLength
                                    + " values, but instance "
                                    + instanceIndex
                                    + " contains "
                                    + length
                                    + "."
                    );
                }
            }
        }
    }

    private static void validateMatrix(
            Object matrix,
            int instanceIndex,
            int expectedDimensionCount,
            DatasetWriteOptions options,
            PreparedReusableParameters reusable
    ) {
        int dimensions = matrixDimensionCount(matrix);
        if (dimensions != expectedDimensionCount) {
            throw new IllegalArgumentException(
                    "Multivariate instance "
                            + instanceIndex
                            + " contains "
                            + dimensions
                            + " dimensions; expected "
                            + expectedDimensionCount
                            + "."
            );
        }
        int expectedLength = -1;
        for (int dimension = 0; dimension < dimensions; dimension++) {
            Object values = Objects.requireNonNull(
                    Array.get(matrix, dimension),
                    "Null dimension "
                            + dimension
                            + " in instance "
                            + instanceIndex
                            + "."
            );
            int length = Array.getLength(values);
            if (length == 0) {
                throw new IllegalArgumentException(
                        "Empty dimension "
                                + dimension
                                + " in instance "
                                + instanceIndex
                                + "."
                );
            }
            if (expectedLength < 0) {
                expectedLength = length;
            } else if (length != expectedLength) {
                throw new IllegalArgumentException(
                        "Row-encoded multivariate instance "
                                + instanceIndex
                                + " has inconsistent dimension lengths."
                );
            }
        }
        if (options.shouldIncludeHeader()
                && options.getFeatureNames().size() != dimensions) {
            throw new IllegalArgumentException(
                    "Multivariate header contains "
                            + options.getFeatureNames().size()
                            + " names, but instances contain "
                            + dimensions
                            + " dimensions."
            );
        }
        if (reusable != null
                && reusable.scope() == StandardizationScope.PER_DIMENSION
                && reusable.centers().length != dimensions) {
            throw new IllegalArgumentException(
                    "PER_DIMENSION statistics contain "
                            + reusable.centers().length
                            + " groups, but multivariate instances contain "
                            + dimensions
                            + " dimensions."
            );
        }
    }

    private static void validateInverseCompatibility(
            Object instance,
            DatasetWriteOptions.DataLayout layout,
            PreparedReusableParameters reusable,
            PerSeriesStandardizationState state,
            int instanceIndex
    ) {
        if (state == null) {
            if (reusable != null
                    && reusable.scope() == StandardizationScope.PER_DIMENSION
                    && !isMatrix(instance)
                    && layout
                    != DatasetWriteOptions.DataLayout.TABULAR) {
                throw new IllegalArgumentException(
                        "Reusable PER_DIMENSION statistics require TABULAR "
                                + "layout for one-dimensional data."
                );
            }
            return;
        }

        if (isMatrix(instance)) {
            int dimensions = matrixDimensionCount(instance);
            int groups = state.getParameterGroupCount();
            if (groups != 1 && groups != dimensions) {
                throw new IllegalArgumentException(
                        "Per-series state "
                                + instanceIndex
                                + " contains "
                                + groups
                                + " groups; expected 1 or "
                                + dimensions
                                + "."
                );
            }
            return;
        }

        if (layout != DatasetWriteOptions.DataLayout.UNIVARIATE_SERIES
                || state.getParameterGroupCount() != 1) {
            throw new IllegalArgumentException(
                    "A one-dimensional per-series inverse state requires "
                            + "UNIVARIATE_SERIES layout and exactly one local "
                            + "parameter group."
            );
        }
    }

    private static int headerFieldCount(
            Object firstInstance,
            DatasetWriteOptions.DataLayout layout,
            DatasetWriteOptions options
    ) {
        int count = isMatrix(firstInstance)
                ? matrixDimensionCount(firstInstance)
                : vectorLength(firstInstance);
        if (options.getFeatureNames().size() != count) {
            throw new IllegalArgumentException(
                    "Header contains "
                            + options.getFeatureNames().size()
                            + " names, but the output record requires "
                            + count
                            + "."
            );
        }
        return count;
    }

    private static void validateLabelCount(
            DatasetWriteOptions options,
            int instanceCount
    ) {
        if (options.shouldEmbedLabels()
                && options.getLabels().size() != instanceCount) {
            throw new IllegalArgumentException(
                    "Embedded-label output requires one label per instance. "
                            + "Labels="
                            + options.getLabels().size()
                            + ", instances="
                            + instanceCount
                            + "."
            );
        }
    }

    private static void validateStateCount(
            List<PerSeriesStandardizationState> states,
            int instanceCount
    ) {
        if (!states.isEmpty() && states.size() != instanceCount) {
            throw new IllegalArgumentException(
                    "Per-series state count "
                            + states.size()
                            + " does not match dataset size "
                            + instanceCount
                            + "."
            );
        }
    }

    private static boolean isSupportedInstance(
            Object instance
    ) {
        if (instance == null) {
            return false;
        }
        return instance instanceof double[]
                || instance instanceof float[]
                || instance instanceof Object[]
                || instance instanceof double[][]
                || instance instanceof float[][]
                || instance instanceof Object[][];
    }

    private static boolean isMatrix(
            Object instance
    ) {
        return instance instanceof double[][]
                || instance instanceof float[][]
                || instance instanceof Object[][];
    }

    private static int vectorLength(
            Object vector
    ) {
        int length = Array.getLength(vector);
        if (length == 0) {
            throw new IllegalArgumentException(
                    "Delimited vector instances cannot be empty."
            );
        }
        return length;
    }

    private static int matrixDimensionCount(
            Object matrix
    ) {
        int dimensions = Array.getLength(matrix);
        if (dimensions == 0) {
            throw new IllegalArgumentException(
                    "Delimited matrix instances cannot contain zero "
                            + "dimensions."
            );
        }
        return dimensions;
    }

    private static void createParentDirectories(
            Path outputPath
    ) throws IOException {
        Path parent = outputPath.toAbsolutePath().getParent();
        if (parent != null) {
            Files.createDirectories(parent);
        }
    }

    private record PreparedReusableParameters(
            StandardizationScope scope,
            double[] centers,
            double[] scales
    ) {
        private static PreparedReusableParameters from(
                StandardizationStats stats
        ) {
            if (stats == null) {
                return null;
            }
            StandardizationScope scope = stats.getScope();
            if (!scope.usesTrainingStatistics()) {
                throw new IllegalArgumentException(
                        "Reusable statistics must use GLOBAL or "
                                + "PER_DIMENSION scope, but received "
                                + scope
                                + "."
                );
            }
            double[] centers = stats.getCenters();
            double[] scales = stats.getScales();
            if (centers.length == 0 || centers.length != scales.length) {
                throw new IllegalArgumentException(
                        "Reusable centers and scales must have equal, nonzero "
                                + "lengths."
                );
            }
            if (scope == StandardizationScope.GLOBAL
                    && centers.length != 1) {
                throw new IllegalArgumentException(
                        "GLOBAL inverse transformation requires one parameter "
                                + "group."
                );
            }
            return new PreparedReusableParameters(
                    scope,
                    centers,
                    scales
            );
        }
    }
}
