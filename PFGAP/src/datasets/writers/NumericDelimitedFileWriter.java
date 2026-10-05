package datasets.writers;

import datasets.ListObjectDataset;
import de.siegmar.fastcsv.writer.CsvWriter;
import preprocessing.standardization.PerSeriesStandardizationState;
import preprocessing.standardization.StandardizationScope;
import preprocessing.standardization.StandardizationStats;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.List;
import java.util.Objects;

/**
 * FastCSV-backed writer for row-oriented primitive numeric data.
 *
 * <p>Supported materialized instance types are {@code double[]} and
 * {@code float[]}. A row may mean either a fixed-width tabular instance or a
 * potentially variable-length univariate series; that distinction is supplied
 * explicitly through {@link DatasetWriteOptions.DataLayout}.</p>
 *
 * <p>Optional inverse standardization is fused with record construction. The
 * writer does not mutate the source arrays and does not construct a complete
 * inverse-transformed dataset copy.</p>
 */
public final class NumericDelimitedFileWriter implements DatasetWriter {

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
                    "Cannot write an empty numeric delimited dataset."
            );
        }
        if (!supports(dataset, options)) {
            Object first = data.get(0);
            throw new IllegalArgumentException(
                    "NUMERIC_DELIMITED output requires homogeneous double[] "
                            + "or float[] instances, but received "
                            + (first == null
                            ? "null"
                            : first.getClass().getTypeName())
                            + "."
            );
        }

        DatasetWriteOptions.DataLayout layout = resolveLayout(
                data,
                options
        );
        Class<?> rowType = data.get(0).getClass();
        int firstValueCount = valueCount(data.get(0), 0);
        PreparedReusableParameters reusable =
                PreparedReusableParameters.from(
                        options.getReusableStatistics()
                );
        List<PerSeriesStandardizationState> states =
                options.getPerSeriesStates();

        validateStateCount(states, data.size());
        validateLabelCount(options, data.size());
        validateHeader(options, layout, firstValueCount);

        boolean fixedWidthRequired =
                layout == DatasetWriteOptions.DataLayout.TABULAR
                        || options.shouldIncludeHeader()
                        || (reusable != null
                        && reusable.scope()
                        == StandardizationScope.PER_DIMENSION);

        validateRows(
                data,
                rowType,
                firstValueCount,
                fixedWidthRequired
        );
        createParentDirectories(options.getOutputPath());

        try (CsvWriter writer = CsvWriter.builder()
                .fieldSeparator(options.getEntrySeparatorCharacter())
                .build(
                        options.getOutputPath(),
                        StandardCharsets.UTF_8,
                        StandardOpenOption.CREATE,
                        StandardOpenOption.TRUNCATE_EXISTING,
                        StandardOpenOption.WRITE
                )) {
            String[] record = null;

            if (options.shouldIncludeHeader()) {
                record = new String[
                        firstValueCount
                                + (options.shouldEmbedLabels() ? 1 : 0)
                ];
                fillHeader(record, options, firstValueCount);
                writer.writeRecord(record);
            }

            for (int instanceIndex = 0;
                 instanceIndex < data.size();
                 instanceIndex++) {
                Object row = data.get(instanceIndex);
                int valueCount = valueCount(row, instanceIndex);
                int fieldCount = valueCount
                        + (options.shouldEmbedLabels() ? 1 : 0);

                if (record == null || record.length != fieldCount) {
                    record = new String[fieldCount];
                }

                PerSeriesStandardizationState state = states.isEmpty()
                        ? null
                        : states.get(instanceIndex);
                validateInverseCompatibility(
                        valueCount,
                        layout,
                        reusable,
                        state,
                        instanceIndex
                );
                fillRecord(
                        record,
                        row,
                        instanceIndex,
                        valueCount,
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
        if (!(first instanceof double[]) && !(first instanceof float[])) {
            return false;
        }
        Class<?> expectedType = first.getClass();
        for (Object row : dataset.getData()) {
            if (row == null || row.getClass() != expectedType) {
                return false;
            }
        }
        return options.getDataLayout()
                != DatasetWriteOptions.DataLayout.MULTIVARIATE_SERIES
                && options.getDataLayout()
                != DatasetWriteOptions.DataLayout.LONG_FORM;
    }

    @Override
    public String formatName() {
        return "NUMERIC_DELIMITED";
    }

    private static DatasetWriteOptions.DataLayout resolveLayout(
            List<Object> data,
            DatasetWriteOptions options
    ) {
        DatasetWriteOptions.DataLayout configured = options.getDataLayout();
        if (configured != DatasetWriteOptions.DataLayout.AUTO) {
            return configured;
        }

        if (options.shouldIncludeHeader()
                || options.hasFeatureNames()
                || (options.hasReusableStatistics()
                && options.getReusableStatistics().getScope()
                == StandardizationScope.PER_DIMENSION)) {
            return DatasetWriteOptions.DataLayout.TABULAR;
        }

        int firstLength = valueCount(data.get(0), 0);
        for (int index = 1; index < data.size(); index++) {
            if (valueCount(data.get(index), index) != firstLength) {
                return DatasetWriteOptions.DataLayout.UNIVARIATE_SERIES;
            }
        }

        // Equal row lengths alone do not reveal semantics. TABULAR preserves
        // the reader's historical default while callers can explicitly select
        // UNIVARIATE_SERIES for fixed-length series.
        return DatasetWriteOptions.DataLayout.TABULAR;
    }

    private static void fillHeader(
            String[] record,
            DatasetWriteOptions options,
            int valueCount
    ) {
        int offset = options.shouldEmbedLabels() && options.isTargetFirst()
                ? 1
                : 0;
        for (int feature = 0; feature < valueCount; feature++) {
            record[offset + feature] = options.getFeatureNames().get(feature);
        }
        if (options.shouldEmbedLabels()) {
            record[options.isTargetFirst() ? 0 : record.length - 1] =
                    options.getTargetName();
        }
    }

    private static void fillRecord(
            String[] record,
            Object row,
            int instanceIndex,
            int valueCount,
            DatasetWriteOptions options,
            PreparedReusableParameters reusable,
            PerSeriesStandardizationState state
    ) {
        int offset = options.shouldEmbedLabels() && options.isTargetFirst()
                ? 1
                : 0;

        if (row instanceof double[] values) {
            for (int position = 0; position < valueCount; position++) {
                record[offset + position] = Double.toString(
                        inverseIfConfigured(
                                values[position],
                                position,
                                reusable,
                                state
                        )
                );
            }
        } else if (row instanceof float[] values) {
            for (int position = 0; position < valueCount; position++) {
                double output = inverseIfConfigured(
                        values[position],
                        position,
                        reusable,
                        state
                );
                record[offset + position] = Float.toString((float) output);
            }
        } else {
            throw unsupportedRowType(row, instanceIndex);
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

    private static double inverseIfConfigured(
            double value,
            int position,
            PreparedReusableParameters reusable,
            PerSeriesStandardizationState state
    ) {
        if (Double.isNaN(value)) {
            return value;
        }
        if (!Double.isFinite(value)) {
            throw new IllegalArgumentException(
                    "Cannot write a nonfinite numeric value other than NaN: "
                            + value
            );
        }

        if (state != null) {
            // A one-dimensional primitive row is one univariate series. Both
            // per-series scopes use one parameter group for this writer.
            return value * state.getScale(0) + state.getCenter(0);
        }
        if (reusable != null) {
            int group = reusable.scope() == StandardizationScope.GLOBAL
                    ? 0
                    : position;
            return value * reusable.scales()[group]
                    + reusable.centers()[group];
        }
        return value;
    }

    private static void validateRows(
            List<Object> data,
            Class<?> expectedType,
            int firstValueCount,
            boolean fixedWidthRequired
    ) {
        for (int instanceIndex = 0;
             instanceIndex < data.size();
             instanceIndex++) {
            Object row = Objects.requireNonNull(
                    data.get(instanceIndex),
                    "Numeric row cannot be null at instance "
                            + instanceIndex
                            + "."
            );
            if (row.getClass() != expectedType) {
                throw new IllegalArgumentException(
                        "Numeric delimited output cannot mix row types. First "
                                + "row type="
                                + expectedType.getTypeName()
                                + ", instance "
                                + instanceIndex
                                + " type="
                                + row.getClass().getTypeName()
                                + "."
                );
            }
            int actual = valueCount(row, instanceIndex);
            if (fixedWidthRequired && actual != firstValueCount) {
                throw new IllegalArgumentException(
                        "Fixed-width numeric delimited output expected "
                                + firstValueCount
                                + " values, but instance "
                                + instanceIndex
                                + " contains "
                                + actual
                                + "."
                );
            }
        }
    }

    private static void validateHeader(
            DatasetWriteOptions options,
            DatasetWriteOptions.DataLayout layout,
            int valueCount
    ) {
        if (!options.shouldIncludeHeader()) {
            return;
        }
        if (layout != DatasetWriteOptions.DataLayout.TABULAR) {
            throw new IllegalArgumentException(
                    "Numeric delimited headers require TABULAR layout."
            );
        }
        if (options.getFeatureNames().size() != valueCount) {
            throw new IllegalArgumentException(
                    "Header contains "
                            + options.getFeatureNames().size()
                            + " feature names, but rows contain "
                            + valueCount
                            + " values."
            );
        }
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

    private static void validateInverseCompatibility(
            int valueCount,
            DatasetWriteOptions.DataLayout layout,
            PreparedReusableParameters reusable,
            PerSeriesStandardizationState state,
            int instanceIndex
    ) {
        if (reusable != null
                && reusable.scope() == StandardizationScope.PER_DIMENSION) {
            if (layout != DatasetWriteOptions.DataLayout.TABULAR) {
                throw new IllegalArgumentException(
                        "Reusable PER_DIMENSION statistics require TABULAR "
                                + "layout for one-dimensional numeric rows."
                );
            }
            if (reusable.centers().length != valueCount) {
                throw new IllegalArgumentException(
                        "Instance "
                                + instanceIndex
                                + " contains "
                                + valueCount
                                + " values, but PER_DIMENSION statistics "
                                + "contain "
                                + reusable.centers().length
                                + " groups."
                );
            }
        }
        if (state != null) {
            if (layout != DatasetWriteOptions.DataLayout.UNIVARIATE_SERIES) {
                throw new IllegalArgumentException(
                        "Per-series inverse states require "
                                + "UNIVARIATE_SERIES layout in the numeric "
                                + "delimited writer."
                );
            }
            if (state.getParameterGroupCount() != 1) {
                throw new IllegalArgumentException(
                        "Univariate instance "
                                + instanceIndex
                                + " requires one local inverse group, but "
                                + "received "
                                + state.getParameterGroupCount()
                                + "."
                );
            }
        }
    }

    private static int valueCount(
            Object row,
            int instanceIndex
    ) {
        Objects.requireNonNull(
                row,
                "Numeric row cannot be null at instance "
                        + instanceIndex
                        + "."
        );
        int count;
        if (row instanceof double[] values) {
            count = values.length;
        } else if (row instanceof float[] values) {
            count = values.length;
        } else {
            throw unsupportedRowType(row, instanceIndex);
        }
        if (count == 0) {
            throw new IllegalArgumentException(
                    "Numeric row cannot be empty at instance "
                            + instanceIndex
                            + "."
            );
        }
        return count;
    }

    private static IllegalArgumentException unsupportedRowType(
            Object row,
            int instanceIndex
    ) {
        return new IllegalArgumentException(
                "Unsupported numeric delimited row type at instance "
                        + instanceIndex
                        + ": "
                        + (row == null
                        ? "null"
                        : row.getClass().getTypeName())
                        + ". Expected double[] or float[]."
        );
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
                        "GLOBAL inverse transformation requires exactly one "
                                + "parameter group."
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
