package datasets.readers;

import ch.randelshofer.fastdoubleparser.JavaDoubleParser;
import ch.randelshofer.fastdoubleparser.JavaFloatParser;
import core.AppContext;
import datasets.ListObjectDataset;
import datasets.NumericStorageType;
import dev.hardwood.InputFile;
import dev.hardwood.reader.ParquetFileReader;
import dev.hardwood.reader.RowReader;
import dev.hardwood.schema.ColumnProjection;
import org.apache.commons.lang3.time.DurationFormatUtils;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * General row-oriented reader for Parquet tabular and long-format datasets.
 *
 * <p>When {@code idColumn} is absent, each Parquet record becomes one
 * one-dimensional observation. This supports tabular records and other
 * row-wise feature vectors. When {@code idColumn} is configured, records are
 * grouped into observations. Within a group, a configured time column controls
 * ordering; otherwise physical Parquet record order is retained.</p>
 *
 * <p>Numeric output is always primitive. FLOAT32 produces {@code float[]} or
 * {@code float[][]}; FLOAT64 produces {@code double[]} or
 * {@code double[][]}. Parquet nulls become primitive NaN when missing numeric
 * values are enabled. Generic output uses {@code Object[]} or
 * {@code Object[][]}, with null preserved for missing values.</p>
 *
 * <p>This general reader intentionally uses Hardwood's projected
 * {@link RowReader} API because it supports heterogeneous feature, label, ID,
 * and time types. Numeric workloads that match the optimized schema should use
 * {@link NumericLongFormatParquetReader}, which consumes Hardwood column
 * batches and avoids one row object and one feature array per record.</p>
 *
 * <p>Multiple label columns are preserved as immutable lists for multi-label
 * and multi-target workflows. Repeated group labels must remain consistent.</p>
 *
 * <p>A configured time column currently requires nonmissing values. Hybrid
 * ordering for partially missing time coordinates is tracked as pre-v1 task
 * LONG-TIME-01.</p>
 */
public class LongFormatParquetReader implements DatasetReader {
    private final String dataFileName;
    private final boolean isNumeric;
    private final boolean hasMissingValues;
    private final boolean isRegression;
    private final NumericStorageType numericStorageType;
    private final String idColumn;
    private final String timeColumn;
    private final List<String> featureColumns;
    private final List<String> labelColumns;
    private final Set<String> missingIndicators;
    private final ColumnProjection projection;

    public LongFormatParquetReader(ReaderOptions options) {
        this(
                requireOptions(options).getDataPath(),
                options.isNumeric(),
                options.hasMissingValues(),
                options.isRegression(),
                options.getIdColumn(),
                options.getTimeColumn(),
                options.getFeatureColumns(),
                options.getLabelColumns(),
                options.getNumericStorageType()
        );
    }

    public LongFormatParquetReader(
            String dataFileName,
            boolean isNumeric,
            boolean hasMissingValues,
            boolean isRegression,
            String idColumn,
            String timeColumn,
            List<String> featureColumns,
            List<String> labelColumns
    ) {
        this(dataFileName, isNumeric, hasMissingValues, isRegression,
                idColumn, timeColumn, featureColumns, labelColumns,
                NumericStorageType.AUTO);
    }

    public LongFormatParquetReader(
            String dataFileName,
            boolean isNumeric,
            boolean hasMissingValues,
            boolean isRegression,
            String idColumn,
            String timeColumn,
            List<String> featureColumns,
            List<String> labelColumns,
            NumericStorageType numericStorageType
    ) {
        this.dataFileName = requireNonblank(dataFileName, "dataFileName");
        this.isNumeric = isNumeric;
        this.hasMissingValues = hasMissingValues;
        this.isRegression = isRegression;
        this.numericStorageType = resolveStorageType(numericStorageType);
        this.idColumn = normalizeNullableString(idColumn);
        this.timeColumn = normalizeNullableString(timeColumn);
        this.featureColumns = copyColumns(featureColumns, "featureColumns", false);
        this.labelColumns = copyColumns(labelColumns, "labelColumns", true);
        this.missingIndicators = snapshotMissingIndicators();
        validateOptions();
        this.projection = buildColumnProjection();
    }

    @Override
    public ListObjectDataset read() throws IOException {
        return idColumn == null
                ? readRowWiseDataset()
                : readGroupedLongFormatDataset();
    }

    public ListObjectDataset readGroupedLongFormatDataset() throws IOException {
        if (idColumn == null) {
            throw new IllegalStateException(
                    "Grouped long-format Parquet reading requires idColumn.");
        }
        long start = System.nanoTime();
        Path path = validateFile();
        Map<Object, List<LongParquetRow>> groupedRows = new LinkedHashMap<>();
        int rowNumber = 0;

        try (ParquetFileReader fileReader =
                     ParquetFileReader.open(InputFile.of(path));
             RowReader reader = fileReader.buildRowReader()
                     .projection(projection)
                     .build()) {
            while (reader.hasNext()) {
                reader.next();
                Object id = normalizeIdentifier(getValue(reader, idColumn));
                if (id == null) {
                    throw new IllegalArgumentException(
                            "Encountered a missing ID in Parquet column '"
                                    + idColumn + "' at record " + rowNumber + ".");
                }

                Object time = null;
                if (timeColumn != null) {
                    time = normalizeTime(getValue(reader, timeColumn));
                    if (time == null) {
                        throw new IllegalArgumentException(
                                "Encountered a missing configured time in Parquet column '"
                                        + timeColumn + "' at record " + rowNumber + ".");
                    }
                }

                Object[] features = readFeatureValues(reader, rowNumber);
                Object label = readLabelValues(reader);
                groupedRows.computeIfAbsent(id, ignored -> new ArrayList<>())
                        .add(new LongParquetRow(
                                id, time, features, label, rowNumber));
                ProgressLogger.logProgress(rowNumber);
                rowNumber++;
            }
        } catch (IllegalArgumentException e) {
            throw e;
        } catch (RuntimeException e) {
            throw new IOException(
                    "Failed while reading long-format Parquet file: " + path, e);
        }

        if (rowNumber == 0 || groupedRows.isEmpty()) {
            throw new IOException(
                    "Long-format Parquet file contains no records: " + path);
        }

        ListObjectDataset dataset = buildGroupedDataset(groupedRows);
        ProgressLogger.logDuration(start, System.nanoTime());
        return dataset;
    }

    private ListObjectDataset readRowWiseDataset() throws IOException {
        long start = System.nanoTime();
        Path path = validateFile();
        ListObjectDataset dataset = new ListObjectDataset();
        int rowIndex = 0;
        int commonLength = -1;
        boolean unequalLengths = false;

        try (ParquetFileReader fileReader =
                     ParquetFileReader.open(InputFile.of(path));
             RowReader reader = fileReader.buildRowReader()
                     .projection(projection)
                     .build()) {
            while (reader.hasNext()) {
                reader.next();
                Object[] features = readFeatureValues(reader, rowIndex);
                Object label = readLabelValues(reader);
                Object data = materializeRow(features);
                dataset.add(label, data, rowIndex);

                int length = dataLength(data);
                if (commonLength < 0) {
                    commonLength = length;
                } else if (length != commonLength) {
                    unequalLengths = true;
                }
                ProgressLogger.logProgress(rowIndex);
                rowIndex++;
            }
        } catch (IllegalArgumentException e) {
            throw e;
        } catch (RuntimeException e) {
            throw new IOException(
                    "Failed while reading row-wise Parquet file: " + path, e);
        }

        if (rowIndex == 0) {
            throw new IOException(
                    "Row-wise Parquet file contains no records: " + path);
        }

        applyDatasetLength(dataset, commonLength, unequalLengths);
        ProgressLogger.logDuration(start, System.nanoTime());
        return dataset;
    }

    private Object[] readFeatureValues(RowReader reader, int recordIndex) {
        Object[] features = new Object[featureColumns.size()];
        for (int index = 0; index < featureColumns.size(); index++) {
            String column = featureColumns.get(index);
            Object value = normalizeValue(getValue(reader, column));
            features[index] = parseFeatureValue(value, column, recordIndex);
        }
        return features;
    }

    private Object parseFeatureValue(
            Object value,
            String column,
            int recordIndex
    ) {
        if (isMissingValue(value)) {
            if (!hasMissingValues) {
                throw new IllegalArgumentException(
                        "Encountered a missing value in Parquet feature column '"
                                + column + "' at record " + recordIndex
                                + ", but hasMissingValues=false.");
            }
            return null;
        }
        if (!isNumeric) {
            return parseGenericValue(value);
        }
        return numericStorageType == NumericStorageType.FLOAT32
                ? toFloat(value, column, recordIndex)
                : toDouble(value, column, recordIndex);
    }

    private Object readLabelValues(RowReader reader) {
        if (labelColumns.isEmpty()) {
            return null;
        }
        if (labelColumns.size() == 1) {
            return parseLabelValue(normalizeValue(
                    getValue(reader, labelColumns.get(0))));
        }
        List<Object> labels = new ArrayList<>(labelColumns.size());
        for (String column : labelColumns) {
            labels.add(parseLabelValue(normalizeValue(
                    getValue(reader, column))));
        }
        return Collections.unmodifiableList(labels);
    }

    private Object parseLabelValue(Object value) {
        if (isMissingValue(value)) {
            return null;
        }
        if (isRegression) {
            return toDouble(value, "label", -1);
        }
        if (value instanceof Byte || value instanceof Short
                || value instanceof Integer) {
            return ((Number) value).intValue();
        }
        if (value instanceof Long integral) {
            return integral >= Integer.MIN_VALUE && integral <= Integer.MAX_VALUE
                    ? integral.intValue()
                    : integral;
        }
        if (value instanceof Number number) {
            return normalizeNumericLabel(number.doubleValue());
        }
        String token = value.toString().trim();
        try {
            return Integer.parseInt(token);
        } catch (NumberFormatException ignored) {
            try {
                return normalizeNumericLabel(
                        JavaDoubleParser.parseDouble(token));
            } catch (NumberFormatException ignoredAgain) {
                return token;
            }
        }
    }

    private static Object normalizeNumericLabel(double value) {
        if (value == Math.rint(value)
                && value >= Integer.MIN_VALUE
                && value <= Integer.MAX_VALUE) {
            return (int) value;
        }
        return value;
    }

    private ListObjectDataset buildGroupedDataset(
            Map<Object, List<LongParquetRow>> groups
    ) {
        ListObjectDataset dataset = new ListObjectDataset(groups.size());
        int instanceIndex = 0;
        int commonLength = -1;
        boolean unequalLengths = false;

        for (List<LongParquetRow> rows : groups.values()) {
            sortRows(rows);
            Object label = inferGroupLabel(rows);
            Object data = materializeSeries(rows);
            dataset.add(label, data, instanceIndex++);

            int length = dataLength(data);
            if (commonLength < 0) {
                commonLength = length;
            } else if (length != commonLength) {
                unequalLengths = true;
            }
        }

        applyDatasetLength(dataset, commonLength, unequalLengths);
        return dataset;
    }

    private void sortRows(List<LongParquetRow> rows) {
        if (timeColumn == null || rows.size() < 2) {
            return;
        }
        rows.sort((first, second) -> {
            int comparison = compareTimeValues(
                    first.timeValue, second.timeValue);
            return comparison != 0
                    ? comparison
                    : Integer.compare(first.inputOrder, second.inputOrder);
        });
    }

    @SuppressWarnings({"rawtypes", "unchecked"})
    private static int compareTimeValues(Object first, Object second) {
        if (first instanceof Number && second instanceof Number) {
            return Double.compare(
                    ((Number) first).doubleValue(),
                    ((Number) second).doubleValue());
        }
        if (first instanceof Comparable
                && first.getClass().isInstance(second)) {
            return ((Comparable) first).compareTo(second);
        }
        return first.toString().compareTo(second.toString());
    }

    private Object inferGroupLabel(List<LongParquetRow> rows) {
        Object label = rows.get(0).label;
        for (LongParquetRow row : rows) {
            if (!Objects.equals(label, row.label)) {
                throw new IllegalArgumentException(
                        "Inconsistent labels in long-format Parquet group for ID: "
                                + row.id);
            }
        }
        return label;
    }

    private Object materializeRow(Object[] features) {
        if (!isNumeric) {
            return features.clone();
        }
        if (numericStorageType == NumericStorageType.FLOAT32) {
            float[] values = new float[features.length];
            for (int index = 0; index < features.length; index++) {
                values[index] = primitiveFloatOrNaN(features[index]);
            }
            return values;
        }
        double[] values = new double[features.length];
        for (int index = 0; index < features.length; index++) {
            values[index] = primitiveDoubleOrNaN(features[index]);
        }
        return values;
    }

    private Object materializeSeries(List<LongParquetRow> rows) {
        int length = rows.size();
        int dimensions = featureColumns.size();
        if (dimensions == 1) {
            return materializeUnivariate(rows, length);
        }
        return materializeMultivariate(rows, dimensions, length);
    }

    private Object materializeUnivariate(
            List<LongParquetRow> rows,
            int length
    ) {
        if (!isNumeric) {
            Object[] values = new Object[length];
            for (int time = 0; time < length; time++) {
                values[time] = rows.get(time).featureValues[0];
            }
            return values;
        }
        if (numericStorageType == NumericStorageType.FLOAT32) {
            float[] values = new float[length];
            for (int time = 0; time < length; time++) {
                values[time] = primitiveFloatOrNaN(
                        rows.get(time).featureValues[0]);
            }
            return values;
        }
        double[] values = new double[length];
        for (int time = 0; time < length; time++) {
            values[time] = primitiveDoubleOrNaN(
                    rows.get(time).featureValues[0]);
        }
        return values;
    }

    private Object materializeMultivariate(
            List<LongParquetRow> rows,
            int dimensions,
            int length
    ) {
        if (!isNumeric) {
            Object[][] values = new Object[dimensions][length];
            for (int time = 0; time < length; time++) {
                Object[] row = rows.get(time).featureValues;
                for (int dimension = 0;
                     dimension < dimensions;
                     dimension++) {
                    values[dimension][time] = row[dimension];
                }
            }
            return values;
        }
        if (numericStorageType == NumericStorageType.FLOAT32) {
            float[][] values = new float[dimensions][length];
            for (int time = 0; time < length; time++) {
                Object[] row = rows.get(time).featureValues;
                for (int dimension = 0;
                     dimension < dimensions;
                     dimension++) {
                    values[dimension][time] =
                            primitiveFloatOrNaN(row[dimension]);
                }
            }
            return values;
        }
        double[][] values = new double[dimensions][length];
        for (int time = 0; time < length; time++) {
            Object[] row = rows.get(time).featureValues;
            for (int dimension = 0;
                 dimension < dimensions;
                 dimension++) {
                values[dimension][time] =
                        primitiveDoubleOrNaN(row[dimension]);
            }
        }
        return values;
    }

    private static float primitiveFloatOrNaN(Object value) {
        return value == null
                ? Float.NaN
                : ((Number) value).floatValue();
    }

    private static double primitiveDoubleOrNaN(Object value) {
        return value == null
                ? Double.NaN
                : ((Number) value).doubleValue();
    }

    private float toFloat(Object value, String column, int recordIndex) {
        if (value instanceof Number number) {
            return number.floatValue();
        }
        if (value instanceof Boolean booleanValue) {
            return booleanValue ? 1.0f : 0.0f;
        }
        try {
            return JavaFloatParser.parseFloat(value.toString().trim());
        } catch (NumberFormatException e) {
            throw numericConversionFailure(
                    value, column, recordIndex, "FLOAT32", e);
        }
    }

    private double toDouble(Object value, String column, int recordIndex) {
        if (value instanceof Number number) {
            return number.doubleValue();
        }
        if (value instanceof Boolean booleanValue) {
            return booleanValue ? 1.0d : 0.0d;
        }
        try {
            return JavaDoubleParser.parseDouble(value.toString().trim());
        } catch (NumberFormatException e) {
            throw numericConversionFailure(
                    value, column, recordIndex, "FLOAT64", e);
        }
    }

    private static IllegalArgumentException numericConversionFailure(
            Object value,
            String column,
            int recordIndex,
            String targetType,
            Exception cause
    ) {
        String location = recordIndex >= 0
                ? " at record " + recordIndex
                : "";
        return new IllegalArgumentException(
                "Could not convert Parquet value '" + value + "' from column '"
                        + column + "'" + location + " to " + targetType + ".",
                cause);
    }

    private Object parseGenericValue(Object value) {
        if (value == null) {
            return null;
        }
        if (!(value instanceof CharSequence)) {
            return value;
        }
        String token = value.toString().trim();
        if (isMissingToken(token)) {
            return null;
        }
        try {
            return JavaDoubleParser.parseDouble(token);
        } catch (NumberFormatException ignored) {
            if (token.equalsIgnoreCase("true")
                    || token.equalsIgnoreCase("false")) {
                return Boolean.parseBoolean(token);
            }
            return token;
        }
    }

    private Object normalizeIdentifier(Object value) {
        Object normalized = normalizeValue(value);
        if (isMissingValue(normalized)) {
            return null;
        }
        // Preserve Parquet's typed identifier semantics. String IDs remain
        // strings, while integral IDs retain their integral source type.
        return normalized instanceof CharSequence
                ? normalized.toString()
                : normalized;
    }

    private Object normalizeTime(Object value) {
        Object normalized = normalizeValue(value);
        return isMissingValue(normalized) ? null : normalized;
    }

    private static Object normalizeValue(Object value) {
        if (value instanceof byte[] bytes) {
            return new String(bytes, StandardCharsets.UTF_8);
        }
        if (value instanceof Character character) {
            return character.toString();
        }
        return value;
    }

    private boolean isMissingValue(Object value) {
        if (value == null) {
            return true;
        }
        return value instanceof CharSequence
                && isMissingToken(value.toString());
    }

    private boolean isMissingToken(String token) {
        if (token == null) {
            return true;
        }
        String trimmed = token.trim();
        return trimmed.isEmpty()
                || missingIndicators.contains(
                trimmed.toUpperCase(Locale.ROOT));
    }

    private ColumnProjection buildColumnProjection() {
        List<String> columns = new ArrayList<>();
        Set<String> used = new HashSet<>();
        addProjectedColumn(columns, used, idColumn);
        addProjectedColumn(columns, used, timeColumn);
        for (String column : featureColumns) {
            addProjectedColumn(columns, used, column);
        }
        for (String column : labelColumns) {
            addProjectedColumn(columns, used, column);
        }
        return ColumnProjection.columns(columns.toArray(String[]::new));
    }

    private static void addProjectedColumn(
            List<String> columns,
            Set<String> used,
            String column
    ) {
        if (column != null && used.add(column)) {
            columns.add(column);
        }
    }

    private void validateOptions() {
        if (featureColumns.isEmpty()) {
            throw new IllegalArgumentException(
                    "LongFormatParquetReader requires at least one feature column.");
        }
        if (idColumn == null && timeColumn != null) {
            throw new IllegalArgumentException(
                    "timeColumn requires grouped mode with idColumn.");
        }
        Set<String> roles = new HashSet<>();
        addUniqueRole(roles, idColumn, "idColumn");
        addUniqueRole(roles, timeColumn, "timeColumn");
        for (String column : featureColumns) {
            addUniqueRole(roles, column, "featureColumns");
        }
        for (String column : labelColumns) {
            addUniqueRole(roles, column, "labelColumns");
        }
    }

    private static void addUniqueRole(
            Set<String> roles,
            String column,
            String role
    ) {
        if (column != null && !roles.add(column)) {
            throw new IllegalArgumentException(
                    "Parquet column is assigned to multiple roles: "
                            + column + " (detected while validating " + role + ").");
        }
    }

    private Path validateFile() throws IOException {
        Path path = Path.of(dataFileName);
        if (!Files.exists(path)) {
            throw new IOException(
                    "Parquet data file does not exist: " + path);
        }
        if (!Files.isRegularFile(path)) {
            throw new IOException(
                    "Parquet data path is not a regular file: " + path);
        }
        if (!Files.isReadable(path)) {
            throw new IOException(
                    "Parquet data file is not readable: " + path);
        }
        return path;
    }

    private static Object getValue(RowReader reader, String column) {
        return reader.isNull(column) ? null : reader.getValue(column);
    }

    private static int dataLength(Object data) {
        if (data instanceof double[] values) {
            return values.length;
        }
        if (data instanceof float[] values) {
            return values.length;
        }
        if (data instanceof Object[] values) {
            return values.length;
        }
        if (data instanceof double[][] values) {
            return values.length == 0 ? 0 : values[0].length;
        }
        if (data instanceof float[][] values) {
            return values.length == 0 ? 0 : values[0].length;
        }
        if (data instanceof Object[][] values) {
            return values.length == 0 ? 0 : values[0].length;
        }
        throw new IllegalArgumentException(
                "Unsupported Parquet observation representation: "
                        + data.getClass().getName());
    }

    private static void applyDatasetLength(
            ListObjectDataset dataset,
            int commonLength,
            boolean unequalLengths
    ) {
        int length = unequalLengths ? 0 : Math.max(commonLength, 0);
        dataset.setLength(length);
        AppContext.length = length;
    }

    private static ReaderOptions requireOptions(ReaderOptions options) {
        if (options == null) {
            throw new IllegalArgumentException(
                    "LongFormatParquetReader requires non-null ReaderOptions.");
        }
        return options;
    }

    private static NumericStorageType resolveStorageType(
            NumericStorageType requested
    ) {
        NumericStorageType value = Objects.requireNonNull(
                requested, "NumericStorageType cannot be null.");
        // The general row reader does not inspect physical Parquet types in
        // advance, so AUTO intentionally uses the safe common FLOAT64 type.
        return value == NumericStorageType.AUTO
                ? NumericStorageType.FLOAT64
                : value;
    }

    private static String requireNonblank(String value, String role) {
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException(
                    "LongFormatParquetReader requires " + role + ".");
        }
        return value.trim();
    }

    private static String normalizeNullableString(String value) {
        if (value == null) {
            return null;
        }
        String trimmed = value.trim();
        return trimmed.isEmpty() || trimmed.equalsIgnoreCase("None")
                ? null
                : trimmed;
    }

    private static List<String> copyColumns(
            List<String> columns,
            String role,
            boolean allowEmpty
    ) {
        if (columns == null || columns.isEmpty()) {
            if (allowEmpty) {
                return List.of();
            }
            throw new IllegalArgumentException(
                    "LongFormatParquetReader requires at least one "
                            + role + " entry.");
        }
        List<String> copy = new ArrayList<>(columns.size());
        Set<String> used = new HashSet<>();
        for (String column : columns) {
            if (column == null || column.isBlank()) {
                throw new IllegalArgumentException(
                        role + " cannot contain null or blank column names.");
            }
            String normalized = column.trim();
            if (!used.add(normalized)) {
                throw new IllegalArgumentException(
                        role + " contains a duplicate column: " + normalized);
            }
            copy.add(normalized);
        }
        return Collections.unmodifiableList(copy);
    }

    private static Set<String> snapshotMissingIndicators() {
        if (AppContext.MissingStrings == null
                || AppContext.MissingStrings.isEmpty()) {
            return Set.of();
        }
        Set<String> values = new HashSet<>();
        for (String indicator : AppContext.MissingStrings) {
            if (indicator == null) {
                continue;
            }
            String normalized = indicator.trim();
            if (!normalized.isEmpty()) {
                values.add(normalized.toUpperCase(Locale.ROOT));
            }
        }
        return values.isEmpty()
                ? Set.of()
                : Collections.unmodifiableSet(values);
    }

    private static final class LongParquetRow {
        private final Object id;
        private final Object timeValue;
        private final Object[] featureValues;
        private final Object label;
        private final int inputOrder;

        private LongParquetRow(
                Object id,
                Object timeValue,
                Object[] featureValues,
                Object label,
                int inputOrder
        ) {
            this.id = id;
            this.timeValue = timeValue;
            this.featureValues = featureValues;
            this.label = label;
            this.inputOrder = inputOrder;
        }
    }

    public static class ProgressLogger {
        public static void logProgress(int index) {
            if (index % 1000 != 0) {
                return;
            }
            if (index % 100000 == 0) {
                System.out.print("\n");
                if (index % 1000000 == 0) {
                    long usedMemory = AppContext.runtime.totalMemory()
                            - AppContext.runtime.freeMemory();
                    System.out.print(index + ":"
                            + usedMemory / 1024 / 1024 + "mb\n");
                }
                return;
            }
            System.out.print(".");
        }

        public static void logDuration(long start, long end) {
            String duration = DurationFormatUtils.formatDuration(
                    (long) ((end - start) / 1e6), "H:m:s.SSS");
            System.out.println("finished in " + duration);
        }
    }
}
