package datasets.readers;

import ch.randelshofer.fastdoubleparser.JavaDoubleParser;
import core.AppContext;
import datasets.ListObjectDataset;
import datasets.NumericStorageType;
import dev.hardwood.InputFile;
import dev.hardwood.reader.ColumnReader;
import dev.hardwood.reader.ColumnReaders;
import dev.hardwood.reader.ParquetFileReader;
import dev.hardwood.reader.RowReader;
import dev.hardwood.reader.Validity;
import dev.hardwood.schema.ColumnProjection;
import preprocessing.standardization.StandardizationStats;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Objects;
import java.util.Set;

/**
 * Reads selected columns from one ordinary Parquet file as separate
 * one-dimensional PFGAP observations.
 *
 * <pre>
 * one selected Parquet column = one dataset observation
 * one Parquet record          = one position in that observation
 * </pre>
 *
 * <p>This representation is useful when a Parquet table stores one complete
 * univariate series per column. It is not the row-wise tabular interpretation
 * used by {@link LongFormatParquetReader}, and it is not a multivariate
 * observation. Labels are intentionally unsupported because the selected
 * columns themselves are the dataset observations.</p>
 *
 * <p>Numeric mode uses Hardwood's batch-oriented {@link ColumnReaders} API and
 * supports physical DOUBLE, FLOAT, INT64, and INT32 columns. FLOAT32 produces
 * primitive {@code float[]}; FLOAT64 produces primitive {@code double[]}.
 * AUTO preserves FLOAT only when every selected column is physically FLOAT;
 * otherwise all selected observations are promoted to FLOAT64. Parquet nulls
 * become primitive NaN when missing values are enabled. Boxed numeric arrays
 * are never produced.</p>
 *
 * <p>Generic mode uses Hardwood's projected {@link RowReader} compatibility
 * path so logical and heterogeneous values can be returned as
 * {@code Object[]} observations.</p>
 */
public class ParquetColumnSeriesReader implements DatasetReader {
    private static final int DEFAULT_INITIAL_SERIES_CAPACITY = 4096;

    private final String dataFileName;
    private final List<String> featureColumns;
    private final boolean isNumeric;
    private final boolean hasMissingValues;
    private final NumericStorageType requestedStorageType;
    private final int initialSeriesCapacity;
    private final ColumnProjection projection;
    private final Set<String> missingIndicators;

    public ParquetColumnSeriesReader(ReaderOptions options) {
        this(requireOptions(options).getDataPath(),
                options.getFeatureColumns(), options.isNumeric(),
                options.hasMissingValues(),
                options.getStandardizationStats(),
                DEFAULT_INITIAL_SERIES_CAPACITY,
                options.getNumericStorageType());
        if (options.getLabelColumns() != null
                && !options.getLabelColumns().isEmpty()) {
            throw new IllegalArgumentException(
                    "ParquetColumnSeriesReader does not interpret ordinary "
                            + "Parquet columns as per-series labels. Each "
                            + "selected feature column is one dataset "
                            + "observation. Use separate metadata or another "
                            + "reader for labels.");
        }
    }

    public ParquetColumnSeriesReader(
            String dataFileName,
            List<String> featureColumns,
            boolean isNumeric,
            boolean hasMissingValues,
            StandardizationStats standardizationStats
    ) {
        this(dataFileName, featureColumns, isNumeric, hasMissingValues,
                standardizationStats, DEFAULT_INITIAL_SERIES_CAPACITY,
                NumericStorageType.AUTO);
    }

    public ParquetColumnSeriesReader(
            String dataFileName,
            List<String> featureColumns,
            boolean isNumeric,
            boolean hasMissingValues,
            StandardizationStats standardizationStats,
            int initialSeriesCapacity
    ) {
        this(dataFileName, featureColumns, isNumeric, hasMissingValues,
                standardizationStats, initialSeriesCapacity,
                NumericStorageType.AUTO);
    }

    public ParquetColumnSeriesReader(
            String dataFileName,
            List<String> featureColumns,
            boolean isNumeric,
            boolean hasMissingValues,
            StandardizationStats standardizationStats,
            int initialSeriesCapacity,
            NumericStorageType numericStorageType
    ) {
        this.dataFileName = requireNonblank(dataFileName, "dataFileName");
        this.featureColumns = copyAndValidateFeatureColumns(featureColumns);
        this.isNumeric = isNumeric;
        this.hasMissingValues = hasMissingValues;
        this.requestedStorageType = Objects.requireNonNull(
                numericStorageType, "numericStorageType cannot be null.");
        if (initialSeriesCapacity < 1) {
            throw new IllegalArgumentException(
                    "initialSeriesCapacity must be at least 1. Received: "
                            + initialSeriesCapacity + ".");
        }
        this.initialSeriesCapacity = initialSeriesCapacity;
        validateStandardizationConfiguration(standardizationStats);
        this.missingIndicators = snapshotMissingIndicators();
        this.projection = ColumnProjection.columns(
                this.featureColumns.toArray(String[]::new));
    }

    @Override
    public ListObjectDataset read() throws IOException {
        Path file = validateFile();
        return isNumeric
                ? readNumericColumns(file)
                : readGenericColumns(file);
    }

    private ListObjectDataset readNumericColumns(Path file)
            throws IOException {
        ValueKind[] kinds = null;
        NumericStorageType outputType = null;
        NumericBuffer[] buffers = null;
        MissingBuffer[] missing = hasMissingValues
                ? new MissingBuffer[featureColumns.size()] : null;
        int recordCount = 0;

        try (ParquetFileReader parquet =
                     ParquetFileReader.open(InputFile.of(file));
             ColumnReaders columns = parquet.buildColumnReaders(projection)
                     .build()) {
            while (columns.nextBatch()) {
                int batchCount = columns.getRecordCount();
                if (batchCount == 0) {
                    continue;
                }
                if (kinds == null) {
                    kinds = detectNumericKinds(columns, file);
                    outputType = resolveOutputType(kinds);
                    buffers = createNumericBuffers(outputType);
                    if (missing != null) {
                        for (int i = 0; i < missing.length; i++) {
                            missing[i] = new MissingBuffer(
                                    initialSeriesCapacity);
                        }
                    }
                }

                for (int columnIndex = 0;
                     columnIndex < featureColumns.size();
                     columnIndex++) {
                    String name = featureColumns.get(columnIndex);
                    ColumnReader reader = columns.getColumnReader(name);
                    Object values = readValues(
                            reader, kinds[columnIndex], name, file);
                    Validity validity = reader.getLeafValidity();
                    appendNumericBatch(values, kinds[columnIndex], validity,
                            batchCount, buffers[columnIndex],
                            missing == null ? null : missing[columnIndex],
                            name, recordCount, file);
                }
                recordCount += batchCount;
                ProgressLogger.logProgress(recordCount);
            }
        } catch (IllegalArgumentException e) {
            throw e;
        } catch (RuntimeException e) {
            throw new IOException(
                    "Failed while reading numeric Parquet column series from "
                            + file + ".", e);
        }

        if (recordCount == 0 || buffers == null) {
            throw new IOException(
                    "Parquet file contains no records: " + file);
        }

        ListObjectDataset dataset =
                new ListObjectDataset(featureColumns.size());
        for (int column = 0; column < featureColumns.size(); column++) {
            Object observation = buffers[column].toArray(
                    missing == null ? null : missing[column]);
            dataset.add(null, observation, column);
        }
        dataset.setLength(recordCount);
        AppContext.length = recordCount;
        return dataset;
    }

    private ValueKind[] detectNumericKinds(
            ColumnReaders columns,
            Path file
    ) {
        ValueKind[] kinds = new ValueKind[featureColumns.size()];
        for (int i = 0; i < featureColumns.size(); i++) {
            String name = featureColumns.get(i);
            kinds[i] = detectNumericKind(
                    columns.getColumnReader(name), name, file);
        }
        return kinds;
    }

    private NumericStorageType resolveOutputType(ValueKind[] kinds) {
        if (requestedStorageType != NumericStorageType.AUTO) {
            return requestedStorageType;
        }
        for (ValueKind kind : kinds) {
            if (kind != ValueKind.FLOAT) {
                return NumericStorageType.FLOAT64;
            }
        }
        return NumericStorageType.FLOAT32;
    }

    private NumericBuffer[] createNumericBuffers(
            NumericStorageType outputType
    ) {
        NumericBuffer[] buffers =
                new NumericBuffer[featureColumns.size()];
        for (int i = 0; i < buffers.length; i++) {
            buffers[i] = outputType == NumericStorageType.FLOAT32
                    ? new FloatBuffer(initialSeriesCapacity)
                    : new DoubleBuffer(initialSeriesCapacity);
        }
        return buffers;
    }

    private void appendNumericBatch(
            Object sourceValues,
            ValueKind sourceKind,
            Validity validity,
            int count,
            NumericBuffer destination,
            MissingBuffer missing,
            String column,
            int recordOffset,
            Path file
    ) {
        boolean hasNulls = validity.hasNulls();
        if (!hasNulls) {
            destination.addAll(sourceValues, sourceKind, count);
            if (missing != null) {
                missing.addRepeated(false, count);
            }
            return;
        }

        for (int row = 0; row < count; row++) {
            boolean isNull = validity.isNull(row);
            if (isNull && !hasMissingValues) {
                throw new IllegalArgumentException(
                        "Null value in Parquet column '" + column
                                + "' at record " + (recordOffset + row)
                                + " in " + file
                                + ", but hasMissingValues=false.");
            }
            destination.add(sourceValues, sourceKind, row, isNull);
            if (missing != null) {
                missing.add(isNull);
            }
        }
    }

    private ListObjectDataset readGenericColumns(Path file)
            throws IOException {
        ObjectBuffer[] buffers = new ObjectBuffer[featureColumns.size()];
        for (int i = 0; i < buffers.length; i++) {
            buffers[i] = new ObjectBuffer(initialSeriesCapacity);
        }
        int recordCount = 0;

        try (ParquetFileReader parquet =
                     ParquetFileReader.open(InputFile.of(file));
             RowReader reader = parquet.buildRowReader()
                     .projection(projection)
                     .build()) {
            while (reader.hasNext()) {
                reader.next();
                for (int column = 0;
                     column < featureColumns.size();
                     column++) {
                    String name = featureColumns.get(column);
                    Object raw = reader.isNull(name)
                            ? null : reader.getValue(name);
                    Object value = parseGenericValue(raw);
                    if (value == null && !hasMissingValues) {
                        throw new IllegalArgumentException(
                                "Missing value in Parquet column '" + name
                                        + "' at record " + recordCount
                                        + ", but hasMissingValues=false.");
                    }
                    buffers[column].add(value);
                }
                recordCount++;
                ProgressLogger.logProgress(recordCount);
            }
        } catch (IllegalArgumentException e) {
            throw e;
        } catch (RuntimeException e) {
            throw new IOException(
                    "Failed while reading generic Parquet column series from "
                            + file + ".", e);
        }

        if (recordCount == 0) {
            throw new IOException(
                    "Parquet file contains no records: " + file);
        }
        ListObjectDataset dataset =
                new ListObjectDataset(featureColumns.size());
        for (int column = 0; column < buffers.length; column++) {
            dataset.add(null, buffers[column].toArray(), column);
        }
        dataset.setLength(recordCount);
        AppContext.length = recordCount;
        return dataset;
    }

    private Object parseGenericValue(Object value) {
        if (value == null) {
            return null;
        }
        if (value instanceof byte[] bytes) {
            value = new String(bytes, StandardCharsets.UTF_8);
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

    private boolean isMissingToken(String token) {
        if (token == null) {
            return true;
        }
        String trimmed = token.trim();
        return trimmed.isEmpty()
                || missingIndicators.contains(
                trimmed.toUpperCase(Locale.ROOT));
    }

    private void validateStandardizationConfiguration(
            StandardizationStats standardizationStats
    ) {
        if (standardizationStats != null && !isNumeric) {
            throw new IllegalArgumentException(
                    "Standardization statistics cannot be used with "
                            + "ParquetColumnSeriesReader when isNumeric=false.");
        }
        // Selected columns are observations, not dimensions. Feature-name
        // compatibility across selected columns is therefore not meaningful.
    }

    private Path validateFile() throws IOException {
        Path file = Path.of(dataFileName);
        if (!Files.isRegularFile(file) || !Files.isReadable(file)) {
            throw new IOException(
                    "Parquet file is not a readable regular file: " + file);
        }
        return file;
    }

    private static ValueKind detectNumericKind(
            ColumnReader reader,
            String column,
            Path file
    ) {
        RuntimeException last = null;
        for (ValueKind kind : ValueKind.values()) {
            try {
                readValues(reader, kind, column, file);
                return kind;
            } catch (RuntimeException e) {
                last = e;
            }
        }
        throw new IllegalArgumentException(
                "Parquet column '" + column + "' in " + file
                        + " is not a supported fixed-width numeric column. "
                        + "Supported physical types are DOUBLE, FLOAT, "
                        + "INT64, and INT32.", last);
    }

    private static Object readValues(
            ColumnReader reader,
            ValueKind kind,
            String column,
            Path file
    ) {
        try {
            return switch (kind) {
                case DOUBLE -> reader.getDoubles();
                case FLOAT -> reader.getFloats();
                case LONG -> reader.getLongs();
                case INT -> reader.getInts();
            };
        } catch (RuntimeException e) {
            throw new IllegalArgumentException(
                    "Parquet column '" + column + "' in " + file
                            + " is incompatible with " + kind + ".", e);
        }
    }

    private static ReaderOptions requireOptions(ReaderOptions options) {
        if (options == null) {
            throw new IllegalArgumentException(
                    "ParquetColumnSeriesReader requires ReaderOptions.");
        }
        return options;
    }

    private static String requireNonblank(String value, String role) {
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException(role + " cannot be blank.");
        }
        return value.trim();
    }

    private static List<String> copyAndValidateFeatureColumns(
            List<String> columns
    ) {
        if (columns == null || columns.isEmpty()) {
            throw new IllegalArgumentException(
                    "ParquetColumnSeriesReader requires at least one "
                            + "feature column.");
        }
        List<String> copy = new ArrayList<>(columns.size());
        Set<String> seen = new HashSet<>();
        for (String column : columns) {
            String normalized = requireNonblank(
                    column, "feature column");
            if (!seen.add(normalized)) {
                throw new IllegalArgumentException(
                        "Duplicate Parquet feature column: " + normalized);
            }
            copy.add(normalized);
        }
        return List.copyOf(copy);
    }

    private static Set<String> snapshotMissingIndicators() {
        if (AppContext.MissingStrings == null
                || AppContext.MissingStrings.isEmpty()) {
            return Set.of();
        }
        Set<String> result = new HashSet<>();
        for (String indicator : AppContext.MissingStrings) {
            if (indicator != null && !indicator.isBlank()) {
                result.add(indicator.trim().toUpperCase(Locale.ROOT));
            }
        }
        return result.isEmpty()
                ? Set.of() : Collections.unmodifiableSet(result);
    }

    private enum ValueKind { DOUBLE, FLOAT, LONG, INT }

    private interface NumericBuffer {
        int size();
        void add(Object source, ValueKind kind, int index, boolean missing);
        void addAll(Object source, ValueKind kind, int count);
        Object toArray(MissingBuffer missing);
    }

    private static final class FloatBuffer implements NumericBuffer {
        private float[] values;
        private int size;

        private FloatBuffer(int capacity) {
            values = new float[Math.max(1, capacity)];
        }

        @Override
        public int size() {
            return size;
        }

        @Override
        public void add(Object source, ValueKind kind,
                        int index, boolean missing) {
            ensure(size + 1);
            values[size++] = missing ? 0.0f : floatAt(source, kind, index);
        }

        @Override
        public void addAll(Object source, ValueKind kind, int count) {
            ensure(size + count);
            if (kind == ValueKind.FLOAT) {
                System.arraycopy(source, 0, values, size, count);
                size += count;
                return;
            }
            for (int i = 0; i < count; i++) {
                values[size++] = floatAt(source, kind, i);
            }
        }

        @Override
        public Object toArray(MissingBuffer missing) {
            float[] result = Arrays.copyOf(values, size);
            if (missing != null) {
                missing.validateLength(size);
                for (int i = 0; i < size; i++) {
                    if (missing.get(i)) {
                        result[i] = Float.NaN;
                    }
                }
            }
            return result;
        }

        private void ensure(int required) {
            if (required > values.length) {
                values = Arrays.copyOf(values,
                        nextCapacity(values.length, required));
            }
        }
    }

    private static final class DoubleBuffer implements NumericBuffer {
        private double[] values;
        private int size;

        private DoubleBuffer(int capacity) {
            values = new double[Math.max(1, capacity)];
        }

        @Override
        public int size() {
            return size;
        }

        @Override
        public void add(Object source, ValueKind kind,
                        int index, boolean missing) {
            ensure(size + 1);
            values[size++] = missing ? 0.0d : doubleAt(source, kind, index);
        }

        @Override
        public void addAll(Object source, ValueKind kind, int count) {
            ensure(size + count);
            if (kind == ValueKind.DOUBLE) {
                System.arraycopy(source, 0, values, size, count);
                size += count;
                return;
            }
            for (int i = 0; i < count; i++) {
                values[size++] = doubleAt(source, kind, i);
            }
        }

        @Override
        public Object toArray(MissingBuffer missing) {
            double[] result = Arrays.copyOf(values, size);
            if (missing != null) {
                missing.validateLength(size);
                for (int i = 0; i < size; i++) {
                    if (missing.get(i)) {
                        result[i] = Double.NaN;
                    }
                }
            }
            return result;
        }

        private void ensure(int required) {
            if (required > values.length) {
                values = Arrays.copyOf(values,
                        nextCapacity(values.length, required));
            }
        }
    }

    private static float floatAt(Object source, ValueKind kind, int index) {
        return switch (kind) {
            case DOUBLE -> (float) ((double[]) source)[index];
            case FLOAT -> ((float[]) source)[index];
            case LONG -> (float) ((long[]) source)[index];
            case INT -> ((int[]) source)[index];
        };
    }

    private static double doubleAt(Object source, ValueKind kind, int index) {
        return switch (kind) {
            case DOUBLE -> ((double[]) source)[index];
            case FLOAT -> ((float[]) source)[index];
            case LONG -> ((long[]) source)[index];
            case INT -> ((int[]) source)[index];
        };
    }

    private static final class MissingBuffer {
        private boolean[] values;
        private int size;

        private MissingBuffer(int capacity) {
            values = new boolean[Math.max(1, capacity)];
        }

        private void add(boolean value) {
            ensure(size + 1);
            values[size++] = value;
        }

        private void addRepeated(boolean value, int count) {
            ensure(size + count);
            Arrays.fill(values, size, size + count, value);
            size += count;
        }

        private boolean get(int index) {
            return values[index];
        }

        private void validateLength(int expected) {
            if (size != expected) {
                throw new IllegalStateException(
                        "Numeric and missing buffers have different lengths.");
            }
        }

        private void ensure(int required) {
            if (required > values.length) {
                values = Arrays.copyOf(values,
                        nextCapacity(values.length, required));
            }
        }
    }

    private static final class ObjectBuffer {
        private Object[] values;
        private int size;

        private ObjectBuffer(int capacity) {
            values = new Object[Math.max(1, capacity)];
        }

        private void add(Object value) {
            ensure(size + 1);
            values[size++] = value;
        }

        private Object[] toArray() {
            return Arrays.copyOf(values, size);
        }

        private void ensure(int required) {
            if (required > values.length) {
                values = Arrays.copyOf(values,
                        nextCapacity(values.length, required));
            }
        }
    }

    private static int nextCapacity(int current, int required) {
        int expanded = current <= Integer.MAX_VALUE / 2
                ? current << 1 : Integer.MAX_VALUE;
        if (expanded < required) {
            expanded = required;
        }
        if (expanded < current) {
            throw new OutOfMemoryError(
                    "Required Parquet column buffer is too large.");
        }
        return expanded;
    }

    public static class ProgressLogger {
        public static void logProgress(int count) {
            if (count > 0 && count % 100000 == 0) {
                System.out.print(".");
            }
        }
    }
}
