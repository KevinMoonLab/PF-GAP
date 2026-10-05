package datasets.readers;

import ch.randelshofer.fastdoubleparser.JavaDoubleParser;
import core.AppContext;
import datasets.NumericStorageType;
import datasets.readers.lazy.LazySeriesReader;
import datasets.readers.lazy.LazySeriesRef;
import dev.hardwood.InputFile;
import dev.hardwood.reader.ParquetFileReader;
import dev.hardwood.reader.RowReader;
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
import java.util.Set;

/**
 * General per-file Parquet multivariate time-series materializer.
 *
 * <p>One Parquet file is one observation, one record is one time position,
 * and each selected feature column is one dimension. Output remains
 * two-dimensional even when one feature is selected.</p>
 *
 * <p>Numeric data delegates to one reusable
 * {@link NumericPerFileParquetSeriesReader}. Generic data uses Hardwood's
 * projected {@link RowReader} compatibility path and returns
 * {@code Object[dimension][time]}. Physical record order is preserved.</p>
 */
public final class PerFileParquetSeriesReader implements LazySeriesReader {
    private static final int DEFAULT_INITIAL_TIME_CAPACITY = 256;

    private final String timeColumn;
    private final List<String> featureColumns;
    private final boolean isNumeric;
    private final boolean hasMissingValues;
    private final StandardizationStats standardizationStats;
    private final int initialTimeCapacity;
    private final NumericStorageType numericStorageType;
    private final ColumnProjection genericProjection;
    private final Set<String> missingIndicators;
    private final NumericPerFileParquetSeriesReader numericReader;

    public PerFileParquetSeriesReader(
            String timeColumn,
            List<String> featureColumns,
            boolean isNumeric,
            boolean hasMissingValues,
            StandardizationStats standardizationStats,
            int initialTimeCapacity
    ) {
        this(timeColumn, featureColumns, isNumeric, hasMissingValues,
                standardizationStats, initialTimeCapacity,
                NumericStorageType.AUTO);
    }

    public PerFileParquetSeriesReader(
            String timeColumn,
            List<String> featureColumns,
            boolean isNumeric,
            boolean hasMissingValues,
            StandardizationStats standardizationStats
    ) {
        this(timeColumn, featureColumns, isNumeric, hasMissingValues,
                standardizationStats, DEFAULT_INITIAL_TIME_CAPACITY,
                NumericStorageType.AUTO);
    }

    public PerFileParquetSeriesReader(
            String timeColumn,
            List<String> featureColumns,
            boolean isNumeric,
            boolean hasMissingValues
    ) {
        this(timeColumn, featureColumns, isNumeric, hasMissingValues,
                null, DEFAULT_INITIAL_TIME_CAPACITY,
                NumericStorageType.AUTO);
    }

    public PerFileParquetSeriesReader(
            String timeColumn,
            List<String> featureColumns,
            boolean isNumeric,
            boolean hasMissingValues,
            StandardizationStats standardizationStats,
            int initialTimeCapacity,
            NumericStorageType numericStorageType
    ) {
        this.timeColumn = normalizeNullableString(timeColumn);
        this.featureColumns = copyAndValidateFeatures(featureColumns);
        this.isNumeric = isNumeric;
        this.hasMissingValues = hasMissingValues;
        this.standardizationStats = standardizationStats;
        if (initialTimeCapacity < 1) {
            throw new IllegalArgumentException(
                    "initialTimeCapacity must be at least 1.");
        }
        this.initialTimeCapacity = initialTimeCapacity;
        this.numericStorageType = numericStorageType == null
                ? NumericStorageType.AUTO : numericStorageType;
        validateConfiguration();
        this.genericProjection = isNumeric ? null
                : ColumnProjection.columns(
                this.featureColumns.toArray(String[]::new));
        this.missingIndicators = snapshotMissingIndicators();

        // Construct once and reuse for repeated eager or lazy materialization.
        this.numericReader = isNumeric
                ? new NumericPerFileParquetSeriesReader(
                this.timeColumn, this.featureColumns, hasMissingValues,
                standardizationStats, initialTimeCapacity,
                NumericPerFileParquetSeriesReader.TimeOrderPolicy.FILE_ORDER,
                this.numericStorageType)
                : null;
    }

    @Override
    public Object read(LazySeriesRef reference) {
        if (reference == null) {
            throw new IllegalArgumentException(
                    "Cannot read null LazySeriesRef.");
        }
        Path file = reference.getFile();
        try {
            return readFileInternal(file, false, reference);
        } catch (IOException e) {
            throw new IllegalStateException(
                    "Failed to read per-file Parquet series for reader key '"
                            + reference.getReaderKey()
                            + "', instance index "
                            + reference.getIndex()
                            + ", file " + file + ".",
                    e);
        }
    }

    public Object readFile(Path file) throws IOException {
        return readFileInternal(file, true, null);
    }

    private Object readFileInternal(
            Path file,
            boolean validateMetadata,
            LazySeriesRef reference
    ) throws IOException {
        if (file == null) {
            throw new IllegalArgumentException(
                    "A non-null file is required.");
        }
        if (validateMetadata) {
            validateFile(file);
        }
        if (isNumeric) {
            if (validateMetadata) {
                return numericReader.readFile(file);
            }
            if (reference == null) {
                throw new IllegalStateException(
                        "Lazy per-file Parquet read requires its original "
                                + "LazySeriesRef.");
            }
            return numericReader.read(reference);
        }
        return readGenericFile(file);
    }

    private Object readGenericFile(Path file) throws IOException {
        ObjectBuffer[] dimensions = new ObjectBuffer[featureColumns.size()];
        for (int i = 0; i < dimensions.length; i++) {
            dimensions[i] = new ObjectBuffer(initialTimeCapacity);
        }
        int recordCount = 0;

        try (ParquetFileReader parquet =
                     ParquetFileReader.open(InputFile.of(file));
             RowReader rows = parquet.buildRowReader()
                     .projection(genericProjection)
                     .build()) {
            while (rows.hasNext()) {
                rows.next();
                for (int dimension = 0;
                     dimension < featureColumns.size();
                     dimension++) {
                    String column = featureColumns.get(dimension);
                    Object raw = rows.isNull(column)
                            ? null : rows.getValue(column);
                    Object value = parseGenericValue(raw);
                    if (value == null && !hasMissingValues) {
                        throw new IllegalArgumentException(
                                "Missing value in Parquet column '" + column
                                        + "' at record " + recordCount
                                        + " in " + file
                                        + ", but hasMissingValues=false.");
                    }
                    dimensions[dimension].add(value);
                }
                recordCount++;
            }
        } catch (IllegalArgumentException e) {
            throw e;
        } catch (RuntimeException e) {
            throw new IOException(
                    "Failed while reading generic Parquet series: " + file,
                    e);
        }

        if (recordCount == 0) {
            throw new IOException(
                    "Parquet series contains no records: " + file);
        }
        Object[][] result = new Object[dimensions.length][];
        for (int dimension = 0;
             dimension < dimensions.length;
             dimension++) {
            result[dimension] = dimensions[dimension].toArray();
            if (result[dimension].length != recordCount) {
                throw new IllegalStateException(
                        "Generic Parquet dimension has an inconsistent length "
                                + "in " + file + ".");
            }
        }
        return result;
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
        return token == null || token.isEmpty()
                || missingIndicators.contains(
                token.toUpperCase(Locale.ROOT));
    }

    private void validateConfiguration() {
        if (timeColumn != null && featureColumns.contains(timeColumn)) {
            throw new IllegalArgumentException(
                    "The time column cannot also be a feature column: "
                            + timeColumn);
        }
        if (standardizationStats != null) {
            if (!isNumeric) {
                throw new IllegalArgumentException(
                        "Standardization statistics require numeric data.");
            }
            standardizationStats.validateFeatureCompatibility(featureColumns);
        }
    }

    private static List<String> copyAndValidateFeatures(
            List<String> columns
    ) {
        if (columns == null || columns.isEmpty()) {
            throw new IllegalArgumentException(
                    "At least one feature column is required.");
        }
        List<String> result = new ArrayList<>(columns.size());
        Set<String> seen = new HashSet<>();
        for (String column : columns) {
            if (column == null || column.isBlank()) {
                throw new IllegalArgumentException(
                        "Feature columns cannot be blank.");
            }
            String normalized = column.trim();
            if (!seen.add(normalized)) {
                throw new IllegalArgumentException(
                        "Duplicate feature column: " + normalized);
            }
            result.add(normalized);
        }
        return List.copyOf(result);
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

    private static String normalizeNullableString(String value) {
        if (value == null) {
            return null;
        }
        String trimmed = value.trim();
        return trimmed.isEmpty() || trimmed.equalsIgnoreCase("None")
                ? null : trimmed;
    }

    private static void validateFile(Path file) throws IOException {
        if (!Files.isRegularFile(file) || !Files.isReadable(file)) {
            throw new IOException(
                    "Parquet series file is not a readable regular file: "
                            + file);
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
                    "Generic Parquet series buffer is too large.");
        }
        return expanded;
    }
}
