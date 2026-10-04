package datasets.readers;

import ch.randelshofer.fastdoubleparser.JavaDoubleParser;
import ch.randelshofer.fastdoubleparser.JavaFloatParser;
import core.AppContext;
import datasets.NumericStorageType;
import datasets.readers.lazy.LazySeriesReader;
import datasets.readers.lazy.LazySeriesRef;
import de.siegmar.fastcsv.reader.CsvReader;
import de.siegmar.fastcsv.reader.CsvRecord;
import preprocessing.standardization.StandardizationStats;
import preprocessing.standardization.Standardizer;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * General per-file delimited multivariate time-series materializer.
 *
 * <p>One file is one observation, one record is one time position, and each
 * selected column is one dimension. Output is always two-dimensional, even
 * when only one feature is selected.</p>
 *
 * <p>Numeric output uses primitive {@code float[][]} or {@code double[][]}.
 * Missing numeric values are represented by primitive NaN. Generic output uses
 * {@code Object[][]}, with missing values represented by null. AUTO resolves
 * to FLOAT64 because delimited text has no physical numeric type.</p>
 *
 * <p>This compatibility reader intentionally retains FastCSV's CsvRecord path
 * for quoted fields, embedded separators, generic parsing, and missing-value
 * handling. Complete numeric workloads should use
 * {@link NumericPerFileDelimitedSeriesReader}, whose callback hot loops avoid
 * record and per-field String materialization.</p>
 */
public final class PerFileDelimitedSeriesReader implements LazySeriesReader {
    private static final int DEFAULT_INITIAL_TIME_CAPACITY = 256;

    private final char fieldSeparator;
    private final boolean hasHeader;
    private final String timeColumn;
    private final List<String> featureColumns;
    private final boolean isNumeric;
    private final boolean hasMissingValues;
    private final StandardizationStats standardizationStats;
    private final int initialTimeCapacity;
    private final NumericStorageType storageType;
    private final Set<String> missingStrings;

    public PerFileDelimitedSeriesReader(
            String entrySeparator, boolean hasHeader, String timeColumn,
            List<String> featureColumns, boolean isNumeric,
            boolean hasMissingValues, StandardizationStats standardizationStats,
            int initialTimeCapacity
    ) {
        this(entrySeparator, hasHeader, timeColumn, featureColumns, isNumeric,
                hasMissingValues, standardizationStats, initialTimeCapacity,
                NumericStorageType.AUTO);
    }

    public PerFileDelimitedSeriesReader(
            String entrySeparator, boolean hasHeader, String timeColumn,
            List<String> featureColumns, boolean isNumeric,
            boolean hasMissingValues, StandardizationStats standardizationStats
    ) {
        this(entrySeparator, hasHeader, timeColumn, featureColumns, isNumeric,
                hasMissingValues, standardizationStats,
                DEFAULT_INITIAL_TIME_CAPACITY, NumericStorageType.AUTO);
    }

    public PerFileDelimitedSeriesReader(
            String entrySeparator, boolean hasHeader, String timeColumn,
            List<String> featureColumns, boolean isNumeric,
            boolean hasMissingValues
    ) {
        this(entrySeparator, hasHeader, timeColumn, featureColumns, isNumeric,
                hasMissingValues, null, DEFAULT_INITIAL_TIME_CAPACITY,
                NumericStorageType.AUTO);
    }

    public PerFileDelimitedSeriesReader(
            String entrySeparator, boolean hasHeader, String timeColumn,
            List<String> featureColumns, boolean isNumeric,
            boolean hasMissingValues, StandardizationStats standardizationStats,
            int initialTimeCapacity, NumericStorageType numericStorageType
    ) {
        String separator = validateSeparator(entrySeparator);
        this.fieldSeparator = separator.charAt(0);
        this.hasHeader = hasHeader;
        this.timeColumn = normalizeNullableString(timeColumn);
        this.featureColumns = copyFeatures(featureColumns);
        this.isNumeric = isNumeric;
        this.hasMissingValues = hasMissingValues;
        this.standardizationStats = standardizationStats;
        if (initialTimeCapacity < 1) {
            throw new IllegalArgumentException(
                    "initialTimeCapacity must be at least 1.");
        }
        this.initialTimeCapacity = initialTimeCapacity;
        NumericStorageType requested = Objects.requireNonNull(
                numericStorageType, "numericStorageType cannot be null.");
        this.storageType = requested == NumericStorageType.AUTO
                ? NumericStorageType.FLOAT64 : requested;
        this.missingStrings = snapshotMissingStrings();
        validateConfiguration();
    }

    @Override
    public Object read(LazySeriesRef reference) {
        if (reference == null) {
            throw new IllegalArgumentException("Cannot read null LazySeriesRef.");
        }
        try {
            return readFile(reference.getFile(), false);
        } catch (IOException e) {
            throw new IllegalStateException(
                    "Failed to read delimited per-file series: "
                            + reference.getFile(), e);
        }
    }

    public Object readFile(Path file) throws IOException {
        return readFile(file, true);
    }

    private Object readFile(Path file, boolean validateMetadata)
            throws IOException {
        if (file == null) {
            throw new IllegalArgumentException("A non-null file is required.");
        }
        if (validateMetadata) {
            validateFile(file);
        }

        try (CsvReader<CsvRecord> reader = CsvReader.builder()
                .fieldSeparator(fieldSeparator)
                .skipEmptyLines(true)
                .detectBomHeader(true)
                .ofCsvRecord(file)) {
            var iterator = reader.iterator();
            if (!iterator.hasNext()) {
                throw new IOException("Delimited series file is empty: " + file);
            }
            CsvRecord first = iterator.next();
            List<String> firstFields = first.getFields();
            if (firstFields.isEmpty()) {
                throw new IOException("Delimited series has no columns: " + file);
            }
            List<String> header = hasHeader ? firstFields : null;
            Selection selection = resolveSelection(
                    file, header, firstFields.size());
            Accumulator accumulator = createAccumulator(
                    selection.featureIndices.length);
            int row = 0;
            if (!hasHeader) {
                appendRecord(file, first, firstFields, row++, selection,
                        accumulator);
            }
            while (iterator.hasNext()) {
                CsvRecord record = iterator.next();
                appendRecord(file, record, record.getFields(), row++,
                        selection, accumulator);
            }
            if (row == 0) {
                throw new IOException(
                        "Delimited series contains a header but no data: "
                                + file);
            }
            Object series = accumulator.toSeries();
            if (standardizationStats != null) {
                Standardizer.transformInstanceInPlace(
                        series, standardizationStats);
            }
            return series;
        } catch (IllegalArgumentException e) {
            throw e;
        } catch (RuntimeException e) {
            throw new IOException(
                    "Failed while parsing delimited series file: " + file, e);
        }
    }

    private void appendRecord(
            Path file, CsvRecord record, List<String> fields, int row,
            Selection selection, Accumulator accumulator
    ) {
        if (fields.size() != selection.columnCount) {
            throw new IllegalArgumentException(
                    "Inconsistent column count in " + file + " at line "
                            + record.getStartingLineNumber() + ". Expected "
                            + selection.columnCount + " but found "
                            + fields.size() + ".");
        }
        for (int dimension = 0;
             dimension < selection.featureIndices.length;
             dimension++) {
            int column = selection.featureIndices[dimension];
            String raw = fields.get(column);
            String token = raw == null ? null : raw.trim();
            boolean missing = isMissing(token);
            if (missing) {
                if (!hasMissingValues) {
                    throw new IllegalArgumentException(
                            "Missing value in " + file + " at data row "
                                    + row + ", column " + column
                                    + " while hasMissingValues=false.");
                }
                accumulator.addMissing(dimension);
            } else if (isNumeric) {
                try {
                    if (storageType == NumericStorageType.FLOAT32) {
                        accumulator.addFloat(dimension,
                                JavaFloatParser.parseFloat(token));
                    } else {
                        accumulator.addDouble(dimension,
                                JavaDoubleParser.parseDouble(token));
                    }
                } catch (NumberFormatException e) {
                    throw new IllegalArgumentException(
                            "Could not parse numeric value '" + token
                                    + "' in " + file + " at data row " + row
                                    + ", column " + column + ".", e);
                }
            } else {
                accumulator.addObject(dimension,
                        DelimitedFileReader.RowParser.parseValue(token));
            }
        }
    }

    private Accumulator createAccumulator(int dimensions) {
        if (!isNumeric) {
            return new ObjectAccumulator(dimensions, initialTimeCapacity);
        }
        return storageType == NumericStorageType.FLOAT32
                ? new FloatAccumulator(dimensions, initialTimeCapacity)
                : new DoubleAccumulator(dimensions, initialTimeCapacity);
    }

    private Selection resolveSelection(
            Path file, List<String> header, int columnCount
    ) {
        int timeIndex = timeColumn == null ? -1
                : hasHeader
                ? findNamedColumn(file, buildHeaderIndex(file, header),
                timeColumn, "time")
                : parseColumnIndex(file, timeColumn, columnCount, "time");

        int[] features;
        if (featureColumns.isEmpty()) {
            features = allColumnsExcept(columnCount, timeIndex);
        } else {
            features = new int[featureColumns.size()];
            Map<String, Integer> index = hasHeader
                    ? buildHeaderIndex(file, header) : Map.of();
            Set<Integer> used = new HashSet<>();
            for (int i = 0; i < featureColumns.size(); i++) {
                String requested = featureColumns.get(i);
                int resolved = hasHeader
                        ? findNamedColumn(file, index, requested, "feature")
                        : parseColumnIndex(file, requested, columnCount,
                        "feature");
                if (resolved == timeIndex) {
                    throw new IllegalArgumentException(
                            "Time column cannot also be a feature: "
                                    + requested);
                }
                if (!used.add(resolved)) {
                    throw new IllegalArgumentException(
                            "Feature column selected more than once: "
                                    + requested);
                }
                features[i] = resolved;
            }
        }
        if (features.length == 0) {
            throw new IllegalArgumentException(
                    "No feature dimensions selected in " + file + ".");
        }
        return new Selection(columnCount, features);
    }

    private static Map<String, Integer> buildHeaderIndex(
            Path file, List<String> header
    ) {
        Map<String, Integer> result = new HashMap<>();
        for (int i = 0; i < header.size(); i++) {
            String name = header.get(i) == null ? "" : header.get(i).trim();
            if (name.isEmpty()) {
                throw new IllegalArgumentException(
                        "Blank header at column " + i + " in " + file + ".");
            }
            if (result.put(name, i) != null) {
                throw new IllegalArgumentException(
                        "Duplicate header '" + name + "' in " + file + ".");
            }
        }
        return result;
    }

    private static int findNamedColumn(
            Path file, Map<String, Integer> index, String name, String role
    ) {
        Integer result = index.get(name);
        if (result == null) {
            throw new IllegalArgumentException(
                    role + " column '" + name + "' not found in " + file + ".");
        }
        return result;
    }

    private static int parseColumnIndex(
            Path file, String value, int count, String role
    ) {
        final int index;
        try {
            index = Integer.parseInt(value.trim());
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException(
                    role + " must be a zero-based column index when "
                            + "hasHeader=false: '" + value + "' in " + file,
                    e);
        }
        if (index < 0 || index >= count) {
            throw new IllegalArgumentException(
                    role + " column index " + index + " is outside [0, "
                            + (count - 1) + "] in " + file + ".");
        }
        return index;
    }

    private static int[] allColumnsExcept(int count, int excluded) {
        int[] result = new int[count - (excluded >= 0 ? 1 : 0)];
        int output = 0;
        for (int i = 0; i < count; i++) {
            if (i != excluded) {
                result[output++] = i;
            }
        }
        return result;
    }

    private boolean isMissing(String token) {
        return token == null || token.isEmpty()
                || missingStrings.contains(token.toUpperCase(Locale.ROOT));
    }

    private interface Accumulator {
        default void addFloat(int dimension, float value) {
            throw new UnsupportedOperationException();
        }
        default void addDouble(int dimension, double value) {
            throw new UnsupportedOperationException();
        }
        default void addObject(int dimension, Object value) {
            throw new UnsupportedOperationException();
        }
        void addMissing(int dimension);
        Object toSeries();
    }

    private static final class FloatAccumulator implements Accumulator {
        private final FloatBuffer[] dimensions;

        private FloatAccumulator(int count, int capacity) {
            dimensions = new FloatBuffer[count];
            for (int i = 0; i < count; i++) {
                dimensions[i] = new FloatBuffer(capacity);
            }
        }
        @Override public void addFloat(int d, float value) {
            dimensions[d].add(value);
        }
        @Override public void addMissing(int d) {
            dimensions[d].add(Float.NaN);
        }
        @Override public Object toSeries() {
            float[][] result = new float[dimensions.length][];
            for (int i = 0; i < dimensions.length; i++) {
                result[i] = dimensions[i].toArray();
            }
            validateLengths(result);
            return result;
        }
    }

    private static final class DoubleAccumulator implements Accumulator {
        private final DoubleBuffer[] dimensions;

        private DoubleAccumulator(int count, int capacity) {
            dimensions = new DoubleBuffer[count];
            for (int i = 0; i < count; i++) {
                dimensions[i] = new DoubleBuffer(capacity);
            }
        }
        @Override public void addDouble(int d, double value) {
            dimensions[d].add(value);
        }
        @Override public void addMissing(int d) {
            dimensions[d].add(Double.NaN);
        }
        @Override public Object toSeries() {
            double[][] result = new double[dimensions.length][];
            for (int i = 0; i < dimensions.length; i++) {
                result[i] = dimensions[i].toArray();
            }
            validateLengths(result);
            return result;
        }
    }

    private static final class ObjectAccumulator implements Accumulator {
        private final ObjectBuffer[] dimensions;

        private ObjectAccumulator(int count, int capacity) {
            dimensions = new ObjectBuffer[count];
            for (int i = 0; i < count; i++) {
                dimensions[i] = new ObjectBuffer(capacity);
            }
        }
        @Override public void addObject(int d, Object value) {
            dimensions[d].add(value);
        }
        @Override public void addMissing(int d) {
            dimensions[d].add(null);
        }
        @Override public Object toSeries() {
            Object[][] result = new Object[dimensions.length][];
            for (int i = 0; i < dimensions.length; i++) {
                result[i] = dimensions[i].toArray();
            }
            validateLengths(result);
            return result;
        }
    }

    private static void validateLengths(float[][] values) {
        int length = values[0].length;
        for (float[] dimension : values) {
            if (dimension.length != length) {
                throw new IllegalStateException(
                        "Per-file dimensions have inconsistent lengths.");
            }
        }
    }
    private static void validateLengths(double[][] values) {
        int length = values[0].length;
        for (double[] dimension : values) {
            if (dimension.length != length) {
                throw new IllegalStateException(
                        "Per-file dimensions have inconsistent lengths.");
            }
        }
    }
    private static void validateLengths(Object[][] values) {
        int length = values[0].length;
        for (Object[] dimension : values) {
            if (dimension.length != length) {
                throw new IllegalStateException(
                        "Per-file dimensions have inconsistent lengths.");
            }
        }
    }

    private static final class FloatBuffer {
        private float[] values; private int size;
        private FloatBuffer(int capacity) {
            values = new float[Math.max(1, capacity)];
        }
        private void add(float value) { ensure(size + 1); values[size++] = value; }
        private float[] toArray() { return Arrays.copyOf(values, size); }
        private void ensure(int required) {
            if (required > values.length) {
                values = Arrays.copyOf(values,
                        nextCapacity(values.length, required));
            }
        }
    }
    private static final class DoubleBuffer {
        private double[] values; private int size;
        private DoubleBuffer(int capacity) {
            values = new double[Math.max(1, capacity)];
        }
        private void add(double value) { ensure(size + 1); values[size++] = value; }
        private double[] toArray() { return Arrays.copyOf(values, size); }
        private void ensure(int required) {
            if (required > values.length) {
                values = Arrays.copyOf(values,
                        nextCapacity(values.length, required));
            }
        }
    }
    private static final class ObjectBuffer {
        private Object[] values; private int size;
        private ObjectBuffer(int capacity) {
            values = new Object[Math.max(1, capacity)];
        }
        private void add(Object value) { ensure(size + 1); values[size++] = value; }
        private Object[] toArray() { return Arrays.copyOf(values, size); }
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
            throw new OutOfMemoryError("Series buffer is too large.");
        }
        return expanded;
    }

    private void validateConfiguration() {
        Set<String> seen = new HashSet<>();
        for (String feature : featureColumns) {
            if (feature == null || feature.isBlank()) {
                throw new IllegalArgumentException(
                        "Feature columns cannot contain blank values.");
            }
            if (!seen.add(feature)) {
                throw new IllegalArgumentException(
                        "Duplicate feature column: " + feature);
            }
        }
        if (standardizationStats != null) {
            if (!isNumeric) {
                throw new IllegalArgumentException(
                        "Standardization requires numeric data.");
            }
            if (!featureColumns.isEmpty()) {
                standardizationStats.validateFeatureCompatibility(
                        featureColumns);
            }
        }
    }

    private static List<String> copyFeatures(List<String> columns) {
        if (columns == null || columns.isEmpty()) {
            return List.of();
        }
        List<String> result = new ArrayList<>(columns.size());
        for (String value : columns) {
            result.add(value == null ? null : value.trim());
        }
        return List.copyOf(result);
    }

    private static Set<String> snapshotMissingStrings() {
        if (AppContext.MissingStrings == null
                || AppContext.MissingStrings.isEmpty()) {
            return Set.of();
        }
        Set<String> result = new HashSet<>();
        for (String value : AppContext.MissingStrings) {
            if (value != null && !value.isBlank()) {
                result.add(value.trim().toUpperCase(Locale.ROOT));
            }
        }
        return result.isEmpty()
                ? Set.of() : Collections.unmodifiableSet(result);
    }

    private static String validateSeparator(String value) {
        if (value == null || value.isEmpty()) {
            throw new IllegalArgumentException(
                    "A non-empty entry separator is required.");
        }
        String normalized = switch (value) {
            case "\\t" -> "\t";
            case "\\n" -> "\n";
            case "\\r" -> "\r";
            default -> value;
        };
        if (normalized.length() != 1
                || normalized.charAt(0) == '\n'
                || normalized.charAt(0) == '\r') {
            throw new IllegalArgumentException(
                    "A non-line, single-character entry separator is required.");
        }
        return normalized;
    }

    private static String normalizeNullableString(String value) {
        if (value == null) { return null; }
        String trimmed = value.trim();
        return trimmed.isEmpty() || trimmed.equalsIgnoreCase("None")
                ? null : trimmed;
    }

    private static void validateFile(Path file) throws IOException {
        if (!Files.isRegularFile(file) || !Files.isReadable(file)) {
            throw new IOException(
                    "Delimited series file is not a readable regular file: "
                            + file);
        }
    }

    private record Selection(int columnCount, int[] featureIndices) {
    }
}
