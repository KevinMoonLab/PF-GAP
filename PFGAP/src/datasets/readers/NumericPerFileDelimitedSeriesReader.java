package datasets.readers;

import ch.randelshofer.fastdoubleparser.JavaDoubleParser;
import ch.randelshofer.fastdoubleparser.JavaFloatParser;
import datasets.NumericStorageType;
import datasets.readers.lazy.LazySeriesReader;
import datasets.readers.lazy.LazySeriesRef;
import de.siegmar.fastcsv.reader.AbstractBaseCsvCallbackHandler;
import de.siegmar.fastcsv.reader.CsvReader;
import preprocessing.standardization.StandardizationScope;
import preprocessing.standardization.StandardizationStats;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * High-throughput numeric reader for one per-file delimited multivariate
 * time-series observation.
 *
 * <pre>
 * one delimited file  = one dataset observation
 * one CSV record      = one time position
 * one selected column = one time-series dimension
 * </pre>
 *
 * <p>Per-file observations are always two-dimensional. A single selected
 * feature is returned as {@code float[1][time]} or
 * {@code double[1][time]}; it is never collapsed.</p>
 *
 * <p>This optimized reader supports complete numeric data only. It rejects
 * empty selected fields and allocates no missing-position masks. FLOAT32 and
 * FLOAT64 use separate monomorphic FastCSV callback handlers. Raw and
 * standardized reads also use separate handlers, so the per-value hot loop has
 * no storage-type branch, no standardization-presence branch, no numeric-buffer
 * interface dispatch, and no buffer downcast.</p>
 *
 * <p>AUTO resolves to FLOAT64 because delimited text has no physical numeric
 * type. Standardization, when configured for lazy materialization, is fused
 * into parsing using double-precision fitted statistics.</p>
 */
public final class NumericPerFileDelimitedSeriesReader
        implements LazySeriesReader {
    private static final int DEFAULT_INITIAL_TIME_CAPACITY = 4096;

    private final char fieldSeparator;
    private final boolean hasHeader;
    private final String timeColumn;
    private final List<String> featureColumns;
    private final StandardizationStats standardizationStats;
    private final int initialTimeCapacity;
    private final NumericStorageType storageType;

    public NumericPerFileDelimitedSeriesReader(
            String entrySeparator,
            boolean hasHeader,
            String timeColumn,
            List<String> featureColumns,
            StandardizationStats standardizationStats
    ) {
        this(entrySeparator, hasHeader, timeColumn, featureColumns,
                standardizationStats, DEFAULT_INITIAL_TIME_CAPACITY,
                NumericStorageType.AUTO);
    }

    public NumericPerFileDelimitedSeriesReader(
            String entrySeparator,
            boolean hasHeader,
            String timeColumn,
            List<String> featureColumns
    ) {
        this(entrySeparator, hasHeader, timeColumn, featureColumns,
                null, DEFAULT_INITIAL_TIME_CAPACITY,
                NumericStorageType.AUTO);
    }

    public NumericPerFileDelimitedSeriesReader(
            String entrySeparator,
            boolean hasHeader,
            String timeColumn,
            List<String> featureColumns,
            StandardizationStats standardizationStats,
            int initialTimeCapacity
    ) {
        this(entrySeparator, hasHeader, timeColumn, featureColumns,
                standardizationStats, initialTimeCapacity,
                NumericStorageType.AUTO);
    }

    public NumericPerFileDelimitedSeriesReader(
            String entrySeparator,
            boolean hasHeader,
            String timeColumn,
            List<String> featureColumns,
            StandardizationStats standardizationStats,
            int initialTimeCapacity,
            NumericStorageType numericStorageType
    ) {
        String separator = validateAndNormalizeSeparator(entrySeparator);
        this.fieldSeparator = separator.charAt(0);
        this.hasHeader = hasHeader;
        this.timeColumn = normalizeNullableString(timeColumn);
        this.featureColumns = copyFeatureColumns(featureColumns);
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
        validateFeatureConfiguration();
        validateStandardizationConfiguration();
    }

    @Override
    public Object read(LazySeriesRef reference) {
        if (reference == null) {
            throw new IllegalArgumentException(
                    "Cannot read a null LazySeriesRef.");
        }
        try {
            return readFileInternal(reference.getFile(), false);
        } catch (IOException e) {
            throw new IllegalStateException(
                    "Failed to read numeric delimited time-series file: "
                            + reference.getFile(), e);
        }
    }

    /** Returns either {@code float[][]} or {@code double[][]}. */
    public Object readFile(Path file) throws IOException {
        return readFileInternal(file, true);
    }

    public float[][] readFloatFile(Path file) throws IOException {
        requireStorageType(NumericStorageType.FLOAT32, "readFloatFile");
        return (float[][]) readFileInternal(file, true);
    }

    public double[][] readDoubleFile(Path file) throws IOException {
        requireStorageType(NumericStorageType.FLOAT64, "readDoubleFile");
        return (double[][]) readFileInternal(file, true);
    }

    private void requireStorageType(
            NumericStorageType required,
            String method
    ) {
        if (storageType != required) {
            throw new IllegalStateException(method + " requires " + required
                    + ", but this reader uses " + storageType + ".");
        }
    }

    private Object readFileInternal(Path file, boolean validateMetadata)
            throws IOException {
        if (file == null) {
            throw new IllegalArgumentException(
                    "NumericPerFileDelimitedSeriesReader requires a file.");
        }
        if (validateMetadata) {
            validateFile(file);
        }

        BaseHandler handler = createHandler(file);
        try (CsvReader<Boolean> csvReader = CsvReader.builder()
                .fieldSeparator(fieldSeparator)
                .skipEmptyLines(false)
                .detectBomHeader(true)
                .build(handler, file)) {
            for (Boolean ignored : csvReader) {
                // Iteration drives callback parsing.
            }
        } catch (IllegalArgumentException e) {
            throw e;
        } catch (RuntimeException e) {
            throw new IOException(
                    "Failed while parsing numeric per-file delimited series: "
                            + file, e);
        }
        return handler.toSeries();
    }

    private BaseHandler createHandler(Path file) {
        if (storageType == NumericStorageType.FLOAT32) {
            return standardizationStats == null
                    ? new RawFloatHandler(file, hasHeader, timeColumn,
                    featureColumns, initialTimeCapacity)
                    : new StandardizedFloatHandler(file, hasHeader,
                    timeColumn, featureColumns, initialTimeCapacity,
                    standardizationStats);
        }
        return standardizationStats == null
                ? new RawDoubleHandler(file, hasHeader, timeColumn,
                featureColumns, initialTimeCapacity)
                : new StandardizedDoubleHandler(file, hasHeader,
                timeColumn, featureColumns, initialTimeCapacity,
                standardizationStats);
    }

    /** Shared schema resolution only. No selected-value parsing occurs here. */
    private abstract static class BaseHandler
            extends AbstractBaseCsvCallbackHandler<Boolean> {
        protected final Path file;
        private final boolean hasHeader;
        private final String timeColumn;
        private final List<String> requestedFeatures;
        protected final int initialCapacity;

        private List<String> firstRecordFields = new ArrayList<>();
        protected int expectedColumnCount;
        protected int[] columnToDimension;
        protected boolean schemaResolved;
        protected int dataRecordCount;

        private BaseHandler(
                Path file,
                boolean hasHeader,
                String timeColumn,
                List<String> requestedFeatures,
                int initialCapacity
        ) {
            this.file = file;
            this.hasHeader = hasHeader;
            this.timeColumn = timeColumn;
            this.requestedFeatures = requestedFeatures;
            this.initialCapacity = initialCapacity;
        }

        @Override
        protected final Boolean buildRecord() {
            int actual = getFieldCount();
            if (!schemaResolved) {
                resolveSelection(actual);
                schemaResolved = true;
                if (!hasHeader) {
                    parseBufferedFirstRecord(firstRecordFields);
                    dataRecordCount++;
                    firstRecordFields = null;
                    return Boolean.TRUE;
                }
                firstRecordFields = null;
                return null;
            }
            if (actual != expectedColumnCount) {
                throw inconsistentColumnCount(actual);
            }
            dataRecordCount++;
            return Boolean.TRUE;
        }

        protected final void bufferFirstField(
                char[] buffer,
                int offset,
                int length
        ) {
            firstRecordFields.add(new String(buffer, offset, length));
        }

        protected final int selectedDimension(int fieldIndex) {
            if (fieldIndex >= expectedColumnCount) {
                throw inconsistentColumnCount(fieldIndex + 1);
            }
            return columnToDimension[fieldIndex];
        }

        protected final TrimmedRange trim(
                char[] buffer,
                int offset,
                int length,
                int fieldIndex
        ) {
            int start = offset;
            int end = offset + length;
            while (start < end && Character.isWhitespace(buffer[start])) {
                start++;
            }
            while (end > start && Character.isWhitespace(buffer[end - 1])) {
                end--;
            }
            if (start == end) {
                throw missingValue(fieldIndex);
            }
            return new TrimmedRange(start, end - start);
        }

        protected final IllegalArgumentException parseFailure(
                int fieldIndex,
                Exception cause
        ) {
            return new IllegalArgumentException(
                    "Could not parse numeric value in " + file
                            + " at data record " + dataRecordCount
                            + ", column " + fieldIndex + ".", cause);
        }

        private IllegalArgumentException missingValue(int fieldIndex) {
            return new IllegalArgumentException(
                    "Missing numeric value in " + file
                            + " at data record " + dataRecordCount
                            + ", column " + fieldIndex + ". The optimized "
                            + "reader does not support missing values.");
        }

        private void resolveSelection(int columnCount) {
            if (columnCount <= 0
                    || firstRecordFields.size() != columnCount) {
                throw new IllegalArgumentException(
                        "Invalid first record in numeric series file: " + file);
            }
            expectedColumnCount = columnCount;
            int timeIndex = resolveTimeColumnIndex(columnCount);
            int[] features = resolveFeatureIndices(columnCount, timeIndex);
            if (features.length == 0) {
                throw new IllegalArgumentException(
                        "Per-file series must contain at least one selected "
                                + "feature dimension: " + file);
            }
            columnToDimension = new int[columnCount];
            Arrays.fill(columnToDimension, -1);
            for (int dimension = 0; dimension < features.length; dimension++) {
                columnToDimension[features[dimension]] = dimension;
            }
            initializeDimensions(features.length);
        }

        private int resolveTimeColumnIndex(int columnCount) {
            if (timeColumn == null) {
                return -1;
            }
            if (hasHeader) {
                Integer index = buildHeaderIndex().get(timeColumn);
                if (index == null) {
                    throw new IllegalArgumentException(
                            "Time column not found: " + timeColumn
                                    + " in " + file);
                }
                return index;
            }
            return parseAndValidateIndex(
                    timeColumn, columnCount, "timeColumn");
        }

        private int[] resolveFeatureIndices(
                int columnCount,
                int timeIndex
        ) {
            if (requestedFeatures.isEmpty()) {
                int[] result = new int[
                        columnCount - (timeIndex >= 0 ? 1 : 0)];
                int output = 0;
                for (int index = 0; index < columnCount; index++) {
                    if (index != timeIndex) {
                        result[output++] = index;
                    }
                }
                return result;
            }

            int[] result = new int[requestedFeatures.size()];
            Set<Integer> used = new HashSet<>();
            Map<String, Integer> header = hasHeader
                    ? buildHeaderIndex() : Map.of();
            for (int i = 0; i < requestedFeatures.size(); i++) {
                String requested = requestedFeatures.get(i);
                int index;
                if (hasHeader) {
                    Integer resolved = header.get(requested);
                    if (resolved == null) {
                        throw new IllegalArgumentException(
                                "Feature column not found: " + requested
                                        + " in " + file);
                    }
                    index = resolved;
                } else {
                    index = parseAndValidateIndex(
                            requested, columnCount, "feature column");
                }
                if (index == timeIndex) {
                    throw new IllegalArgumentException(
                            "Time column cannot also be a feature column: "
                                    + requested);
                }
                if (!used.add(index)) {
                    throw new IllegalArgumentException(
                            "Feature column selected more than once: "
                                    + requested);
                }
                result[i] = index;
            }
            return result;
        }

        private Map<String, Integer> buildHeaderIndex() {
            Map<String, Integer> result = new HashMap<>();
            for (int i = 0; i < firstRecordFields.size(); i++) {
                String name = firstRecordFields.get(i).trim();
                if (name.isEmpty()) {
                    throw new IllegalArgumentException(
                            "Blank header at column " + i + " in " + file);
                }
                if (result.put(name, i) != null) {
                    throw new IllegalArgumentException(
                            "Duplicate header '" + name + "' in " + file);
                }
            }
            return result;
        }

        private int parseAndValidateIndex(
                String value,
                int columnCount,
                String role
        ) {
            final int index;
            try {
                index = Integer.parseInt(value.trim());
            } catch (NumberFormatException e) {
                throw new IllegalArgumentException(
                        role + " must be a zero-based integer column index "
                                + "when hasHeader=false. Received '" + value
                                + "' for " + file + ".", e);
            }
            if (index < 0 || index >= columnCount) {
                throw new IllegalArgumentException(
                        role + " index " + index + " is outside [0, "
                                + (columnCount - 1) + "] for " + file + ".");
            }
            return index;
        }

        protected final void validateCompletedSeries(
                int dimensionCount,
                LengthLookup lookup
        ) {
            if (!schemaResolved) {
                throw new IllegalArgumentException(
                        "Delimited series file is empty: " + file);
            }
            if (dataRecordCount == 0) {
                throw new IllegalArgumentException(
                        "Delimited series file contains no data records: "
                                + file);
            }
            if (dimensionCount == 0) {
                throw new IllegalStateException(
                        "No selected dimensions were initialized: " + file);
            }
            int expected = lookup.length(0);
            for (int dimension = 1;
                 dimension < dimensionCount;
                 dimension++) {
                if (lookup.length(dimension) != expected) {
                    throw new IllegalStateException(
                            "Selected dimensions have inconsistent lengths in "
                                    + file + ".");
                }
            }
        }

        private IllegalArgumentException inconsistentColumnCount(int actual) {
            return new IllegalArgumentException(
                    "Inconsistent column count in " + file
                            + " at CSV record beginning on line "
                            + getStartingLineNumber() + ". Expected "
                            + expectedColumnCount + " columns but found "
                            + actual + ".");
        }

        protected abstract void initializeDimensions(int dimensionCount);
        protected abstract void parseBufferedFirstRecord(List<String> fields);
        protected abstract Object toSeries();
    }

    private static final class RawFloatHandler extends BaseHandler {
        private FloatBuffer[] dimensions;

        private RawFloatHandler(Path file, boolean hasHeader,
                                String timeColumn, List<String> features,
                                int capacity) {
            super(file, hasHeader, timeColumn, features, capacity);
        }

        @Override
        public void handleField(int fieldIndex, char[] buffer, int offset,
                                int length, boolean quoted) {
            if (!schemaResolved) {
                bufferFirstField(buffer, offset, length);
                return;
            }
            int dimension = selectedDimension(fieldIndex);
            if (dimension < 0) {
                return;
            }
            TrimmedRange range = trim(buffer, offset, length, fieldIndex);
            try {
                dimensions[dimension].add(JavaFloatParser.parseFloat(
                        buffer, range.offset, range.length));
            } catch (NumberFormatException e) {
                throw parseFailure(fieldIndex, e);
            }
        }

        @Override
        protected void initializeDimensions(int count) {
            dimensions = new FloatBuffer[count];
            for (int i = 0; i < count; i++) {
                dimensions[i] = new FloatBuffer(initialCapacity);
            }
        }

        @Override
        protected void parseBufferedFirstRecord(List<String> fields) {
            for (int column = 0; column < fields.size(); column++) {
                int dimension = columnToDimension[column];
                if (dimension < 0) {
                    continue;
                }
                String token = fields.get(column).trim();
                if (token.isEmpty()) {
                    throw new IllegalArgumentException(
                            "Missing numeric value in " + file
                                    + " at data record 0, column " + column
                                    + ". The optimized reader does not support "
                                    + "missing values.");
                }
                try {
                    dimensions[dimension].add(
                            JavaFloatParser.parseFloat(token));
                } catch (NumberFormatException e) {
                    throw parseFailure(column, e);
                }
            }
        }

        @Override
        protected Object toSeries() {
            validateCompletedSeries(dimensions.length,
                    index -> dimensions[index].size());
            float[][] result = new float[dimensions.length][];
            for (int i = 0; i < dimensions.length; i++) {
                result[i] = dimensions[i].toArray();
            }
            return result;
        }
    }

    private static final class StandardizedFloatHandler extends BaseHandler {
        private final StandardizationStats stats;
        private FloatBuffer[] dimensions;
        private double[] centers;
        private double[] inverseScales;

        private StandardizedFloatHandler(Path file, boolean hasHeader,
                                         String timeColumn,
                                         List<String> features, int capacity,
                                         StandardizationStats stats) {
            super(file, hasHeader, timeColumn, features, capacity);
            this.stats = stats;
        }

        @Override
        public void handleField(int fieldIndex, char[] buffer, int offset,
                                int length, boolean quoted) {
            if (!schemaResolved) {
                bufferFirstField(buffer, offset, length);
                return;
            }
            int dimension = selectedDimension(fieldIndex);
            if (dimension < 0) {
                return;
            }
            TrimmedRange range = trim(buffer, offset, length, fieldIndex);
            try {
                float value = JavaFloatParser.parseFloat(
                        buffer, range.offset, range.length);
                dimensions[dimension].add((float) ((value
                        - centers[dimension]) * inverseScales[dimension]));
            } catch (NumberFormatException e) {
                throw parseFailure(fieldIndex, e);
            }
        }

        @Override
        protected void initializeDimensions(int count) {
            validateStandardizationDimensionCount(stats, count);
            dimensions = new FloatBuffer[count];
            centers = new double[count];
            inverseScales = new double[count];
            for (int i = 0; i < count; i++) {
                dimensions[i] = new FloatBuffer(initialCapacity);
                centers[i] = stats.getCenterForDimension(i);
                inverseScales[i] = 1.0 / stats.getScaleForDimension(i);
            }
        }

        @Override
        protected void parseBufferedFirstRecord(List<String> fields) {
            for (int column = 0; column < fields.size(); column++) {
                int dimension = columnToDimension[column];
                if (dimension < 0) {
                    continue;
                }
                String token = fields.get(column).trim();
                if (token.isEmpty()) {
                    throw new IllegalArgumentException(
                            "Missing numeric value in " + file
                                    + " at data record 0, column " + column
                                    + ". The optimized reader does not support "
                                    + "missing values.");
                }
                try {
                    float value = JavaFloatParser.parseFloat(token);
                    dimensions[dimension].add((float) ((value
                            - centers[dimension])
                            * inverseScales[dimension]));
                } catch (NumberFormatException e) {
                    throw parseFailure(column, e);
                }
            }
        }

        @Override
        protected Object toSeries() {
            validateCompletedSeries(dimensions.length,
                    index -> dimensions[index].size());
            float[][] result = new float[dimensions.length][];
            for (int i = 0; i < dimensions.length; i++) {
                result[i] = dimensions[i].toArray();
            }
            return result;
        }
    }

    private static final class RawDoubleHandler extends BaseHandler {
        private DoubleBuffer[] dimensions;

        private RawDoubleHandler(Path file, boolean hasHeader,
                                 String timeColumn, List<String> features,
                                 int capacity) {
            super(file, hasHeader, timeColumn, features, capacity);
        }

        @Override
        public void handleField(int fieldIndex, char[] buffer, int offset,
                                int length, boolean quoted) {
            if (!schemaResolved) {
                bufferFirstField(buffer, offset, length);
                return;
            }
            int dimension = selectedDimension(fieldIndex);
            if (dimension < 0) {
                return;
            }
            TrimmedRange range = trim(buffer, offset, length, fieldIndex);
            try {
                dimensions[dimension].add(JavaDoubleParser.parseDouble(
                        buffer, range.offset, range.length));
            } catch (NumberFormatException e) {
                throw parseFailure(fieldIndex, e);
            }
        }

        @Override
        protected void initializeDimensions(int count) {
            dimensions = new DoubleBuffer[count];
            for (int i = 0; i < count; i++) {
                dimensions[i] = new DoubleBuffer(initialCapacity);
            }
        }

        @Override
        protected void parseBufferedFirstRecord(List<String> fields) {
            for (int column = 0; column < fields.size(); column++) {
                int dimension = columnToDimension[column];
                if (dimension < 0) {
                    continue;
                }
                String token = fields.get(column).trim();
                if (token.isEmpty()) {
                    throw new IllegalArgumentException(
                            "Missing numeric value in " + file
                                    + " at data record 0, column " + column
                                    + ". The optimized reader does not support "
                                    + "missing values.");
                }
                try {
                    dimensions[dimension].add(
                            JavaDoubleParser.parseDouble(token));
                } catch (NumberFormatException e) {
                    throw parseFailure(column, e);
                }
            }
        }

        @Override
        protected Object toSeries() {
            validateCompletedSeries(dimensions.length,
                    index -> dimensions[index].size());
            double[][] result = new double[dimensions.length][];
            for (int i = 0; i < dimensions.length; i++) {
                result[i] = dimensions[i].toArray();
            }
            return result;
        }
    }

    private static final class StandardizedDoubleHandler
            extends BaseHandler {
        private final StandardizationStats stats;
        private DoubleBuffer[] dimensions;
        private double[] centers;
        private double[] inverseScales;

        private StandardizedDoubleHandler(Path file, boolean hasHeader,
                                          String timeColumn,
                                          List<String> features, int capacity,
                                          StandardizationStats stats) {
            super(file, hasHeader, timeColumn, features, capacity);
            this.stats = stats;
        }

        @Override
        public void handleField(int fieldIndex, char[] buffer, int offset,
                                int length, boolean quoted) {
            if (!schemaResolved) {
                bufferFirstField(buffer, offset, length);
                return;
            }
            int dimension = selectedDimension(fieldIndex);
            if (dimension < 0) {
                return;
            }
            TrimmedRange range = trim(buffer, offset, length, fieldIndex);
            try {
                double value = JavaDoubleParser.parseDouble(
                        buffer, range.offset, range.length);
                dimensions[dimension].add((value - centers[dimension])
                        * inverseScales[dimension]);
            } catch (NumberFormatException e) {
                throw parseFailure(fieldIndex, e);
            }
        }

        @Override
        protected void initializeDimensions(int count) {
            validateStandardizationDimensionCount(stats, count);
            dimensions = new DoubleBuffer[count];
            centers = new double[count];
            inverseScales = new double[count];
            for (int i = 0; i < count; i++) {
                dimensions[i] = new DoubleBuffer(initialCapacity);
                centers[i] = stats.getCenterForDimension(i);
                inverseScales[i] = 1.0 / stats.getScaleForDimension(i);
            }
        }

        @Override
        protected void parseBufferedFirstRecord(List<String> fields) {
            for (int column = 0; column < fields.size(); column++) {
                int dimension = columnToDimension[column];
                if (dimension < 0) {
                    continue;
                }
                String token = fields.get(column).trim();
                if (token.isEmpty()) {
                    throw new IllegalArgumentException(
                            "Missing numeric value in " + file
                                    + " at data record 0, column " + column
                                    + ". The optimized reader does not support "
                                    + "missing values.");
                }
                try {
                    double value = JavaDoubleParser.parseDouble(token);
                    dimensions[dimension].add((value - centers[dimension])
                            * inverseScales[dimension]);
                } catch (NumberFormatException e) {
                    throw parseFailure(column, e);
                }
            }
        }

        @Override
        protected Object toSeries() {
            validateCompletedSeries(dimensions.length,
                    index -> dimensions[index].size());
            double[][] result = new double[dimensions.length][];
            for (int i = 0; i < dimensions.length; i++) {
                result[i] = dimensions[i].toArray();
            }
            return result;
        }
    }

    private static void validateStandardizationDimensionCount(
            StandardizationStats stats,
            int dimensionCount
    ) {
        if (stats.getScope() == StandardizationScope.PER_DIMENSION
                && stats.getStatisticGroupCount() != dimensionCount) {
            throw new IllegalArgumentException(
                    "Series contains " + dimensionCount
                            + " dimensions, but PER_DIMENSION statistics "
                            + "contain " + stats.getStatisticGroupCount()
                            + " groups.");
        }
    }

    @FunctionalInterface
    private interface LengthLookup {
        int length(int dimension);
    }

    private record TrimmedRange(int offset, int length) {
    }

    private static final class FloatBuffer {
        private float[] values;
        private int size;

        private FloatBuffer(int capacity) {
            values = new float[Math.max(1, capacity)];
        }

        private void add(float value) {
            ensure(size + 1);
            values[size++] = value;
        }

        private int size() {
            return size;
        }

        private float[] toArray() {
            return size == values.length
                    ? values : Arrays.copyOf(values, size);
        }

        private void ensure(int required) {
            if (required > values.length) {
                values = Arrays.copyOf(values,
                        nextCapacity(values.length, required));
            }
        }
    }

    private static final class DoubleBuffer {
        private double[] values;
        private int size;

        private DoubleBuffer(int capacity) {
            values = new double[Math.max(1, capacity)];
        }

        private void add(double value) {
            ensure(size + 1);
            values[size++] = value;
        }

        private int size() {
            return size;
        }

        private double[] toArray() {
            return size == values.length
                    ? values : Arrays.copyOf(values, size);
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
                    "Required numeric series buffer is too large.");
        }
        return expanded;
    }

    private void validateFile(Path file) throws IOException {
        if (!Files.isRegularFile(file) || !Files.isReadable(file)) {
            throw new IOException(
                    "Numeric delimited time-series file is not a readable "
                            + "regular file: " + file);
        }
    }

    private void validateFeatureConfiguration() {
        Set<String> used = new HashSet<>();
        for (String feature : featureColumns) {
            if (feature == null || feature.isBlank()) {
                throw new IllegalArgumentException(
                        "Feature-column names or indices cannot be blank.");
            }
            if (!used.add(feature)) {
                throw new IllegalArgumentException(
                        "Feature column selected more than once: " + feature);
            }
        }
    }

    private void validateStandardizationConfiguration() {
        if (standardizationStats != null && !featureColumns.isEmpty()) {
            standardizationStats.validateFeatureCompatibility(featureColumns);
        }
    }

    private static List<String> copyFeatureColumns(List<String> columns) {
        if (columns == null || columns.isEmpty()) {
            return List.of();
        }
        List<String> copy = new ArrayList<>(columns.size());
        for (String column : columns) {
            copy.add(column == null ? null : column.trim());
        }
        return List.copyOf(copy);
    }

    private static String validateAndNormalizeSeparator(String separator) {
        if (separator == null || separator.isEmpty()) {
            throw new IllegalArgumentException(
                    "A non-empty entry separator is required.");
        }
        String normalized = switch (separator) {
            case "\\t" -> "\t";
            case "\\n" -> "\n";
            case "\\r" -> "\r";
            default -> separator;
        };
        if (normalized.length() != 1) {
            throw new IllegalArgumentException(
                    "A single-character entry separator is required: '"
                            + separator + "'.");
        }
        if (normalized.charAt(0) == '\n'
                || normalized.charAt(0) == '\r') {
            throw new IllegalArgumentException(
                    "The entry separator cannot be a line separator.");
        }
        return normalized;
    }

    private static String normalizeNullableString(String value) {
        if (value == null) {
            return null;
        }
        String trimmed = value.trim();
        return trimmed.isEmpty() || trimmed.equalsIgnoreCase("None")
                ? null : trimmed;
    }
}