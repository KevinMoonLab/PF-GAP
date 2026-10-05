package datasets.readers;

import datasets.NumericStorageType;
import datasets.readers.lazy.LazySeriesReader;
import datasets.readers.lazy.LazySeriesRef;
import dev.hardwood.InputFile;
import dev.hardwood.reader.ColumnReader;
import dev.hardwood.reader.ColumnReaders;
import dev.hardwood.reader.ParquetFileReader;
import dev.hardwood.reader.Validity;
import dev.hardwood.schema.ColumnProjection;
import preprocessing.standardization.StandardizationStats;
import preprocessing.standardization.Standardizer;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;

/**
 * High-throughput numeric reader for one per-file Parquet multivariate series.
 *
 * <p>Each selected Parquet column is one dimension and each record is one time
 * position. Output is always {@code float[dimension][time]} or
 * {@code double[dimension][time]}, including the one-dimension case.</p>
 *
 * <p>Physical DOUBLE, FLOAT, INT64, and INT32 feature columns are read through
 * Hardwood's primitive batch accessors. Nulls become primitive NaN when
 * enabled. FILE_ORDER does not project the configured time column. The legacy
 * SORT_DOUBLE_TIME policy name is retained for source compatibility, but its
 * implementation accepts all four supported fixed-width numeric time types.</p>
 *
 * <p>Reader-time standardization remains available for lazy materialization.
 * Eager coordinators should pass null statistics and let the eager pipeline
 * transform the completed dataset.</p>
 */
public final class NumericPerFileParquetSeriesReader
        implements LazySeriesReader {
    private static final int DEFAULT_INITIAL_TIME_CAPACITY = 256;

    public enum TimeOrderPolicy {
        FILE_ORDER,
        SORT_DOUBLE_TIME
    }

    private final String timeColumn;
    private final List<String> featureColumns;
    private final boolean hasMissingValues;
    private final StandardizationStats standardizationStats;
    private final int initialTimeCapacity;
    private final TimeOrderPolicy timeOrderPolicy;
    private final NumericStorageType requestedStorageType;
    private final ColumnProjection projection;

    public NumericPerFileParquetSeriesReader(
            String timeColumn,
            List<String> featureColumns,
            boolean hasMissingValues,
            StandardizationStats standardizationStats
    ) {
        this(timeColumn, featureColumns, hasMissingValues,
                standardizationStats, DEFAULT_INITIAL_TIME_CAPACITY,
                TimeOrderPolicy.FILE_ORDER, NumericStorageType.AUTO);
    }

    public NumericPerFileParquetSeriesReader(
            String timeColumn,
            List<String> featureColumns,
            boolean hasMissingValues
    ) {
        this(timeColumn, featureColumns, hasMissingValues, null,
                DEFAULT_INITIAL_TIME_CAPACITY, TimeOrderPolicy.FILE_ORDER,
                NumericStorageType.AUTO);
    }

    public NumericPerFileParquetSeriesReader(
            String timeColumn,
            List<String> featureColumns,
            boolean hasMissingValues,
            StandardizationStats standardizationStats,
            int initialTimeCapacity,
            TimeOrderPolicy timeOrderPolicy
    ) {
        this(timeColumn, featureColumns, hasMissingValues,
                standardizationStats, initialTimeCapacity, timeOrderPolicy,
                NumericStorageType.AUTO);
    }

    public NumericPerFileParquetSeriesReader(
            String timeColumn,
            List<String> featureColumns,
            boolean hasMissingValues,
            StandardizationStats standardizationStats,
            int initialTimeCapacity,
            TimeOrderPolicy timeOrderPolicy,
            NumericStorageType numericStorageType
    ) {
        this.timeColumn = normalizeNullableString(timeColumn);
        this.featureColumns = copyFeatures(featureColumns);
        this.hasMissingValues = hasMissingValues;
        this.standardizationStats = standardizationStats;
        if (initialTimeCapacity < 1) {
            throw new IllegalArgumentException(
                    "initialTimeCapacity must be at least 1.");
        }
        this.initialTimeCapacity = initialTimeCapacity;
        this.timeOrderPolicy = timeOrderPolicy == null
                ? TimeOrderPolicy.FILE_ORDER : timeOrderPolicy;
        this.requestedStorageType = Objects.requireNonNull(
                numericStorageType, "numericStorageType cannot be null.");
        if (this.timeOrderPolicy == TimeOrderPolicy.SORT_DOUBLE_TIME
                && this.timeColumn == null) {
            throw new IllegalArgumentException(
                    "SORT_DOUBLE_TIME requires a time column.");
        }
        if (standardizationStats != null) {
            standardizationStats.validateFeatureCompatibility(
                    this.featureColumns);
        }
        this.projection = buildProjection();
    }

    @Override
    public Object read(LazySeriesRef reference) {
        if (reference == null) {
            throw new IllegalArgumentException("Cannot read null LazySeriesRef.");
        }
        try {
            return readFileInternal(reference.getFile(), false);
        } catch (IOException e) {
            throw new IllegalStateException(
                    "Failed to read numeric Parquet series: "
                            + reference.getFile(), e);
        }
    }

    public Object readFile(Path file) throws IOException {
        return readFileInternal(file, true);
    }

    public float[][] readFloatFile(Path file) throws IOException {
        if (requestedStorageType != NumericStorageType.FLOAT32) {
            throw new IllegalStateException(
                    "readFloatFile requires FLOAT32 configuration.");
        }
        return (float[][]) readFileInternal(file, true);
    }

    public double[][] readDoubleFile(Path file) throws IOException {
        if (requestedStorageType == NumericStorageType.FLOAT32) {
            throw new IllegalStateException(
                    "readDoubleFile requires FLOAT64 or AUTO configuration.");
        }
        return (double[][]) readFileInternal(file, true);
    }

    private Object readFileInternal(Path file, boolean validateMetadata)
            throws IOException {
        if (file == null) {
            throw new IllegalArgumentException("A non-null file is required.");
        }
        if (validateMetadata) {
            validateFile(file);
        }

        ValueKind[] featureKinds = null;
        ValueKind timeKind = null;
        NumericStorageType outputType = null;
        NumericBuffer[] features = null;
        DoubleBuffer times = timeOrderPolicy == TimeOrderPolicy.SORT_DOUBLE_TIME
                ? new DoubleBuffer(initialTimeCapacity) : null;
        MissingBuffer timeMissing = times == null
                ? null : new MissingBuffer(initialTimeCapacity);
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
                if (featureKinds == null) {
                    featureKinds = detectFeatureKinds(columns, file);
                    outputType = resolveOutputType(featureKinds);
                    features = createFeatureBuffers(outputType);
                    if (times != null) {
                        timeKind = detectKind(
                                columns.getColumnReader(timeColumn),
                                timeColumn, file);
                    }
                }

                for (int d = 0; d < featureColumns.size(); d++) {
                    String name = featureColumns.get(d);
                    ColumnReader reader = columns.getColumnReader(name);
                    Object source = readValues(reader, featureKinds[d],
                            name, file);
                    appendFeatureBatch(source, featureKinds[d],
                            reader.getLeafValidity(), batchCount, features[d],
                            name, recordCount, file);
                }
                if (times != null) {
                    ColumnReader reader = columns.getColumnReader(timeColumn);
                    Object source = readValues(reader, timeKind,
                            timeColumn, file);
                    appendTimeBatch(source, timeKind,
                            reader.getLeafValidity(), batchCount, times,
                            timeMissing);
                }
                recordCount += batchCount;
            }
        } catch (IllegalArgumentException e) {
            throw e;
        } catch (RuntimeException e) {
            throw new IOException(
                    "Failed while reading numeric Parquet series: " + file, e);
        }

        if (recordCount == 0 || features == null) {
            throw new IOException("Parquet series contains no records: " + file);
        }
        for (NumericBuffer feature : features) {
            if (feature.size() != recordCount) {
                throw new IllegalStateException(
                        "Feature buffer length differs from record count in "
                                + file + ".");
            }
        }

        int[] order = times == null ? null
                : buildStableTimeOrder(times.toArray(), timeMissing.toArray());
        Object series = materialize(features, outputType, order);
        if (standardizationStats != null) {
            Standardizer.transformInstanceInPlace(
                    series, standardizationStats);
        }
        return series;
    }

    private ValueKind[] detectFeatureKinds(
            ColumnReaders columns, Path file
    ) {
        ValueKind[] kinds = new ValueKind[featureColumns.size()];
        for (int i = 0; i < kinds.length; i++) {
            String name = featureColumns.get(i);
            kinds[i] = detectKind(
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

    private NumericBuffer[] createFeatureBuffers(NumericStorageType output) {
        NumericBuffer[] result = new NumericBuffer[featureColumns.size()];
        for (int i = 0; i < result.length; i++) {
            result[i] = output == NumericStorageType.FLOAT32
                    ? new FloatBuffer(initialTimeCapacity)
                    : new DoubleBuffer(initialTimeCapacity);
        }
        return result;
    }

    private void appendFeatureBatch(
            Object source,
            ValueKind kind,
            Validity validity,
            int count,
            NumericBuffer destination,
            String column,
            int recordOffset,
            Path file
    ) {
        if (!validity.hasNulls()) {
            destination.addAll(source, kind, count);
            return;
        }
        for (int row = 0; row < count; row++) {
            boolean missing = validity.isNull(row);
            if (missing && !hasMissingValues) {
                throw new IllegalArgumentException(
                        "Null in feature column '" + column + "' at record "
                                + (recordOffset + row) + " in " + file
                                + ", but hasMissingValues=false.");
            }
            destination.add(source, kind, row, missing);
        }
    }

    private static void appendTimeBatch(
            Object source,
            ValueKind kind,
            Validity validity,
            int count,
            DoubleBuffer times,
            MissingBuffer missing
    ) {
        boolean hasNulls = validity.hasNulls();
        for (int row = 0; row < count; row++) {
            boolean isMissing = hasNulls && validity.isNull(row);
            times.add(isMissing ? 0.0d : doubleAt(source, kind, row));
            missing.add(isMissing);
        }
    }

    private Object materialize(
            NumericBuffer[] features,
            NumericStorageType output,
            int[] order
    ) {
        if (output == NumericStorageType.FLOAT32) {
            float[][] result = new float[features.length][];
            for (int d = 0; d < features.length; d++) {
                float[] values = ((FloatBuffer) features[d]).toArray();
                result[d] = order == null ? values : reorder(values, order);
            }
            return result;
        }
        double[][] result = new double[features.length][];
        for (int d = 0; d < features.length; d++) {
            double[] values = ((DoubleBuffer) features[d]).toArray();
            result[d] = order == null ? values : reorder(values, order);
        }
        return result;
    }

    private static int[] buildStableTimeOrder(
            double[] times,
            boolean[] missing
    ) {
        Integer[] boxed = new Integer[times.length];
        for (int i = 0; i < boxed.length; i++) {
            boxed[i] = i;
        }
        Arrays.sort(boxed, (left, right) -> {
            if (missing[left] != missing[right]) {
                return missing[left] ? 1 : -1;
            }
            if (missing[left]) {
                return Integer.compare(left, right);
            }
            int comparison = Double.compare(times[left], times[right]);
            return comparison != 0
                    ? comparison : Integer.compare(left, right);
        });
        int[] order = new int[boxed.length];
        for (int i = 0; i < order.length; i++) {
            order[i] = boxed[i];
        }
        return order;
    }

    private static float[] reorder(float[] source, int[] order) {
        float[] result = new float[order.length];
        for (int i = 0; i < order.length; i++) {
            result[i] = source[order[i]];
        }
        return result;
    }

    private static double[] reorder(double[] source, int[] order) {
        double[] result = new double[order.length];
        for (int i = 0; i < order.length; i++) {
            result[i] = source[order[i]];
        }
        return result;
    }

    private ColumnProjection buildProjection() {
        List<String> columns = new ArrayList<>();
        if (timeOrderPolicy == TimeOrderPolicy.SORT_DOUBLE_TIME) {
            columns.add(timeColumn);
        }
        for (String feature : featureColumns) {
            if (!columns.contains(feature)) {
                columns.add(feature);
            }
        }
        return ColumnProjection.columns(columns.toArray(String[]::new));
    }

    private static ValueKind detectKind(
            ColumnReader reader, String column, Path file
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
                "Column '" + column + "' in " + file
                        + " is not physical DOUBLE, FLOAT, INT64, or INT32.",
                last);
    }

    private static Object readValues(
            ColumnReader reader, ValueKind kind, String column, Path file
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
                    "Column '" + column + "' in " + file
                            + " is incompatible with " + kind + ".", e);
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

    private interface NumericBuffer {
        int size();
        void add(Object source, ValueKind kind, int index, boolean missing);
        void addAll(Object source, ValueKind kind, int count);
    }

    private static final class FloatBuffer implements NumericBuffer {
        private float[] values;
        private int size;
        private FloatBuffer(int capacity) {
            values = new float[Math.max(1, capacity)];
        }
        private void add(float value) {
            ensure(size + 1); values[size++] = value;
        }
        @Override public int size() { return size; }
        @Override public void add(Object source, ValueKind kind,
                                  int index, boolean missing) {
            add(missing ? Float.NaN : floatAt(source, kind, index));
        }
        @Override public void addAll(Object source, ValueKind kind, int count) {
            ensure(size + count);
            if (kind == ValueKind.FLOAT) {
                System.arraycopy(source, 0, values, size, count);
                size += count;
            } else {
                for (int i = 0; i < count; i++) {
                    values[size++] = floatAt(source, kind, i);
                }
            }
        }
        private float[] toArray() { return Arrays.copyOf(values, size); }
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
        private void add(double value) {
            ensure(size + 1); values[size++] = value;
        }
        @Override public int size() { return size; }
        @Override public void add(Object source, ValueKind kind,
                                  int index, boolean missing) {
            add(missing ? Double.NaN : doubleAt(source, kind, index));
        }
        @Override public void addAll(Object source, ValueKind kind, int count) {
            ensure(size + count);
            if (kind == ValueKind.DOUBLE) {
                System.arraycopy(source, 0, values, size, count);
                size += count;
            } else {
                for (int i = 0; i < count; i++) {
                    values[size++] = doubleAt(source, kind, i);
                }
            }
        }
        private double[] toArray() { return Arrays.copyOf(values, size); }
        private void ensure(int required) {
            if (required > values.length) {
                values = Arrays.copyOf(values,
                        nextCapacity(values.length, required));
            }
        }
    }

    private static final class MissingBuffer {
        private boolean[] values;
        private int size;
        private MissingBuffer(int capacity) {
            values = new boolean[Math.max(1, capacity)];
        }
        private void add(boolean value) {
            ensure(size + 1); values[size++] = value;
        }
        private boolean[] toArray() { return Arrays.copyOf(values, size); }
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
            throw new OutOfMemoryError("Parquet series buffer is too large.");
        }
        return expanded;
    }

    private static List<String> copyFeatures(List<String> columns) {
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

    private enum ValueKind { DOUBLE, FLOAT, LONG, INT }
}
