package datasets.readers;

import core.AppContext;
import datasets.ListObjectDataset;
import datasets.NumericStorageType;
import dev.hardwood.InputFile;
import dev.hardwood.reader.ColumnReader;
import dev.hardwood.reader.ColumnReaders;
import dev.hardwood.reader.ParquetFileReader;
import dev.hardwood.reader.Validity;
import dev.hardwood.schema.ColumnProjection;
import preprocessing.standardization.StandardizationStats;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * High-throughput grouped numeric long-format Parquet reader.
 *
 * <p>One Parquet record represents one position in an observation. Records are
 * grouped by {@code idColumn}. Each projected feature column becomes one
 * dimension. A configured time column controls within-group ordering;
 * otherwise physical Parquet record order is retained.</p>
 *
 * <p>The hot path uses Hardwood's batch-oriented {@link ColumnReaders} API.
 * Fixed-width feature columns are decoded into primitive batch arrays and
 * appended directly to primitive per-group buffers. The implementation avoids
 * RowReader calls, per-record feature arrays, and boxed numeric feature
 * storage.</p>
 *
 * <p>Supported feature physical types are DOUBLE, FLOAT, INT64, and INT32.
 * Explicit FLOAT32 stores primitive float values. Explicit FLOAT64 stores
 * primitive double values. AUTO preserves FLOAT32 only when every projected
 * feature column is physically FLOAT; otherwise it promotes to FLOAT64.</p>
 *
 * <p>Parquet null feature values become primitive NaN when missing values are
 * enabled. No boxed numeric observation arrays are produced.</p>
 *
 * <p>ID, time, and label columns may use a supported fixed-width primitive or
 * string representation. Multiple label columns are preserved as immutable
 * lists. Repeated labels must be consistent within each ID group.</p>
 *
 * <p>A configured time column currently requires nonmissing values. Hybrid
 * ordering for partially missing time coordinates is tracked as pre-v1 task
 * LONG-TIME-01.</p>
 */
public class NumericLongFormatParquetReader implements DatasetReader {
    private static final int DEFAULT_INITIAL_GROUP_CAPACITY = 256;

    private final String dataFileName;
    private final boolean hasMissingValues;
    private final boolean isRegression;
    private final NumericStorageType requestedStorageType;
    private final String idColumn;
    private final String timeColumn;
    private final List<String> featureColumns;
    private final List<String> labelColumns;
    private final StandardizationStats standardizationStats;
    private final int initialGroupCapacity;
    private final List<String> projectedColumns;
    private final ColumnProjection projection;

    public NumericLongFormatParquetReader(ReaderOptions options) {
        this(requireOptions(options).getDataPath(),
                options.hasMissingValues(), options.isRegression(),
                options.getIdColumn(), options.getTimeColumn(),
                options.getFeatureColumns(), options.getLabelColumns(),
                options.getStandardizationStats(),
                DEFAULT_INITIAL_GROUP_CAPACITY,
                options.getNumericStorageType());
        if (!options.isNumeric()) {
            throw new IllegalArgumentException(
                    "NumericLongFormatParquetReader requires isNumeric=true.");
        }
    }

    public NumericLongFormatParquetReader(
            String dataFileName,
            boolean hasMissingValues,
            boolean isRegression,
            String idColumn,
            String timeColumn,
            List<String> featureColumns,
            List<String> labelColumns,
            StandardizationStats standardizationStats
    ) {
        this(dataFileName, hasMissingValues, isRegression, idColumn,
                timeColumn, featureColumns, labelColumns,
                standardizationStats, DEFAULT_INITIAL_GROUP_CAPACITY,
                NumericStorageType.AUTO);
    }

    public NumericLongFormatParquetReader(
            String dataFileName,
            boolean hasMissingValues,
            boolean isRegression,
            String idColumn,
            String timeColumn,
            List<String> featureColumns,
            List<String> labelColumns,
            StandardizationStats standardizationStats,
            int initialGroupCapacity,
            NumericStorageType numericStorageType
    ) {
        this.dataFileName = requireNonblank(dataFileName, "dataFileName");
        this.hasMissingValues = hasMissingValues;
        this.isRegression = isRegression;
        this.idColumn = requireNonblank(idColumn, "idColumn");
        this.timeColumn = normalizeNullableString(timeColumn);
        this.featureColumns = copyColumns(featureColumns, "featureColumns", false);
        this.labelColumns = copyColumns(labelColumns, "labelColumns", true);
        this.standardizationStats = standardizationStats;
        this.requestedStorageType = Objects.requireNonNull(
                numericStorageType, "numericStorageType cannot be null.");
        if (initialGroupCapacity < 1) {
            throw new IllegalArgumentException(
                    "initialGroupCapacity must be at least 1.");
        }
        this.initialGroupCapacity = initialGroupCapacity;
        validateRoles();
        if (standardizationStats != null) {
            standardizationStats.validateFeatureCompatibility(this.featureColumns);
        }
        this.projectedColumns = buildProjectedColumns();
        this.projection = ColumnProjection.columns(
                projectedColumns.toArray(String[]::new));
    }

    @Override
    public ListObjectDataset read() throws IOException {
        long start = System.nanoTime();
        Path path = validateFile();
        ReadState state = new ReadState();

        try (ParquetFileReader fileReader =
                     ParquetFileReader.open(InputFile.of(path));
             ColumnReaders columns = fileReader.buildColumnReaders(projection)
                     .build()) {
            while (columns.nextBatch()) {
                int count = columns.getRecordCount();
                if (count == 0) {
                    continue;
                }
                Batch batch = decodeBatch(columns, count, state, path);
                appendBatch(batch, count, state, path);
                state.totalRecordCount += count;
            }
        } catch (IllegalArgumentException e) {
            throw e;
        } catch (RuntimeException e) {
            throw new IOException(
                    "Failed while decoding numeric long-format Parquet file: "
                            + path, e);
        }

        if (state.totalRecordCount == 0 || state.groups.isEmpty()) {
            throw new IOException(
                    "Numeric long-format Parquet file contains no records: "
                            + path);
        }

        ListObjectDataset dataset = buildDataset(state);
        LongFormatParquetReader.ProgressLogger.logDuration(
                start, System.nanoTime());
        return dataset;
    }

    private Batch decodeBatch(
            ColumnReaders columns,
            int count,
            ReadState state,
            Path path
    ) {
        ColumnReader idReader = columns.getColumnReader(idColumn);
        if (state.idKind == null) {
            state.idKind = detectKind(idReader, idColumn, false, path);
        }
        Object idValues = readValues(idReader, state.idKind, idColumn, path);
        Validity idValidity = idReader.getLeafValidity();

        Object timeValues = null;
        Validity timeValidity = Validity.NO_NULLS;
        if (timeColumn != null) {
            ColumnReader timeReader = columns.getColumnReader(timeColumn);
            if (state.timeKind == null) {
                state.timeKind = detectKind(timeReader, timeColumn, false, path);
            }
            timeValues = readValues(
                    timeReader, state.timeKind, timeColumn, path);
            timeValidity = timeReader.getLeafValidity();
        }

        if (state.featureKinds == null) {
            state.featureKinds = new ValueKind[featureColumns.size()];
            boolean allFloat = true;
            for (int feature = 0; feature < featureColumns.size(); feature++) {
                String name = featureColumns.get(feature);
                state.featureKinds[feature] = detectKind(
                        columns.getColumnReader(name), name, true, path);
                allFloat &= state.featureKinds[feature] == ValueKind.FLOAT;
            }
            state.storageType = requestedStorageType == NumericStorageType.AUTO
                    ? (allFloat ? NumericStorageType.FLOAT32
                    : NumericStorageType.FLOAT64)
                    : requestedStorageType;
        }

        Object[] featureValues = new Object[featureColumns.size()];
        Validity[] featureValidities = new Validity[featureColumns.size()];
        for (int feature = 0; feature < featureColumns.size(); feature++) {
            String name = featureColumns.get(feature);
            ColumnReader reader = columns.getColumnReader(name);
            featureValues[feature] = readValues(
                    reader, state.featureKinds[feature], name, path);
            featureValidities[feature] = reader.getLeafValidity();
        }

        if (state.labelKinds == null) {
            state.labelKinds = new ValueKind[labelColumns.size()];
            for (int label = 0; label < labelColumns.size(); label++) {
                String name = labelColumns.get(label);
                state.labelKinds[label] = detectKind(
                        columns.getColumnReader(name), name, false, path);
            }
        }
        Object[] labelValues = new Object[labelColumns.size()];
        Validity[] labelValidities = new Validity[labelColumns.size()];
        for (int label = 0; label < labelColumns.size(); label++) {
            String name = labelColumns.get(label);
            ColumnReader reader = columns.getColumnReader(name);
            labelValues[label] = readValues(
                    reader, state.labelKinds[label], name, path);
            labelValidities[label] = reader.getLeafValidity();
        }

        return new Batch(idValues, idValidity, timeValues, timeValidity,
                featureValues, featureValidities,
                labelValues, labelValidities);
    }

    private void appendBatch(
            Batch batch,
            int count,
            ReadState state,
            Path path
    ) {
        boolean idHasNulls = batch.idValidity.hasNulls();
        boolean timeHasNulls = timeColumn != null
                && batch.timeValidity.hasNulls();
        boolean[] featureHasNulls = new boolean[featureColumns.size()];
        for (int feature = 0; feature < featureColumns.size(); feature++) {
            featureHasNulls[feature] =
                    batch.featureValidities[feature].hasNulls();
        }
        boolean[] labelHasNulls = new boolean[labelColumns.size()];
        for (int label = 0; label < labelColumns.size(); label++) {
            labelHasNulls[label] = batch.labelValidities[label].hasNulls();
        }

        for (int row = 0; row < count; row++) {
            int recordIndex = state.totalRecordCount + row;
            if (idHasNulls && batch.idValidity.isNull(row)) {
                throw dataError("Missing ID", recordIndex, path);
            }
            Object id = valueAt(batch.idValues, state.idKind, row);
            if (id == null) {
                throw dataError("Missing ID", recordIndex, path);
            }

            Object time = null;
            if (timeColumn != null) {
                if (timeHasNulls && batch.timeValidity.isNull(row)) {
                    throw dataError("Missing configured time", recordIndex, path);
                }
                time = valueAt(batch.timeValues, state.timeKind, row);
            }

            Object label = materializeLabel(
                    batch, state, labelHasNulls, row);
            GroupAccumulator group = state.groups.computeIfAbsent(
                    id, ignored -> new GroupAccumulator(
                            featureColumns.size(), hasMissingValues,
                            initialGroupCapacity, state.storageType));
            group.validateLabel(label, id);

            for (int feature = 0; feature < featureColumns.size(); feature++) {
                boolean missing = featureHasNulls[feature]
                        && batch.featureValidities[feature].isNull(row);
                if (missing && !hasMissingValues) {
                    throw dataError(
                            "Null feature in column '"
                                    + featureColumns.get(feature)
                                    + "' while hasMissingValues=false",
                            recordIndex, path);
                }
                if (state.storageType == NumericStorageType.FLOAT32) {
                    float value = missing ? 0.0f : numericFloatAt(
                            batch.featureValues[feature],
                            state.featureKinds[feature], row);
                    group.addFloat(feature, value, missing);
                } else {
                    double value = missing ? 0.0d : numericDoubleAt(
                            batch.featureValues[feature],
                            state.featureKinds[feature], row);
                    group.addDouble(feature, value, missing);
                }
            }
            group.addOrdering(time, recordIndex);
            LongFormatParquetReader.ProgressLogger.logProgress(recordIndex);
        }
    }

    private Object materializeLabel(
            Batch batch,
            ReadState state,
            boolean[] labelHasNulls,
            int row
    ) {
        if (labelColumns.isEmpty()) {
            return null;
        }
        if (labelColumns.size() == 1) {
            return parseLabelAt(batch, state, labelHasNulls, 0, row);
        }
        List<Object> labels = new ArrayList<>(labelColumns.size());
        for (int label = 0; label < labelColumns.size(); label++) {
            labels.add(parseLabelAt(
                    batch, state, labelHasNulls, label, row));
        }
        return Collections.unmodifiableList(labels);
    }

    private Object parseLabelAt(
            Batch batch,
            ReadState state,
            boolean[] labelHasNulls,
            int label,
            int row
    ) {
        if (labelHasNulls[label]
                && batch.labelValidities[label].isNull(row)) {
            return null;
        }
        Object value = valueAt(
                batch.labelValues[label], state.labelKinds[label], row);
        if (value == null || !isRegression) {
            return value;
        }
        if (!(value instanceof Number number)) {
            throw new IllegalArgumentException(
                    "Regression label column '" + labelColumns.get(label)
                            + "' is not numeric.");
        }
        return number.doubleValue();
    }

    private ListObjectDataset buildDataset(ReadState state) {
        ListObjectDataset dataset = new ListObjectDataset(state.groups.size());
        int instance = 0;
        int commonLength = -1;
        boolean unequalLengths = false;
        for (GroupAccumulator group : state.groups.values()) {
            group.sortByTimeIfNeeded(timeColumn != null);
            dataset.add(group.label, group.toSeries(), instance++);
            int length = group.size();
            if (commonLength < 0) {
                commonLength = length;
            } else if (length != commonLength) {
                unequalLengths = true;
            }
        }
        int datasetLength = unequalLengths ? 0 : Math.max(commonLength, 0);
        dataset.setLength(datasetLength);
        AppContext.length = datasetLength;
        return dataset;
    }

    private static ValueKind detectKind(
            ColumnReader reader,
            String column,
            boolean numericOnly,
            Path path
    ) {
        RuntimeException last = null;
        for (ValueKind kind : numericOnly
                ? ValueKind.NUMERIC_DETECTION_ORDER
                : ValueKind.GENERIC_DETECTION_ORDER) {
            try {
                readValues(reader, kind, column, path);
                return kind;
            } catch (RuntimeException e) {
                last = e;
            }
        }
        throw new IllegalArgumentException(
                "Unsupported Parquet physical type for column '" + column
                        + "' in " + path + ".", last);
    }

    private static Object readValues(
            ColumnReader reader,
            ValueKind kind,
            String column,
            Path path
    ) {
        try {
            return switch (kind) {
                case DOUBLE -> reader.getDoubles();
                case FLOAT -> reader.getFloats();
                case LONG -> reader.getLongs();
                case INT -> reader.getInts();
                case BOOLEAN -> reader.getBooleans();
                case STRING -> reader.getStrings();
            };
        } catch (RuntimeException e) {
            throw new IllegalArgumentException(
                    "Column '" + column + "' in " + path
                            + " is incompatible with " + kind + ".", e);
        }
    }

    private static Object valueAt(Object values, ValueKind kind, int index) {
        return switch (kind) {
            case DOUBLE -> ((double[]) values)[index];
            case FLOAT -> ((float[]) values)[index];
            case LONG -> ((long[]) values)[index];
            case INT -> ((int[]) values)[index];
            case BOOLEAN -> ((boolean[]) values)[index];
            case STRING -> ((String[]) values)[index];
        };
    }

    private static float numericFloatAt(
            Object values,
            ValueKind kind,
            int index
    ) {
        return switch (kind) {
            case DOUBLE -> (float) ((double[]) values)[index];
            case FLOAT -> ((float[]) values)[index];
            case LONG -> (float) ((long[]) values)[index];
            case INT -> ((int[]) values)[index];
            default -> throw new IllegalArgumentException(
                    "Feature column is not numeric: " + kind);
        };
    }

    private static double numericDoubleAt(
            Object values,
            ValueKind kind,
            int index
    ) {
        return switch (kind) {
            case DOUBLE -> ((double[]) values)[index];
            case FLOAT -> ((float[]) values)[index];
            case LONG -> ((long[]) values)[index];
            case INT -> ((int[]) values)[index];
            default -> throw new IllegalArgumentException(
                    "Feature column is not numeric: " + kind);
        };
    }

    private static IllegalArgumentException dataError(
            String message,
            int record,
            Path path
    ) {
        return new IllegalArgumentException(
                message + " at Parquet record " + record + " in " + path + ".");
    }

    private List<String> buildProjectedColumns() {
        List<String> columns = new ArrayList<>();
        columns.add(idColumn);
        if (timeColumn != null) {
            columns.add(timeColumn);
        }
        columns.addAll(featureColumns);
        columns.addAll(labelColumns);
        return List.copyOf(columns);
    }

    private void validateRoles() {
        Set<String> used = new HashSet<>();
        addRole(used, idColumn);
        addRole(used, timeColumn);
        featureColumns.forEach(column -> addRole(used, column));
        labelColumns.forEach(column -> addRole(used, column));
    }

    private static void addRole(Set<String> used, String column) {
        if (column != null && !used.add(column)) {
            throw new IllegalArgumentException(
                    "Parquet column is assigned to multiple roles: " + column);
        }
    }

    private Path validateFile() throws IOException {
        Path path = Path.of(dataFileName);
        if (!Files.exists(path) || !Files.isRegularFile(path)
                || !Files.isReadable(path)) {
            throw new IOException(
                    "Numeric long-format Parquet file is not a readable regular file: "
                            + path);
        }
        return path;
    }

    private static ReaderOptions requireOptions(ReaderOptions options) {
        if (options == null) {
            throw new IllegalArgumentException(
                    "NumericLongFormatParquetReader requires ReaderOptions.");
        }
        return options;
    }

    private static String requireNonblank(String value, String role) {
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException(
                    "NumericLongFormatParquetReader requires " + role + ".");
        }
        return value.trim();
    }

    private static String normalizeNullableString(String value) {
        if (value == null) {
            return null;
        }
        String trimmed = value.trim();
        return trimmed.isEmpty() || trimmed.equalsIgnoreCase("None")
                ? null : trimmed;
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
                    "At least one " + role + " entry is required.");
        }
        List<String> copy = new ArrayList<>(columns.size());
        Set<String> used = new HashSet<>();
        for (String column : columns) {
            String normalized = requireNonblank(column, role + " entry");
            if (!used.add(normalized)) {
                throw new IllegalArgumentException(
                        role + " contains a duplicate: " + normalized);
            }
            copy.add(normalized);
        }
        return List.copyOf(copy);
    }

    private enum ValueKind {
        DOUBLE, FLOAT, LONG, INT, BOOLEAN, STRING;

        private static final ValueKind[] NUMERIC_DETECTION_ORDER = {
                DOUBLE, FLOAT, LONG, INT
        };
        private static final ValueKind[] GENERIC_DETECTION_ORDER = {
                STRING, LONG, INT, DOUBLE, FLOAT, BOOLEAN
        };
    }

    private static final class Batch {
        private final Object idValues;
        private final Validity idValidity;
        private final Object timeValues;
        private final Validity timeValidity;
        private final Object[] featureValues;
        private final Validity[] featureValidities;
        private final Object[] labelValues;
        private final Validity[] labelValidities;

        private Batch(
                Object idValues,
                Validity idValidity,
                Object timeValues,
                Validity timeValidity,
                Object[] featureValues,
                Validity[] featureValidities,
                Object[] labelValues,
                Validity[] labelValidities
        ) {
            this.idValues = idValues;
            this.idValidity = idValidity;
            this.timeValues = timeValues;
            this.timeValidity = timeValidity;
            this.featureValues = featureValues;
            this.featureValidities = featureValidities;
            this.labelValues = labelValues;
            this.labelValidities = labelValidities;
        }
    }

    private static final class ReadState {
        private final Map<Object, GroupAccumulator> groups =
                new LinkedHashMap<>();
        private ValueKind idKind;
        private ValueKind timeKind;
        private ValueKind[] featureKinds;
        private ValueKind[] labelKinds;
        private NumericStorageType storageType;
        private int totalRecordCount;
    }

    private static final class GroupAccumulator {
        private final NumericBuffer[] features;
        private final MissingBuffer[] missing;
        private final List<Object> times;
        private final IntBuffer inputOrders;
        private final boolean trackMissing;
        private Object label;
        private boolean labelInitialized;

        private GroupAccumulator(
                int featureCount,
                boolean trackMissing,
                int initialCapacity,
                NumericStorageType storageType
        ) {
            this.trackMissing = trackMissing;
            this.features = new NumericBuffer[featureCount];
            this.missing = trackMissing
                    ? new MissingBuffer[featureCount] : null;
            for (int i = 0; i < featureCount; i++) {
                features[i] = storageType == NumericStorageType.FLOAT32
                        ? new FloatBuffer(initialCapacity)
                        : new DoubleBuffer(initialCapacity);
                if (trackMissing) {
                    missing[i] = new MissingBuffer(initialCapacity);
                }
            }
            this.times = new ArrayList<>(initialCapacity);
            this.inputOrders = new IntBuffer(initialCapacity);
        }

        private void validateLabel(Object rowLabel, Object id) {
            if (!labelInitialized) {
                label = rowLabel;
                labelInitialized = true;
            } else if (!Objects.equals(label, rowLabel)) {
                throw new IllegalArgumentException(
                        "Inconsistent labels in numeric Parquet group for ID: "
                                + id);
            }
        }

        private void addFloat(
                int feature,
                float value,
                boolean isMissing
        ) {
            ((FloatBuffer) features[feature]).add(value);
            if (trackMissing) {
                missing[feature].add(isMissing);
            }
        }

        private void addDouble(
                int feature,
                double value,
                boolean isMissing
        ) {
            ((DoubleBuffer) features[feature]).add(value);
            if (trackMissing) {
                missing[feature].add(isMissing);
            }
        }

        private void addOrdering(Object time, int inputOrder) {
            times.add(time);
            inputOrders.add(inputOrder);
        }

        private int size() {
            return features[0].size();
        }

        private void sortByTimeIfNeeded(boolean hasTime) {
            if (!hasTime || size() < 2) {
                return;
            }
            Integer[] order = new Integer[size()];
            for (int i = 0; i < order.length; i++) {
                order[i] = i;
            }
            Arrays.sort(order, Comparator
                    .comparing((Integer i) -> times.get(i),
                            GroupAccumulator::compareTimes)
                    .thenComparingInt(inputOrders::get));
            boolean sorted = true;
            for (int i = 0; i < order.length; i++) {
                if (order[i] != i) {
                    sorted = false;
                    break;
                }
            }
            if (sorted) {
                return;
            }
            for (NumericBuffer feature : features) {
                feature.reorder(order);
            }
            if (trackMissing) {
                for (MissingBuffer buffer : missing) {
                    buffer.reorder(order);
                }
            }
        }

        private Object toSeries() {
            MissingBuffer firstMissing = trackMissing ? missing[0] : null;
            if (features.length == 1) {
                return features[0].toArray(firstMissing);
            }
            if (features[0] instanceof FloatBuffer) {
                float[][] result = new float[features.length][];
                for (int i = 0; i < features.length; i++) {
                    result[i] = (float[]) features[i].toArray(
                            trackMissing ? missing[i] : null);
                }
                return result;
            }
            double[][] result = new double[features.length][];
            for (int i = 0; i < features.length; i++) {
                result[i] = (double[]) features[i].toArray(
                        trackMissing ? missing[i] : null);
            }
            return result;
        }

        @SuppressWarnings({"rawtypes", "unchecked"})
        private static int compareTimes(Object first, Object second) {
            if (first instanceof Number && second instanceof Number) {
                return Double.compare(((Number) first).doubleValue(),
                        ((Number) second).doubleValue());
            }
            if (first instanceof Comparable
                    && first.getClass().isInstance(second)) {
                return ((Comparable) first).compareTo(second);
            }
            return first.toString().compareTo(second.toString());
        }
    }

    private interface NumericBuffer {
        int size();
        Object toArray(MissingBuffer missing);
        void reorder(Integer[] order);
    }

    private static final class FloatBuffer implements NumericBuffer {
        private float[] values;
        private int size;

        private FloatBuffer(int capacity) {
            values = new float[Math.max(1, capacity)];
        }

        private void add(float value) {
            ensure(size + 1);
            values[size++] = value;
        }

        @Override
        public int size() {
            return size;
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

        @Override
        public void reorder(Integer[] order) {
            float[] reordered = new float[size];
            for (int i = 0; i < size; i++) {
                reordered[i] = values[order[i]];
            }
            values = reordered;
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

        private void add(double value) {
            ensure(size + 1);
            values[size++] = value;
        }

        @Override
        public int size() {
            return size;
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

        @Override
        public void reorder(Integer[] order) {
            double[] reordered = new double[size];
            for (int i = 0; i < size; i++) {
                reordered[i] = values[order[i]];
            }
            values = reordered;
        }

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
            ensure(size + 1);
            values[size++] = value;
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

        private void reorder(Integer[] order) {
            boolean[] reordered = new boolean[size];
            for (int i = 0; i < size; i++) {
                reordered[i] = values[order[i]];
            }
            values = reordered;
        }

        private void ensure(int required) {
            if (required > values.length) {
                values = Arrays.copyOf(values,
                        nextCapacity(values.length, required));
            }
        }
    }

    private static final class IntBuffer {
        private int[] values;
        private int size;

        private IntBuffer(int capacity) {
            values = new int[Math.max(1, capacity)];
        }

        private void add(int value) {
            ensure(size + 1);
            values[size++] = value;
        }

        private int get(int index) {
            return values[index];
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
                    "Required numeric Parquet buffer is too large.");
        }
        return expanded;
    }
}
