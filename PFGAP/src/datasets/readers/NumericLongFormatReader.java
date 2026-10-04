package datasets.readers;

import ch.randelshofer.fastdoubleparser.JavaDoubleParser;
import ch.randelshofer.fastdoubleparser.JavaFloatParser;
import core.AppContext;
import datasets.ListObjectDataset;
import datasets.NumericStorageType;
import de.siegmar.fastcsv.reader.AbstractBaseCsvCallbackHandler;
import de.siegmar.fastcsv.reader.CsvReader;
import preprocessing.standardization.StandardizationStats;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * High-throughput reader for grouped numeric long-format delimited data.
 *
 * <p>Each record represents one position in an observation. Records are
 * grouped by the configured ID column and sorted by the configured time column
 * when one is present. Without a time column, source-record order is retained.
 * Explicit time values are used for ordering and are not included in the
 * materialized feature arrays.</p>
 *
 * <p>Output is primitive-only:</p>
 * <ul>
 *     <li>FLOAT64: {@code double[]} or {@code double[][]}</li>
 *     <li>FLOAT32: {@code float[]} or {@code float[][]}</li>
 * </ul>
 *
 * <p>FLOAT32 values are parsed directly with {@link JavaFloatParser} and
 * accumulated in primitive float buffers. FLOAT64 values are parsed with
 * {@link JavaDoubleParser} and accumulated in primitive double buffers.
 * Numeric missing values are represented by the corresponding primitive NaN.
 * {@link NumericStorageType#AUTO} resolves to FLOAT64 for delimited text.</p>
 *
 * <p>IDs are preserved as trimmed strings so textually distinct identifiers,
 * such as {@code 001} and {@code 1}, remain distinct. A configured time column
 * must be nonmissing. Multiple label columns are preserved as an immutable
 * list for future multi-label and multi-target workflows.</p>
 */
public class NumericLongFormatReader implements DatasetReader {
    private static final int DEFAULT_INITIAL_GROUP_CAPACITY = 256;

    private final String dataFileName;
    private final char fieldSeparator;
    private final boolean hasHeader;
    private final boolean hasMissingValues;
    private final boolean isRegression;
    private final NumericStorageType numericStorageType;
    private final String idColumn;
    private final String timeColumn;
    private final List<String> featureColumns;
    private final List<String> labelColumns;
    private final int initialGroupCapacity;
    private final Set<String> missingIndicators;

    public NumericLongFormatReader(ReaderOptions options) {
        this(
                requireOptions(options).getDataPath(),
                options.getEntrySeparator(),
                options.hasHeader(),
                options.hasMissingValues(),
                options.isRegression(),
                options.getIdColumn(),
                options.getTimeColumn(),
                options.getFeatureColumns(),
                options.getLabelColumns(),
                options.getStandardizationStats(),
                DEFAULT_INITIAL_GROUP_CAPACITY,
                options.getNumericStorageType()
        );
        if (!options.isNumeric()) {
            throw new IllegalArgumentException(
                    "NumericLongFormatReader requires ReaderOptions.isNumeric=true.");
        }
    }

    public NumericLongFormatReader(
            String dataFileName,
            String entrySeparator,
            boolean hasHeader,
            boolean hasMissingValues,
            boolean isRegression,
            String idColumn,
            String timeColumn,
            List<String> featureColumns,
            List<String> labelColumns,
            StandardizationStats standardizationStats
    ) {
        this(dataFileName, entrySeparator, hasHeader, hasMissingValues,
                isRegression, idColumn, timeColumn, featureColumns,
                labelColumns, standardizationStats,
                DEFAULT_INITIAL_GROUP_CAPACITY, NumericStorageType.AUTO);
    }

    public NumericLongFormatReader(
            String dataFileName,
            String entrySeparator,
            boolean hasHeader,
            boolean hasMissingValues,
            boolean isRegression,
            String idColumn,
            String timeColumn,
            List<String> featureColumns,
            List<String> labelColumns,
            StandardizationStats standardizationStats,
            int initialGroupCapacity
    ) {
        this(dataFileName, entrySeparator, hasHeader, hasMissingValues,
                isRegression, idColumn, timeColumn, featureColumns,
                labelColumns, standardizationStats, initialGroupCapacity,
                NumericStorageType.AUTO);
    }

    public NumericLongFormatReader(
            String dataFileName,
            String entrySeparator,
            boolean hasHeader,
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
        String normalizedSeparator = validateAndNormalizeSeparator(entrySeparator);
        this.fieldSeparator = normalizedSeparator.charAt(0);
        this.hasHeader = hasHeader;
        this.hasMissingValues = hasMissingValues;
        this.isRegression = isRegression;
        this.numericStorageType = resolveStorageType(numericStorageType);
        this.idColumn = requireNonblank(idColumn, "idColumn");
        this.timeColumn = normalizeNullableString(timeColumn);
        this.featureColumns = copyColumns(featureColumns, "featureColumns", false);
        this.labelColumns = copyColumns(labelColumns, "labelColumns", true);

        if (standardizationStats != null) {
            standardizationStats.validateFeatureCompatibility(this.featureColumns);
        }
        if (initialGroupCapacity < 1) {
            throw new IllegalArgumentException(
                    "NumericLongFormatReader initialGroupCapacity must be at least 1. Received: "
                            + initialGroupCapacity + ".");
        }
        this.initialGroupCapacity = initialGroupCapacity;
        this.missingIndicators = snapshotMissingIndicators();
        validateOptions();
    }

    @Override
    public ListObjectDataset read() throws IOException {
        long start = System.nanoTime();
        Path file = validateDataFile();
        NumericLongFormatCallbackHandler handler =
                new NumericLongFormatCallbackHandler(
                        file, idColumn, timeColumn, featureColumns, labelColumns,
                        hasMissingValues, isRegression, missingIndicators,
                        initialGroupCapacity, numericStorageType);

        try (CsvReader<Boolean> csvReader = CsvReader.builder()
                .fieldSeparator(fieldSeparator)
                .skipEmptyLines(true)
                .detectBomHeader(true)
                .build(handler, file)) {
            for (Boolean ignored : csvReader) {
                // Consuming the iterator drives callback parsing.
            }
        } catch (IllegalArgumentException e) {
            throw e;
        } catch (RuntimeException e) {
            throw new IOException(
                    "Failed while parsing numeric long-format file: " + file, e);
        }

        ListObjectDataset dataset = handler.buildDataset();
        LongFormatReader.ProgressLogger.logDuration(start, System.nanoTime());
        return dataset;
    }

    private void validateOptions() {
        if (!hasHeader) {
            throw new IllegalArgumentException(
                    "NumericLongFormatReader currently requires hasHeader=true.");
        }
        if (featureColumns.isEmpty()) {
            throw new IllegalArgumentException(
                    "NumericLongFormatReader requires at least one feature column.");
        }

        Set<String> reserved = new HashSet<>();
        reserved.add(idColumn);
        if (timeColumn != null && !reserved.add(timeColumn)) {
            throw new IllegalArgumentException(
                    "idColumn and timeColumn cannot identify the same column.");
        }
        for (String column : featureColumns) {
            if (!reserved.add(column)) {
                throw new IllegalArgumentException(
                        "Feature column overlaps an ID, time, or previously selected feature column: "
                                + column);
            }
        }
        for (String column : labelColumns) {
            if (!reserved.add(column)) {
                throw new IllegalArgumentException(
                        "Label column overlaps an ID, time, feature, or previously selected label column: "
                                + column);
            }
        }
    }

    private Path validateDataFile() throws IOException {
        Path path = Path.of(dataFileName);
        if (!Files.exists(path)) {
            throw new IOException("Numeric long-format file does not exist: " + path);
        }
        if (!Files.isRegularFile(path)) {
            throw new IOException("Numeric long-format path is not a regular file: " + path);
        }
        if (!Files.isReadable(path)) {
            throw new IOException("Numeric long-format file is not readable: " + path);
        }
        return path;
    }

    private static final class NumericLongFormatCallbackHandler
            extends AbstractBaseCsvCallbackHandler<Boolean> {
        private final Path file;
        private final String idColumn;
        private final String timeColumn;
        private final List<String> featureColumns;
        private final List<String> labelColumns;
        private final boolean hasMissingValues;
        private final boolean isRegression;
        private final Set<String> missingIndicators;
        private final int initialGroupCapacity;
        private final NumericStorageType storageType;
        private final Map<String, GroupAccumulator> groups = new LinkedHashMap<>();

        private List<String> headerFields = new ArrayList<>();
        private int columnCount;
        private int idIndex = -1;
        private int timeIndex = -1;
        private int[] columnToFeature;
        private int[] columnToLabel;
        private float[] currentFloatFeatures;
        private double[] currentDoubleFeatures;
        private boolean[] currentMissing;
        private Object[] currentLabels;
        private String currentId;
        private Object currentTime;
        private boolean schemaResolved;
        private int inputOrder;
        private int dataRecordCount;

        private NumericLongFormatCallbackHandler(
                Path file,
                String idColumn,
                String timeColumn,
                List<String> featureColumns,
                List<String> labelColumns,
                boolean hasMissingValues,
                boolean isRegression,
                Set<String> missingIndicators,
                int initialGroupCapacity,
                NumericStorageType storageType
        ) {
            this.file = file;
            this.idColumn = idColumn;
            this.timeColumn = timeColumn;
            this.featureColumns = featureColumns;
            this.labelColumns = labelColumns;
            this.hasMissingValues = hasMissingValues;
            this.isRegression = isRegression;
            this.missingIndicators = missingIndicators;
            this.initialGroupCapacity = initialGroupCapacity;
            this.storageType = storageType;
        }

        @Override
        public void handleField(
                int fieldIndex,
                char[] buffer,
                int offset,
                int length,
                boolean quoted
        ) {
            if (!schemaResolved) {
                headerFields.add(new String(buffer, offset, length));
                return;
            }
            if (fieldIndex >= columnCount) {
                throw inconsistentColumnCount(fieldIndex + 1);
            }
            if (fieldIndex == idIndex) {
                currentId = parseIdentifier(buffer, offset, length);
                return;
            }
            if (fieldIndex == timeIndex) {
                currentTime = parseRequiredTime(buffer, offset, length);
                return;
            }
            int featureIndex = columnToFeature[fieldIndex];
            if (featureIndex >= 0) {
                parseFeature(featureIndex, fieldIndex, buffer, offset, length);
                return;
            }
            int labelIndex = columnToLabel[fieldIndex];
            if (labelIndex >= 0) {
                currentLabels[labelIndex] = parseLabel(buffer, offset, length);
            }
        }

        @Override
        protected Boolean buildRecord() {
            int actualColumnCount = getFieldCount();
            if (!schemaResolved) {
                resolveSchema(actualColumnCount);
                schemaResolved = true;
                headerFields = null;
                return null;
            }
            if (actualColumnCount != columnCount) {
                throw inconsistentColumnCount(actualColumnCount);
            }
            if (currentId == null) {
                throw recordError("Encountered a missing ID");
            }
            if (timeIndex >= 0 && currentTime == null) {
                throw recordError("Encountered a missing configured time value");
            }

            Object label = materializeCurrentLabel();
            GroupAccumulator group = groups.computeIfAbsent(
                    currentId,
                    ignored -> new GroupAccumulator(
                            featureColumns.size(), hasMissingValues,
                            initialGroupCapacity, storageType));
            group.append(currentFloatFeatures, currentDoubleFeatures,
                    currentMissing, currentTime, label, inputOrder, currentId);

            resetCurrentRecord();
            dataRecordCount++;
            inputOrder++;
            LongFormatReader.ProgressLogger.logProgress(dataRecordCount - 1);
            return Boolean.TRUE;
        }

        private void resolveSchema(int actualColumnCount) {
            if (actualColumnCount <= 0) {
                throw new IllegalArgumentException(
                        "Numeric long-format file has no columns: " + file);
            }
            if (headerFields.size() != actualColumnCount) {
                throw new IllegalStateException(
                        "FastCSV header field count mismatch in file " + file
                                + ". FastCSV count=" + actualColumnCount
                                + ", buffered count=" + headerFields.size() + ".");
            }

            Map<String, Integer> headerIndex = buildHeaderIndex();
            columnCount = actualColumnCount;
            idIndex = requireColumn(headerIndex, idColumn, "idColumn");
            timeIndex = timeColumn == null
                    ? -1
                    : requireColumn(headerIndex, timeColumn, "timeColumn");
            columnToFeature = new int[columnCount];
            columnToLabel = new int[columnCount];
            Arrays.fill(columnToFeature, -1);
            Arrays.fill(columnToLabel, -1);

            for (int i = 0; i < featureColumns.size(); i++) {
                columnToFeature[requireColumn(
                        headerIndex, featureColumns.get(i), "featureColumns")] = i;
            }
            for (int i = 0; i < labelColumns.size(); i++) {
                columnToLabel[requireColumn(
                        headerIndex, labelColumns.get(i), "labelColumns")] = i;
            }

            if (storageType == NumericStorageType.FLOAT32) {
                currentFloatFeatures = new float[featureColumns.size()];
            } else {
                currentDoubleFeatures = new double[featureColumns.size()];
            }
            currentMissing = hasMissingValues
                    ? new boolean[featureColumns.size()]
                    : null;
            currentLabels = new Object[labelColumns.size()];
        }

        private Map<String, Integer> buildHeaderIndex() {
            Map<String, Integer> index = new HashMap<>(
                    Math.max(16, headerFields.size() * 2));
            for (int i = 0; i < headerFields.size(); i++) {
                String raw = headerFields.get(i);
                String name = raw == null ? "" : raw.trim();
                if (name.isEmpty()) {
                    throw new IllegalArgumentException(
                            "Numeric long-format file contains a blank header at column "
                                    + i + ": " + file);
                }
                Integer previous = index.put(name, i);
                if (previous != null) {
                    throw new IllegalArgumentException(
                            "Numeric long-format file contains duplicate header '"
                                    + name + "': " + file);
                }
            }
            return index;
        }

        private int requireColumn(
                Map<String, Integer> headerIndex,
                String columnName,
                String optionName
        ) {
            Integer index = headerIndex.get(columnName);
            if (index == null) {
                throw new IllegalArgumentException(
                        "Column not found for " + optionName + ": " + columnName
                                + ". Available columns: " + headerIndex.keySet());
            }
            return index;
        }

        private void parseFeature(
                int featureIndex,
                int fieldIndex,
                char[] buffer,
                int offset,
                int length
        ) {
            TrimmedRange range = trimRange(buffer, offset, length);
            if (isMissing(buffer, range.offset, range.length)) {
                if (!hasMissingValues) {
                    throw recordError(
                            "Encountered a missing numeric feature at column "
                                    + fieldIndex + " while hasMissingValues=false");
                }
                currentMissing[featureIndex] = true;
                if (storageType == NumericStorageType.FLOAT32) {
                    currentFloatFeatures[featureIndex] = 0.0f;
                } else {
                    currentDoubleFeatures[featureIndex] = 0.0d;
                }
                return;
            }

            try {
                if (storageType == NumericStorageType.FLOAT32) {
                    currentFloatFeatures[featureIndex] =
                            JavaFloatParser.parseFloat(
                                    buffer, range.offset, range.length);
                } else {
                    currentDoubleFeatures[featureIndex] =
                            JavaDoubleParser.parseDouble(
                                    buffer, range.offset, range.length);
                }
                if (currentMissing != null) {
                    currentMissing[featureIndex] = false;
                }
            } catch (NumberFormatException e) {
                throw new IllegalArgumentException(
                        recordLocation("Could not parse numeric feature at column "
                                + fieldIndex), e);
            }
        }

        private String parseIdentifier(
                char[] buffer,
                int offset,
                int length
        ) {
            TrimmedRange range = trimRange(buffer, offset, length);
            if (isMissing(buffer, range.offset, range.length)) {
                return null;
            }
            return new String(buffer, range.offset, range.length);
        }

        private Object parseRequiredTime(
                char[] buffer,
                int offset,
                int length
        ) {
            TrimmedRange range = trimRange(buffer, offset, length);
            if (isMissing(buffer, range.offset, range.length)) {
                throw recordError("Encountered a missing configured time value");
            }
            return parseGenericString(
                    new String(buffer, range.offset, range.length));
        }

        private Object parseLabel(
                char[] buffer,
                int offset,
                int length
        ) {
            TrimmedRange range = trimRange(buffer, offset, length);
            if (isMissing(buffer, range.offset, range.length)) {
                return null;
            }
            String value = new String(buffer, range.offset, range.length);
            if (isRegression) {
                return JavaDoubleParser.parseDouble(value);
            }
            try {
                return Integer.parseInt(value);
            } catch (NumberFormatException ignored) {
                try {
                    double numeric = JavaDoubleParser.parseDouble(value);
                    if (numeric == Math.rint(numeric)
                            && numeric >= Integer.MIN_VALUE
                            && numeric <= Integer.MAX_VALUE) {
                        return (int) numeric;
                    }
                    return numeric;
                } catch (NumberFormatException ignoredAgain) {
                    return value;
                }
            }
        }

        private Object parseGenericString(String value) {
            try {
                return Integer.parseInt(value);
            } catch (NumberFormatException ignored) {
                try {
                    return JavaDoubleParser.parseDouble(value);
                } catch (NumberFormatException ignoredAgain) {
                    if (value.equalsIgnoreCase("true")
                            || value.equalsIgnoreCase("false")) {
                        return Boolean.parseBoolean(value);
                    }
                    return value;
                }
            }
        }

        private Object materializeCurrentLabel() {
            if (currentLabels.length == 0) {
                return null;
            }
            if (currentLabels.length == 1) {
                return currentLabels[0];
            }
            List<Object> labels = new ArrayList<>(currentLabels.length);
            labels.addAll(Arrays.asList(currentLabels.clone()));
            return Collections.unmodifiableList(labels);
        }

        private void resetCurrentRecord() {
            currentId = null;
            currentTime = null;
            if (currentFloatFeatures != null) {
                Arrays.fill(currentFloatFeatures, 0.0f);
            }
            if (currentDoubleFeatures != null) {
                Arrays.fill(currentDoubleFeatures, 0.0d);
            }
            if (currentMissing != null) {
                Arrays.fill(currentMissing, false);
            }
            Arrays.fill(currentLabels, null);
        }

        private boolean isMissing(char[] buffer, int offset, int length) {
            if (length == 0) {
                return true;
            }
            if (missingIndicators.isEmpty()) {
                return false;
            }
            String token = new String(buffer, offset, length)
                    .toUpperCase(Locale.ROOT);
            return missingIndicators.contains(token);
        }

        private TrimmedRange trimRange(char[] buffer, int offset, int length) {
            int start = offset;
            int end = offset + length;
            while (start < end && Character.isWhitespace(buffer[start])) {
                start++;
            }
            while (end > start && Character.isWhitespace(buffer[end - 1])) {
                end--;
            }
            return new TrimmedRange(start, end - start);
        }

        private IllegalArgumentException inconsistentColumnCount(int actual) {
            return new IllegalArgumentException(
                    "Inconsistent column count in numeric long-format file "
                            + file + " at CSV record beginning on line "
                            + getStartingLineNumber() + ". Expected "
                            + columnCount + " columns but found " + actual + ".");
        }

        private IllegalArgumentException recordError(String message) {
            return new IllegalArgumentException(recordLocation(message));
        }

        private String recordLocation(String message) {
            return message + " in numeric long-format file " + file
                    + " at data record " + dataRecordCount
                    + ", beginning on CSV line " + getStartingLineNumber() + ".";
        }

        private ListObjectDataset buildDataset() {
            if (!schemaResolved) {
                throw new IllegalArgumentException(
                        "Numeric long-format file is empty: " + file);
            }
            if (dataRecordCount == 0 || groups.isEmpty()) {
                throw new IllegalArgumentException(
                        "Numeric long-format file contains no data records: " + file);
            }

            ListObjectDataset dataset = new ListObjectDataset(groups.size());
            int instanceIndex = 0;
            int commonLength = -1;
            boolean unequalLengths = false;

            for (GroupAccumulator group : groups.values()) {
                group.sortByTimeIfNeeded(timeColumn != null);
                dataset.add(group.getLabel(), group.toSeries(), instanceIndex++);
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
    }

    private static final class GroupAccumulator {
        private final NumericBuffer[] featureValues;
        private final MissingBuffer[] missingPositions;
        private final boolean nullableOutput;
        private final List<Object> timeValues;
        private final IntBuffer inputOrders;
        private Object label;
        private boolean labelInitialized;

        private GroupAccumulator(
                int featureCount,
                boolean nullableOutput,
                int initialCapacity,
                NumericStorageType storageType
        ) {
            this.nullableOutput = nullableOutput;
            this.featureValues = new NumericBuffer[featureCount];
            this.missingPositions = nullableOutput
                    ? new MissingBuffer[featureCount]
                    : null;
            for (int i = 0; i < featureCount; i++) {
                featureValues[i] = storageType == NumericStorageType.FLOAT32
                        ? new PrimitiveFloatBuffer(initialCapacity)
                        : new PrimitiveDoubleBuffer(initialCapacity);
                if (nullableOutput) {
                    missingPositions[i] = new MissingBuffer(initialCapacity);
                }
            }
            this.timeValues = new ArrayList<>(initialCapacity);
            this.inputOrders = new IntBuffer(initialCapacity);
        }

        private void append(
                float[] floatValues,
                double[] doubleValues,
                boolean[] missing,
                Object timeValue,
                Object rowLabel,
                int inputOrder,
                String id
        ) {
            validateLabel(rowLabel, id);
            for (int i = 0; i < featureValues.length; i++) {
                if (featureValues[i] instanceof PrimitiveFloatBuffer buffer) {
                    buffer.add(floatValues[i]);
                } else if (featureValues[i] instanceof PrimitiveDoubleBuffer buffer) {
                    buffer.add(doubleValues[i]);
                } else {
                    throw new IllegalStateException("Unsupported numeric buffer type.");
                }
                if (nullableOutput) {
                    missingPositions[i].add(missing[i]);
                }
            }
            timeValues.add(timeValue);
            inputOrders.add(inputOrder);
        }

        private void validateLabel(Object rowLabel, String id) {
            if (!labelInitialized) {
                label = rowLabel;
                labelInitialized = true;
                return;
            }
            if (!Objects.equals(label, rowLabel)) {
                throw new IllegalArgumentException(
                        "Inconsistent labels found within numeric long-format group for id: "
                                + id);
            }
        }

        private Object getLabel() {
            return label;
        }

        private int size() {
            return featureValues.length == 0 ? 0 : featureValues[0].size();
        }

        private void sortByTimeIfNeeded(boolean hasTimeColumn) {
            if (!hasTimeColumn || size() < 2) {
                return;
            }
            Integer[] order = new Integer[size()];
            for (int i = 0; i < order.length; i++) {
                order[i] = i;
            }
            Arrays.sort(order,
                    Comparator.comparing(
                                    (Integer index) -> timeValues.get(index),
                                    GroupAccumulator::compareTimeValues)
                            .thenComparingInt(inputOrders::get));

            boolean alreadySorted = true;
            for (int i = 0; i < order.length; i++) {
                if (order[i] != i) {
                    alreadySorted = false;
                    break;
                }
            }
            if (alreadySorted) {
                return;
            }

            for (NumericBuffer values : featureValues) {
                values.reorder(order);
            }
            if (nullableOutput) {
                for (MissingBuffer missing : missingPositions) {
                    missing.reorder(order);
                }
            }

            List<Object> reorderedTimes = new ArrayList<>(order.length);
            int[] reorderedOrders = new int[order.length];
            for (int output = 0; output < order.length; output++) {
                int source = order[output];
                reorderedTimes.add(timeValues.get(source));
                reorderedOrders[output] = inputOrders.get(source);
            }
            timeValues.clear();
            timeValues.addAll(reorderedTimes);
            inputOrders.replaceWith(reorderedOrders);
        }

        private Object toSeries() {
            if (featureValues.length == 1) {
                return featureValues[0].toArray(
                        nullableOutput ? missingPositions[0] : null);
            }
            if (featureValues[0] instanceof PrimitiveFloatBuffer) {
                float[][] result = new float[featureValues.length][];
                for (int i = 0; i < featureValues.length; i++) {
                    result[i] = (float[]) featureValues[i].toArray(
                            nullableOutput ? missingPositions[i] : null);
                }
                return result;
            }
            double[][] result = new double[featureValues.length][];
            for (int i = 0; i < featureValues.length; i++) {
                result[i] = (double[]) featureValues[i].toArray(
                        nullableOutput ? missingPositions[i] : null);
            }
            return result;
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
    }

    private interface NumericBuffer {
        int size();
        Object toArray(MissingBuffer missing);
        void reorder(Integer[] order);
    }

    private static final class PrimitiveFloatBuffer implements NumericBuffer {
        private float[] values;
        private int size;

        private PrimitiveFloatBuffer(int initialCapacity) {
            values = new float[Math.max(1, initialCapacity)];
        }

        private void add(float value) {
            ensureCapacity(size + 1);
            values[size++] = value;
        }

        @Override
        public int size() {
            return size;
        }

        @Override
        public Object toArray(MissingBuffer missing) {
            validateMissingLength(missing, size);
            float[] result = Arrays.copyOf(values, size);
            if (missing != null) {
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
            for (int i = 0; i < order.length; i++) {
                reordered[i] = values[order[i]];
            }
            values = reordered;
        }

        private void ensureCapacity(int required) {
            if (required > values.length) {
                values = Arrays.copyOf(values,
                        nextCapacity(values.length, required));
            }
        }
    }

    private static final class PrimitiveDoubleBuffer implements NumericBuffer {
        private double[] values;
        private int size;

        private PrimitiveDoubleBuffer(int initialCapacity) {
            values = new double[Math.max(1, initialCapacity)];
        }

        private void add(double value) {
            ensureCapacity(size + 1);
            values[size++] = value;
        }

        @Override
        public int size() {
            return size;
        }

        @Override
        public Object toArray(MissingBuffer missing) {
            validateMissingLength(missing, size);
            double[] result = Arrays.copyOf(values, size);
            if (missing != null) {
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
            for (int i = 0; i < order.length; i++) {
                reordered[i] = values[order[i]];
            }
            values = reordered;
        }

        private void ensureCapacity(int required) {
            if (required > values.length) {
                values = Arrays.copyOf(values,
                        nextCapacity(values.length, required));
            }
        }
    }

    private static void validateMissingLength(MissingBuffer missing, int size) {
        if (missing != null && missing.size() != size) {
            throw new IllegalStateException(
                    "Numeric and missing-position buffers have different lengths.");
        }
    }

    private static final class MissingBuffer {
        private boolean[] values;
        private int size;

        private MissingBuffer(int initialCapacity) {
            values = new boolean[Math.max(1, initialCapacity)];
        }

        private void add(boolean value) {
            ensureCapacity(size + 1);
            values[size++] = value;
        }

        private boolean get(int index) {
            return values[index];
        }

        private int size() {
            return size;
        }

        private void reorder(Integer[] order) {
            boolean[] reordered = new boolean[size];
            for (int i = 0; i < order.length; i++) {
                reordered[i] = values[order[i]];
            }
            values = reordered;
        }

        private void ensureCapacity(int required) {
            if (required > values.length) {
                values = Arrays.copyOf(values,
                        nextCapacity(values.length, required));
            }
        }
    }

    private static final class IntBuffer {
        private int[] values;
        private int size;

        private IntBuffer(int initialCapacity) {
            values = new int[Math.max(1, initialCapacity)];
        }

        private void add(int value) {
            ensureCapacity(size + 1);
            values[size++] = value;
        }

        private int get(int index) {
            return values[index];
        }

        private void replaceWith(int[] replacement) {
            values = replacement;
            size = replacement.length;
        }

        private void ensureCapacity(int required) {
            if (required > values.length) {
                values = Arrays.copyOf(values,
                        nextCapacity(values.length, required));
            }
        }
    }

    private static int nextCapacity(int current, int required) {
        int expanded = current <= Integer.MAX_VALUE / 2
                ? current << 1
                : Integer.MAX_VALUE;
        if (expanded < required) {
            expanded = required;
        }
        if (expanded < 0 || expanded < current) {
            throw new OutOfMemoryError(
                    "Required numeric long-format buffer is too large.");
        }
        return expanded;
    }

    private static final class TrimmedRange {
        private final int offset;
        private final int length;

        private TrimmedRange(int offset, int length) {
            this.offset = offset;
            this.length = length;
        }
    }

    private static ReaderOptions requireOptions(ReaderOptions options) {
        if (options == null) {
            throw new IllegalArgumentException(
                    "NumericLongFormatReader requires non-null ReaderOptions.");
        }
        return options;
    }

    private static NumericStorageType resolveStorageType(
            NumericStorageType requested
    ) {
        NumericStorageType value = Objects.requireNonNull(
                requested, "NumericStorageType cannot be null.");
        return value == NumericStorageType.AUTO
                ? NumericStorageType.FLOAT64
                : value;
    }

    private static String requireNonblank(String value, String argumentName) {
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException(
                    "NumericLongFormatReader requires " + argumentName + ".");
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

    private static String validateAndNormalizeSeparator(String separator) {
        if (separator == null || separator.isEmpty()) {
            throw new IllegalArgumentException(
                    "NumericLongFormatReader requires entrySeparator.");
        }
        String normalized = switch (separator) {
            case "\\t" -> "\t";
            case "\\n" -> "\n";
            case "\\r" -> "\r";
            default -> separator;
        };
        if (normalized.length() != 1) {
            throw new IllegalArgumentException(
                    "NumericLongFormatReader requires a single-character entrySeparator. Received: '"
                            + separator + "'.");
        }
        char value = normalized.charAt(0);
        if (value == '\n' || value == '\r') {
            throw new IllegalArgumentException(
                    "entrySeparator cannot be a line-separator character.");
        }
        return normalized;
    }

    private static List<String> copyColumns(
            List<String> columns,
            String argumentName,
            boolean allowEmpty
    ) {
        if (columns == null || columns.isEmpty()) {
            if (allowEmpty) {
                return List.of();
            }
            throw new IllegalArgumentException(
                    "NumericLongFormatReader requires at least one "
                            + argumentName + " entry.");
        }
        List<String> copy = new ArrayList<>(columns.size());
        Set<String> used = new HashSet<>();
        for (String column : columns) {
            if (column == null || column.isBlank()) {
                throw new IllegalArgumentException(
                        argumentName + " cannot contain null or blank names.");
            }
            String normalized = column.trim();
            if (!used.add(normalized)) {
                throw new IllegalArgumentException(
                        argumentName + " contains a duplicate column: "
                                + normalized);
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
        Set<String> normalized = new HashSet<>();
        for (String indicator : AppContext.MissingStrings) {
            if (indicator == null) {
                continue;
            }
            String trimmed = indicator.trim();
            if (!trimmed.isEmpty()) {
                normalized.add(trimmed.toUpperCase(Locale.ROOT));
            }
        }
        return normalized.isEmpty()
                ? Set.of()
                : Collections.unmodifiableSet(normalized);
    }
}