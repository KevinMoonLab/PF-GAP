package datasets.readers;

import ch.randelshofer.fastdoubleparser.JavaDoubleParser;
import core.AppContext;
import datasets.ListObjectDataset;
import de.siegmar.fastcsv.reader.AbstractBaseCsvCallbackHandler;
import de.siegmar.fastcsv.reader.CsvReader;
import preprocessing.standardization.StandardizationStats;
import preprocessing.standardization.Standardizer;

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
 * High-throughput reader for numeric long-format delimited time-series data.
 *
 * <p>Expected organization:</p>
 *
 * <pre>
 * id,time,temp,pressure,label
 * A,0,10.1,100.0,class1
 * A,1,10.4,101.2,class1
 * A,2,10.2,100.8,class1
 * B,0,5.2,80.1,class2
 * B,1,5.4,81.0,class2
 * </pre>
 *
 * <p>One record represents one time point. Records are grouped by the
 * configured {@code idColumn}. Within each group, records are sorted by
 * {@code timeColumn} when one is configured. Otherwise, input order is
 * retained.</p>
 *
 * <p>Output representations:</p>
 *
 * <pre>
 * One feature:
 *     no missing values:       double[time]
 *     missing values:          Double[time]
 *
 * Multiple features:
 *     no missing values:       double[feature][time]
 *     missing values:          Double[feature][time]
 * </pre>
 *
 * <p>This implementation uses FastCSV's custom callback API. Selected
 * numerical fields are parsed directly from FastCSV's character buffer using
 * JavaDoubleParser. It therefore avoids materializing CsvRecord objects,
 * field lists, feature strings, LongRow objects, and boxed feature values
 * during parsing.</p>
 *
 * <p>Missing numeric values are supported. During parsing, numerical values
 * remain in primitive buffers and missing positions are stored separately.
 * Boxing occurs only when the final output representation must be
 * {@code Double[]} or {@code Double[][]}.</p>
 *
 * <p>This reader currently requires:</p>
 *
 * <ul>
 *     <li>A header record</li>
 *     <li>A nonempty ID column</li>
 *     <li>At least one numeric feature column</li>
 *     <li>A single-character field separator</li>
 * </ul>
 *
 * <p>The general {@link LongFormatReader} should be used for mixed,
 * categorical, or row-wise data.</p>
 */
public class NumericLongFormatReader
        implements DatasetReader {

    private static final int DEFAULT_INITIAL_GROUP_CAPACITY =
            256;

    private final String dataFileName;
    private final String entrySeparator;
    private final char fieldSeparator;
    private final boolean hasHeader;
    private final boolean hasMissingValues;
    private final boolean isRegression;

    private final String idColumn;
    private final String timeColumn;
    private final List<String> featureColumns;
    private final List<String> labelColumns;

    private final StandardizationStats standardizationStats;
    private final int initialGroupCapacity;
    private final Set<String> missingIndicators;

    public NumericLongFormatReader(
            ReaderOptions options
    ) {
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
                DEFAULT_INITIAL_GROUP_CAPACITY
        );

        if (!options.isNumeric()) {
            throw new IllegalArgumentException(
                    "NumericLongFormatReader requires "
                            + "ReaderOptions.isNumeric=true."
            );
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
        this(
                dataFileName,
                entrySeparator,
                hasHeader,
                hasMissingValues,
                isRegression,
                idColumn,
                timeColumn,
                featureColumns,
                labelColumns,
                standardizationStats,
                DEFAULT_INITIAL_GROUP_CAPACITY
        );
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
        this.dataFileName =
                requireNonblank(
                        dataFileName,
                        "dataFileName"
                );

        this.entrySeparator =
                validateAndNormalizeSeparator(
                        entrySeparator
                );

        this.fieldSeparator =
                this.entrySeparator.charAt(0);

        this.hasHeader =
                hasHeader;

        this.hasMissingValues =
                hasMissingValues;

        this.isRegression =
                isRegression;

        this.idColumn =
                requireNonblank(
                        idColumn,
                        "idColumn"
                );

        this.timeColumn =
                normalizeNullableString(
                        timeColumn
                );

        this.featureColumns =
                copyColumns(
                        featureColumns,
                        "featureColumns",
                        false
                );

        this.labelColumns =
                copyColumns(
                        labelColumns,
                        "labelColumns",
                        true
                );

        this.standardizationStats =
                standardizationStats;

        if (initialGroupCapacity < 1) {
            throw new IllegalArgumentException(
                    "NumericLongFormatReader initialGroupCapacity must be "
                            + "at least 1. Received: "
                            + initialGroupCapacity
                            + "."
            );
        }

        this.initialGroupCapacity =
                initialGroupCapacity;

        this.missingIndicators =
                snapshotMissingIndicators();

        validateOptions();
    }

    @Override
    public ListObjectDataset read()
            throws IOException {

        long start =
                System.nanoTime();

        Path file =
                validateDataFile();

        NumericLongFormatCallbackHandler handler =
                new NumericLongFormatCallbackHandler(
                        file,
                        idColumn,
                        timeColumn,
                        featureColumns,
                        labelColumns,
                        hasMissingValues,
                        isRegression,
                        missingIndicators,
                        initialGroupCapacity
                );

        try (CsvReader<Boolean> csvReader =
                     CsvReader.builder()
                             .fieldSeparator(
                                     fieldSeparator
                             )
                             /*
                              * Blank physical lines are formatting. Records
                              * containing delimiters and empty selected fields
                              * are still passed to the callback and recognized
                              * as missing values.
                              */
                             .skipEmptyLines(
                                     true
                             )
                             .detectBomHeader(
                                     true
                             )
                             .build(
                                     handler,
                                     file
                             )) {

            /*
             * Consuming the reader drives FastCSV parsing. The Boolean return
             * values are only record-completion markers. The callback handler
             * owns the grouped primitive data.
             */
            for (Boolean ignored : csvReader) {
                // Intentionally empty.
            }
        } catch (IllegalArgumentException e) {
            throw e;
        } catch (RuntimeException e) {
            throw new IOException(
                    "Failed while parsing numeric long-format file: "
                            + file,
                    e
            );
        }

        ListObjectDataset dataset =
                handler.buildDataset(
                        standardizationStats
                );

        long end =
                System.nanoTime();

        LongFormatReader.ProgressLogger.logDuration(
                start,
                end
        );

        return dataset;
    }

    private void validateOptions() {
        if (!hasHeader) {
            throw new IllegalArgumentException(
                    "NumericLongFormatReader currently requires "
                            + "hasHeader=true."
            );
        }

        if (featureColumns.isEmpty()) {
            throw new IllegalArgumentException(
                    "NumericLongFormatReader requires at least one "
                            + "feature column."
            );
        }

        Set<String> reserved =
                new HashSet<>();

        reserved.add(
                idColumn
        );

        if (timeColumn != null
                && !reserved.add(timeColumn)) {

            throw new IllegalArgumentException(
                    "idColumn and timeColumn cannot identify the same column."
            );
        }

        for (String featureColumn : featureColumns) {
            if (!reserved.add(featureColumn)) {
                throw new IllegalArgumentException(
                        "Feature column overlaps an ID, time, or previously "
                                + "selected feature column: "
                                + featureColumn
                );
            }
        }

        for (String labelColumn : labelColumns) {
            if (!reserved.add(labelColumn)) {
                throw new IllegalArgumentException(
                        "Label column overlaps an ID, time, feature, or "
                                + "previously selected label column: "
                                + labelColumn
                );
            }
        }

        if (standardizationStats != null) {
            standardizationStats.validateFeatureCompatibility(
                    featureColumns
            );
        }
    }

    private Path validateDataFile()
            throws IOException {

        Path path =
                Path.of(
                        dataFileName
                );

        if (!Files.exists(path)) {
            throw new IOException(
                    "Numeric long-format file does not exist: "
                            + path
            );
        }

        if (!Files.isRegularFile(path)) {
            throw new IOException(
                    "Numeric long-format path is not a regular file: "
                            + path
            );
        }

        if (!Files.isReadable(path)) {
            throw new IOException(
                    "Numeric long-format file is not readable: "
                            + path
            );
        }

        return path;
    }

    private static ReaderOptions requireOptions(
            ReaderOptions options
    ) {
        if (options == null) {
            throw new IllegalArgumentException(
                    "NumericLongFormatReader requires non-null "
                            + "ReaderOptions."
            );
        }

        return options;
    }

    private static String requireNonblank(
            String value,
            String argumentName
    ) {
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException(
                    "NumericLongFormatReader requires "
                            + argumentName
                            + "."
            );
        }

        return value.trim();
    }

    private static String normalizeNullableString(
            String value
    ) {
        if (value == null) {
            return null;
        }

        String trimmed =
                value.trim();

        if (trimmed.isEmpty()
                || trimmed.equalsIgnoreCase("None")) {

            return null;
        }

        return trimmed;
    }

    private static String validateAndNormalizeSeparator(
            String separator
    ) {
        if (separator == null || separator.isEmpty()) {
            throw new IllegalArgumentException(
                    "NumericLongFormatReader requires entrySeparator."
            );
        }

        String normalized =
                switch (separator) {
                    case "\\t" -> "\t";
                    case "\\n" -> "\n";
                    case "\\r" -> "\r";
                    default -> separator;
                };

        if (normalized.length() != 1) {
            throw new IllegalArgumentException(
                    "NumericLongFormatReader requires a single-character "
                            + "entrySeparator. Received: '"
                            + separator
                            + "'."
            );
        }

        char value =
                normalized.charAt(0);

        if (value == '\n' || value == '\r') {
            throw new IllegalArgumentException(
                    "entrySeparator cannot be a line-separator character."
            );
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
                            + argumentName
                            + " entry."
            );
        }

        List<String> copy =
                new ArrayList<>(
                        columns.size()
                );

        Set<String> used =
                new HashSet<>();

        for (String column : columns) {
            if (column == null || column.isBlank()) {
                throw new IllegalArgumentException(
                        argumentName
                                + " cannot contain null or blank names."
                );
            }

            String normalized =
                    column.trim();

            if (!used.add(normalized)) {
                throw new IllegalArgumentException(
                        argumentName
                                + " contains a duplicate column: "
                                + normalized
                );
            }

            copy.add(
                    normalized
            );
        }

        return Collections.unmodifiableList(
                copy
        );
    }

    private static Set<String> snapshotMissingIndicators() {
        if (AppContext.MissingStrings == null
                || AppContext.MissingStrings.isEmpty()) {

            return Set.of();
        }

        Set<String> normalized =
                new HashSet<>();

        for (String indicator : AppContext.MissingStrings) {
            if (indicator == null) {
                continue;
            }

            String trimmed =
                    indicator.trim();

            if (!trimmed.isEmpty()) {
                normalized.add(
                        trimmed.toUpperCase(
                                Locale.ROOT
                        )
                );
            }
        }

        if (normalized.isEmpty()) {
            return Set.of();
        }

        return Collections.unmodifiableSet(
                normalized
        );
    }

    /**
     * FastCSV callback that groups numeric records directly into primitive
     * per-ID feature buffers.
     */
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

        private final Map<Object, GroupAccumulator> groups =
                new LinkedHashMap<>();

        private List<String> headerFields =
                new ArrayList<>();

        private int columnCount;
        private int idIndex =
                -1;

        private int timeIndex =
                -1;

        private int[] columnToFeature;
        private int[] columnToLabel;

        private double[] currentFeatures;
        private boolean[] currentMissing;
        private Object[] currentLabels;

        private Object currentId;
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
                int initialGroupCapacity
        ) {
            this.file =
                    file;

            this.idColumn =
                    idColumn;

            this.timeColumn =
                    timeColumn;

            this.featureColumns =
                    featureColumns;

            this.labelColumns =
                    labelColumns;

            this.hasMissingValues =
                    hasMissingValues;

            this.isRegression =
                    isRegression;

            this.missingIndicators =
                    missingIndicators;

            this.initialGroupCapacity =
                    initialGroupCapacity;
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
                headerFields.add(
                        new String(
                                buffer,
                                offset,
                                length
                        )
                );

                return;
            }

            if (fieldIndex >= columnCount) {
                throw inconsistentColumnCount(
                        fieldIndex + 1
                );
            }

            if (fieldIndex == idIndex) {
                currentId =
                        parseIdentifier(
                                buffer,
                                offset,
                                length
                        );

                return;
            }

            if (fieldIndex == timeIndex) {
                currentTime =
                        parseGenericField(
                                buffer,
                                offset,
                                length
                        );

                return;
            }

            int featureIndex =
                    columnToFeature[fieldIndex];

            if (featureIndex >= 0) {
                parseFeature(
                        featureIndex,
                        fieldIndex,
                        buffer,
                        offset,
                        length
                );

                return;
            }

            int labelIndex =
                    columnToLabel[fieldIndex];

            if (labelIndex >= 0) {
                currentLabels[labelIndex] =
                        parseLabel(
                                buffer,
                                offset,
                                length
                        );
            }
        }

        @Override
        protected Boolean buildRecord() {
            int actualColumnCount =
                    getFieldCount();

            if (!schemaResolved) {
                resolveSchema(
                        actualColumnCount
                );

                schemaResolved =
                        true;

                headerFields =
                        null;

                return null;
            }

            if (actualColumnCount != columnCount) {
                throw inconsistentColumnCount(
                        actualColumnCount
                );
            }

            if (currentId == null) {
                throw new IllegalArgumentException(
                        "Encountered a missing ID in numeric long-format "
                                + "file "
                                + file
                                + " at data record "
                                + dataRecordCount
                                + ", beginning on CSV line "
                                + getStartingLineNumber()
                                + "."
                );
            }

            Object label =
                    materializeCurrentLabel();

            GroupAccumulator group =
                    groups.computeIfAbsent(
                            currentId,
                            ignored -> new GroupAccumulator(
                                    featureColumns.size(),
                                    hasMissingValues,
                                    initialGroupCapacity
                            )
                    );

            group.append(
                    currentFeatures,
                    currentMissing,
                    currentTime,
                    label,
                    inputOrder,
                    currentId
            );

            resetCurrentRecord();

            dataRecordCount++;
            inputOrder++;

            LongFormatReader.ProgressLogger.logProgress(
                    dataRecordCount - 1
            );

            return Boolean.TRUE;
        }

        private void resolveSchema(
                int actualColumnCount
        ) {
            if (actualColumnCount <= 0) {
                throw new IllegalArgumentException(
                        "Numeric long-format file has no columns: "
                                + file
                );
            }

            if (headerFields.size() != actualColumnCount) {
                throw new IllegalStateException(
                        "FastCSV header field count mismatch in file "
                                + file
                                + ". FastCSV count="
                                + actualColumnCount
                                + ", buffered count="
                                + headerFields.size()
                                + "."
                );
            }

            Map<String, Integer> headerIndex =
                    buildHeaderIndex();

            columnCount =
                    actualColumnCount;

            idIndex =
                    requireColumn(
                            headerIndex,
                            idColumn,
                            "idColumn"
                    );

            timeIndex =
                    timeColumn == null
                            ? -1
                            : requireColumn(
                            headerIndex,
                            timeColumn,
                            "timeColumn"
                    );

            columnToFeature =
                    new int[columnCount];

            columnToLabel =
                    new int[columnCount];

            Arrays.fill(
                    columnToFeature,
                    -1
            );

            Arrays.fill(
                    columnToLabel,
                    -1
            );

            for (int featureIndex = 0;
                 featureIndex < featureColumns.size();
                 featureIndex++) {

                String featureColumn =
                        featureColumns.get(
                                featureIndex
                        );

                int columnIndex =
                        requireColumn(
                                headerIndex,
                                featureColumn,
                                "featureColumns"
                        );

                columnToFeature[columnIndex] =
                        featureIndex;
            }

            for (int labelIndex = 0;
                 labelIndex < labelColumns.size();
                 labelIndex++) {

                String labelColumn =
                        labelColumns.get(
                                labelIndex
                        );

                int columnIndex =
                        requireColumn(
                                headerIndex,
                                labelColumn,
                                "labelColumns"
                        );

                columnToLabel[columnIndex] =
                        labelIndex;
            }

            currentFeatures =
                    new double[featureColumns.size()];

            currentMissing =
                    hasMissingValues
                            ? new boolean[featureColumns.size()]
                            : null;

            currentLabels =
                    new Object[labelColumns.size()];
        }

        private Map<String, Integer> buildHeaderIndex() {
            Map<String, Integer> headerIndex =
                    new HashMap<>(
                            Math.max(
                                    16,
                                    headerFields.size() * 2
                            )
                    );

            for (int index = 0;
                 index < headerFields.size();
                 index++) {

                String rawName =
                        headerFields.get(
                                index
                        );

                String name =
                        rawName == null
                                ? ""
                                : rawName.trim();

                if (name.isEmpty()) {
                    throw new IllegalArgumentException(
                            "Numeric long-format file contains a blank "
                                    + "header at column "
                                    + index
                                    + ": "
                                    + file
                    );
                }

                Integer previous =
                        headerIndex.put(
                                name,
                                index
                        );

                if (previous != null) {
                    throw new IllegalArgumentException(
                            "Numeric long-format file contains duplicate "
                                    + "header '"
                                    + name
                                    + "': "
                                    + file
                    );
                }
            }

            return headerIndex;
        }

        private int requireColumn(
                Map<String, Integer> headerIndex,
                String columnName,
                String optionName
        ) {
            Integer index =
                    headerIndex.get(
                            columnName
                    );

            if (index == null) {
                throw new IllegalArgumentException(
                        "Column not found for "
                                + optionName
                                + ": "
                                + columnName
                                + ". Available columns: "
                                + headerIndex.keySet()
                );
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
            TrimmedRange range =
                    trimRange(
                            buffer,
                            offset,
                            length
                    );

            if (isMissing(
                    buffer,
                    range.offset,
                    range.length
            )) {
                if (!hasMissingValues) {
                    throw new IllegalArgumentException(
                            "Encountered a missing numeric feature in file "
                                    + file
                                    + " at data record "
                                    + dataRecordCount
                                    + ", column "
                                    + fieldIndex
                                    + ", beginning on CSV line "
                                    + getStartingLineNumber()
                                    + ", but hasMissingValues=false."
                    );
                }

                currentFeatures[featureIndex] =
                        0.0;

                currentMissing[featureIndex] =
                        true;

                return;
            }

            try {
                currentFeatures[featureIndex] =
                        JavaDoubleParser.parseDouble(
                                buffer,
                                range.offset,
                                range.length
                        );

                if (currentMissing != null) {
                    currentMissing[featureIndex] =
                            false;
                }
            } catch (NumberFormatException e) {
                throw new IllegalArgumentException(
                        "Could not parse numeric feature in file "
                                + file
                                + " at data record "
                                + dataRecordCount
                                + ", column "
                                + fieldIndex
                                + ", beginning on CSV line "
                                + getStartingLineNumber()
                                + ".",
                        e
                );
            }
        }

        private Object parseIdentifier(
                char[] buffer,
                int offset,
                int length
        ) {
            TrimmedRange range =
                    trimRange(
                            buffer,
                            offset,
                            length
                    );

            if (isMissing(
                    buffer,
                    range.offset,
                    range.length
            )) {
                return null;
            }

            String value =
                    new String(
                            buffer,
                            range.offset,
                            range.length
                    );

            /*
             * ID values are parsed generically so numeric IDs group by
             * numeric value and textual IDs remain strings.
             */
            return parseGenericString(
                    value
            );
        }

        private Object parseGenericField(
                char[] buffer,
                int offset,
                int length
        ) {
            TrimmedRange range =
                    trimRange(
                            buffer,
                            offset,
                            length
                    );

            if (isMissing(
                    buffer,
                    range.offset,
                    range.length
            )) {
                return null;
            }

            String value =
                    new String(
                            buffer,
                            range.offset,
                            range.length
                    );

            return parseGenericString(
                    value
            );
        }

        private Object parseLabel(
                char[] buffer,
                int offset,
                int length
        ) {
            TrimmedRange range =
                    trimRange(
                            buffer,
                            offset,
                            length
                    );

            if (isMissing(
                    buffer,
                    range.offset,
                    range.length
            )) {
                return null;
            }

            String value =
                    new String(
                            buffer,
                            range.offset,
                            range.length
                    );

            if (isRegression) {
                return JavaDoubleParser.parseDouble(
                        value
                );
            }

            try {
                return Integer.parseInt(
                        value
                );
            } catch (NumberFormatException ignored) {
                try {
                    double numericValue =
                            JavaDoubleParser.parseDouble(
                                    value
                            );

                    if (numericValue == Math.rint(numericValue)
                            && numericValue >= Integer.MIN_VALUE
                            && numericValue <= Integer.MAX_VALUE) {

                        return (int) numericValue;
                    }

                    return numericValue;
                } catch (NumberFormatException ignoredAgain) {
                    return value;
                }
            }
        }

        private Object parseGenericString(
                String value
        ) {
            try {
                return Integer.parseInt(
                        value
                );
            } catch (NumberFormatException ignored) {
                try {
                    return JavaDoubleParser.parseDouble(
                            value
                    );
                } catch (NumberFormatException ignoredAgain) {
                    if (value.equalsIgnoreCase("true")
                            || value.equalsIgnoreCase("false")) {

                        return Boolean.parseBoolean(
                                value
                        );
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

            List<Object> labels =
                    new ArrayList<>(
                            currentLabels.length
                    );

            labels.addAll(
                    Arrays.asList(
                            currentLabels.clone()
                    )
            );

            return Collections.unmodifiableList(
                    labels
            );
        }

        private void resetCurrentRecord() {
            currentId =
                    null;

            currentTime =
                    null;

            Arrays.fill(
                    currentFeatures,
                    0.0
            );

            if (currentMissing != null) {
                Arrays.fill(
                        currentMissing,
                        false
                );
            }

            Arrays.fill(
                    currentLabels,
                    null
            );
        }

        private boolean isMissing(
                char[] buffer,
                int offset,
                int length
        ) {
            if (length == 0) {
                return true;
            }

            if (missingIndicators.isEmpty()) {
                return false;
            }

            String token =
                    new String(
                            buffer,
                            offset,
                            length
                    )
                            .toUpperCase(
                                    Locale.ROOT
                            );

            return missingIndicators.contains(
                    token
            );
        }

        private TrimmedRange trimRange(
                char[] buffer,
                int offset,
                int length
        ) {
            int start =
                    offset;

            int end =
                    offset + length;

            while (start < end
                    && Character.isWhitespace(buffer[start])) {

                start++;
            }

            while (end > start
                    && Character.isWhitespace(buffer[end - 1])) {

                end--;
            }

            return new TrimmedRange(
                    start,
                    end - start
            );
        }

        private IllegalArgumentException inconsistentColumnCount(
                int actualColumnCount
        ) {
            return new IllegalArgumentException(
                    "Inconsistent column count in numeric long-format file "
                            + file
                            + " at CSV record beginning on line "
                            + getStartingLineNumber()
                            + ". Expected "
                            + columnCount
                            + " columns but found "
                            + actualColumnCount
                            + "."
            );
        }

        private ListObjectDataset buildDataset(
                StandardizationStats standardizationStats
        ) {
            if (!schemaResolved) {
                throw new IllegalArgumentException(
                        "Numeric long-format file is empty: "
                                + file
                );
            }

            if (dataRecordCount == 0 || groups.isEmpty()) {
                throw new IllegalArgumentException(
                        "Numeric long-format file contains no data records: "
                                + file
                );
            }

            ListObjectDataset dataset =
                    new ListObjectDataset(
                            groups.size()
                    );

            int instanceIndex =
                    0;

            int commonLength =
                    -1;

            boolean unequalLengths =
                    false;

            for (Map.Entry<Object, GroupAccumulator> entry
                    : groups.entrySet()) {

                Object id =
                        entry.getKey();

                GroupAccumulator group =
                        entry.getValue();

                group.sortByTimeIfNeeded(
                        timeColumn != null
                );

                Object data =
                        group.toSeries();

                if (standardizationStats != null) {
                    Standardizer.transformInstanceInPlace(
                            data,
                            standardizationStats
                    );
                }

                Object label =
                        group.getLabel();

                dataset.add(
                        label,
                        data,
                        instanceIndex
                );

                int length =
                        group.size();

                if (commonLength < 0) {
                    commonLength =
                            length;
                } else if (length != commonLength) {
                    unequalLengths =
                            true;
                }

                instanceIndex++;
            }

            int datasetLength =
                    unequalLengths
                            ? 0
                            : Math.max(
                            commonLength,
                            0
                    );

            dataset.setLength(
                    datasetLength
            );

            AppContext.length =
                    datasetLength;

            return dataset;
        }
    }

    /**
     * Accumulates all records for one ID directly into primitive
     * feature-major buffers.
     */
    private static final class GroupAccumulator {

        private final PrimitiveDoubleBuffer[] featureValues;
        private final MissingBuffer[] missingPositions;
        private final boolean nullableOutput;

        private final List<Object> timeValues;
        private final IntBuffer inputOrders;

        private Object label;
        private boolean labelInitialized;

        private GroupAccumulator(
                int featureCount,
                boolean nullableOutput,
                int initialCapacity
        ) {
            this.nullableOutput =
                    nullableOutput;

            this.featureValues =
                    new PrimitiveDoubleBuffer[featureCount];

            this.missingPositions =
                    nullableOutput
                            ? new MissingBuffer[featureCount]
                            : null;

            for (int featureIndex = 0;
                 featureIndex < featureCount;
                 featureIndex++) {

                featureValues[featureIndex] =
                        new PrimitiveDoubleBuffer(
                                initialCapacity
                        );

                if (nullableOutput) {
                    missingPositions[featureIndex] =
                            new MissingBuffer(
                                    initialCapacity
                            );
                }
            }

            this.timeValues =
                    new ArrayList<>(
                            initialCapacity
                    );

            this.inputOrders =
                    new IntBuffer(
                            initialCapacity
                    );
        }

        private void append(
                double[] values,
                boolean[] missing,
                Object timeValue,
                Object rowLabel,
                int inputOrder,
                Object id
        ) {
            validateLabel(
                    rowLabel,
                    id
            );

            for (int featureIndex = 0;
                 featureIndex < featureValues.length;
                 featureIndex++) {

                featureValues[featureIndex].add(
                        values[featureIndex]
                );

                if (nullableOutput) {
                    missingPositions[featureIndex].add(
                            missing[featureIndex]
                    );
                }
            }

            timeValues.add(
                    timeValue
            );

            inputOrders.add(
                    inputOrder
            );
        }

        private void validateLabel(
                Object rowLabel,
                Object id
        ) {
            if (!labelInitialized) {
                label =
                        rowLabel;

                labelInitialized =
                        true;

                return;
            }

            if (!Objects.equals(
                    label,
                    rowLabel
            )) {
                throw new IllegalArgumentException(
                        "Inconsistent labels found within numeric "
                                + "long-format group for id: "
                                + id
                );
            }
        }

        private Object getLabel() {
            return label;
        }

        private int size() {
            return featureValues.length == 0
                    ? 0
                    : featureValues[0].size();
        }

        private void sortByTimeIfNeeded(
                boolean hasTimeColumn
        ) {
            if (!hasTimeColumn || size() < 2) {
                return;
            }

            Integer[] order =
                    new Integer[size()];

            for (int index = 0;
                 index < order.length;
                 index++) {

                order[index] =
                        index;
            }

            Arrays.sort(
                    order,
                    Comparator
                            .comparing(
                                    (Integer index) ->
                                            timeValues.get(index),
                                    GroupAccumulator::compareTimeValues
                            )
                            .thenComparingInt(
                                    inputOrders::get
                            )
            );

            boolean alreadySorted =
                    true;

            for (int index = 0;
                 index < order.length;
                 index++) {

                if (order[index] != index) {
                    alreadySorted =
                            false;

                    break;
                }
            }

            if (alreadySorted) {
                return;
            }

            for (PrimitiveDoubleBuffer values : featureValues) {
                values.reorder(
                        order
                );
            }

            if (nullableOutput) {
                for (MissingBuffer missing : missingPositions) {
                    missing.reorder(
                            order
                    );
                }
            }

            List<Object> reorderedTimes =
                    new ArrayList<>(
                            order.length
                    );

            int[] reorderedInputOrders =
                    new int[order.length];

            for (int outputIndex = 0;
                 outputIndex < order.length;
                 outputIndex++) {

                int sourceIndex =
                        order[outputIndex];

                reorderedTimes.add(
                        timeValues.get(sourceIndex)
                );

                reorderedInputOrders[outputIndex] =
                        inputOrders.get(sourceIndex);
            }

            timeValues.clear();
            timeValues.addAll(
                    reorderedTimes
            );

            inputOrders.replaceWith(
                    reorderedInputOrders
            );
        }

        private Object toSeries() {
            if (featureValues.length == 1) {
                if (!nullableOutput) {
                    return featureValues[0].toArray();
                }

                return featureValues[0].toNullableBoxedArray(
                        missingPositions[0]
                );
            }

            if (!nullableOutput) {
                double[][] result =
                        new double[featureValues.length][];

                for (int featureIndex = 0;
                     featureIndex < featureValues.length;
                     featureIndex++) {

                    result[featureIndex] =
                            featureValues[featureIndex].toArray();
                }

                return result;
            }

            Double[][] result =
                    new Double[featureValues.length][];

            for (int featureIndex = 0;
                 featureIndex < featureValues.length;
                 featureIndex++) {

                result[featureIndex] =
                        featureValues[featureIndex]
                                .toNullableBoxedArray(
                                        missingPositions[featureIndex]
                                );
            }

            return result;
        }

        @SuppressWarnings({
                "rawtypes",
                "unchecked"
        })
        private static int compareTimeValues(
                Object first,
                Object second
        ) {
            if (first == null && second == null) {
                return 0;
            }

            if (first == null) {
                return -1;
            }

            if (second == null) {
                return 1;
            }

            if (first instanceof Number
                    && second instanceof Number) {

                return Double.compare(
                        ((Number) first).doubleValue(),
                        ((Number) second).doubleValue()
                );
            }

            if (first instanceof Comparable
                    && first.getClass().isInstance(second)) {

                return ((Comparable) first).compareTo(
                        second
                );
            }

            return first.toString().compareTo(
                    second.toString()
            );
        }
    }

    private static final class PrimitiveDoubleBuffer {

        private double[] values;
        private int size;

        private PrimitiveDoubleBuffer(
                int initialCapacity
        ) {
            values =
                    new double[
                            Math.max(
                                    1,
                                    initialCapacity
                            )
                            ];
        }

        private void add(
                double value
        ) {
            ensureCapacity(
                    size + 1
            );

            values[size++] =
                    value;
        }

        private int size() {
            return size;
        }

        private double[] toArray() {
            if (size == values.length) {
                return values;
            }

            return Arrays.copyOf(
                    values,
                    size
            );
        }

        private Double[] toNullableBoxedArray(
                MissingBuffer missing
        ) {
            if (missing.size() != size) {
                throw new IllegalStateException(
                        "Numeric and missing-position buffers have "
                                + "different lengths."
                );
            }

            Double[] result =
                    new Double[size];

            for (int index = 0;
                 index < size;
                 index++) {

                result[index] =
                        missing.get(index)
                                ? null
                                : values[index];
            }

            return result;
        }

        private void reorder(
                Integer[] order
        ) {
            double[] reordered =
                    new double[size];

            for (int outputIndex = 0;
                 outputIndex < order.length;
                 outputIndex++) {

                reordered[outputIndex] =
                        values[order[outputIndex]];
            }

            values =
                    reordered;
        }

        private void ensureCapacity(
                int requiredCapacity
        ) {
            if (requiredCapacity <= values.length) {
                return;
            }

            int expandedCapacity =
                    nextCapacity(
                            values.length,
                            requiredCapacity
                    );

            values =
                    Arrays.copyOf(
                            values,
                            expandedCapacity
                    );
        }
    }

    private static final class MissingBuffer {

        private boolean[] values;
        private int size;

        private MissingBuffer(
                int initialCapacity
        ) {
            values =
                    new boolean[
                            Math.max(
                                    1,
                                    initialCapacity
                            )
                            ];
        }

        private void add(
                boolean value
        ) {
            ensureCapacity(
                    size + 1
            );

            values[size++] =
                    value;
        }

        private boolean get(
                int index
        ) {
            return values[index];
        }

        private int size() {
            return size;
        }

        private void reorder(
                Integer[] order
        ) {
            boolean[] reordered =
                    new boolean[size];

            for (int outputIndex = 0;
                 outputIndex < order.length;
                 outputIndex++) {

                reordered[outputIndex] =
                        values[order[outputIndex]];
            }

            values =
                    reordered;
        }

        private void ensureCapacity(
                int requiredCapacity
        ) {
            if (requiredCapacity <= values.length) {
                return;
            }

            int expandedCapacity =
                    nextCapacity(
                            values.length,
                            requiredCapacity
                    );

            values =
                    Arrays.copyOf(
                            values,
                            expandedCapacity
                    );
        }
    }

    private static final class IntBuffer {

        private int[] values;
        private int size;

        private IntBuffer(
                int initialCapacity
        ) {
            values =
                    new int[
                            Math.max(
                                    1,
                                    initialCapacity
                            )
                            ];
        }

        private void add(
                int value
        ) {
            ensureCapacity(
                    size + 1
            );

            values[size++] =
                    value;
        }

        private int get(
                int index
        ) {
            return values[index];
        }

        private void replaceWith(
                int[] replacement
        ) {
            values =
                    replacement;

            size =
                    replacement.length;
        }

        private void ensureCapacity(
                int requiredCapacity
        ) {
            if (requiredCapacity <= values.length) {
                return;
            }

            int expandedCapacity =
                    nextCapacity(
                            values.length,
                            requiredCapacity
                    );

            values =
                    Arrays.copyOf(
                            values,
                            expandedCapacity
                    );
        }
    }

    private static int nextCapacity(
            int currentCapacity,
            int requiredCapacity
    ) {
        int expandedCapacity =
                currentCapacity <= Integer.MAX_VALUE / 2
                        ? currentCapacity << 1
                        : Integer.MAX_VALUE;

        if (expandedCapacity < requiredCapacity) {
            expandedCapacity =
                    requiredCapacity;
        }

        if (expandedCapacity < 0
                || expandedCapacity < currentCapacity) {

            throw new OutOfMemoryError(
                    "Required numeric long-format buffer is too large."
            );
        }

        return expandedCapacity;
    }

    private static final class TrimmedRange {

        private final int offset;
        private final int length;

        private TrimmedRange(
                int offset,
                int length
        ) {
            this.offset =
                    offset;

            this.length =
                    length;
        }
    }
}