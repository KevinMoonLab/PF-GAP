package datasets.readers;

import ch.randelshofer.fastdoubleparser.JavaDoubleParser;
import core.AppContext;
import datasets.readers.lazy.LazySeriesReader;
import datasets.readers.lazy.LazySeriesRef;
import de.siegmar.fastcsv.reader.CsvReader;
import de.siegmar.fastcsv.reader.CsvRecord;
import preprocessing.standardization.StandardizationStats;
import preprocessing.standardization.Standardizer;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

/**
 * Materializes one time series from one delimited file.
 *
 * <p>This class is shared by:</p>
 *
 * <ul>
 *     <li>PerFileDelimitedReader</li>
 *     <li>LazyPerFileDelimitedReader</li>
 * </ul>
 *
 * <p>The class itself is not inherently eager or lazy. It reads one file
 * whenever {@link #read(LazySeriesRef)} or {@link #readFile(Path)} is called.
 * The higher-level dataset reader determines when that materialization occurs.</p>
 *
 * <p>Expected file organization:</p>
 *
 * <pre>
 * one record = one time point
 * one selected column = one time-series dimension
 * </pre>
 *
 * <p>Returned representations:</p>
 *
 * <pre>
 * numeric data without missing values:
 *     double[dimension][time]
 *
 * numeric data with missing values:
 *     Double[dimension][time]
 *
 * nonnumeric data:
 *     Object[dimension][time]
 * </pre>
 *
 * <p>When numeric data permits missing values, missing entries are represented
 * as {@code null} in the returned {@code Double[][]}. During parsing, missing
 * positions are tracked separately from numerical values so that a genuine
 * numerical {@code NaN} is not confused with a missing value.</p>
 *
 * <p>Header behavior:</p>
 *
 * <ul>
 *     <li>
 *         When {@code hasHeader=true}, {@code featureColumns} and
 *         {@code timeColumn} are interpreted as column names.
 *     </li>
 *     <li>
 *         When {@code hasHeader=false}, they are interpreted as zero-based
 *         integer column indices.
 *     </li>
 *     <li>
 *         When {@code featureColumns} is empty, every column except the
 *         configured time column is treated as a feature column.
 *     </li>
 * </ul>
 *
 * <p>The reader is backed by FastCSV and supports quoted fields, delimiters
 * inside quoted fields, and embedded record separators in quoted fields.
 * Empty records are ignored.</p>
 *
 * <p>This implementation currently requires a single-character field
 * separator. This covers ordinary CSV, TSV, pipe-delimited, and
 * semicolon-delimited files.</p>
 */
public class PerFileDelimitedSeriesReader
        implements LazySeriesReader {

    private static final int DEFAULT_INITIAL_TIME_CAPACITY =
            256;

    private static final int MINIMUM_INITIAL_TIME_CAPACITY =
            1;

    private final String entrySeparator;
    private final char fieldSeparator;
    private final boolean hasHeader;
    private final String timeColumn;
    private final List<String> featureColumns;
    private final boolean isNumeric;
    private final boolean hasMissingValues;
    private final StandardizationStats standardizationStats;
    private final int initialTimeCapacity;
    private final Set<String> missingStrings;

    public PerFileDelimitedSeriesReader(
            String entrySeparator,
            boolean hasHeader,
            String timeColumn,
            List<String> featureColumns,
            boolean isNumeric,
            boolean hasMissingValues,
            StandardizationStats standardizationStats,
            int initialTimeCapacity
    ) {
        this.entrySeparator =
                validateAndNormalizeSeparator(
                        entrySeparator
                );

        this.fieldSeparator =
                this.entrySeparator.charAt(0);

        this.hasHeader =
                hasHeader;

        this.timeColumn =
                normalizeNullableString(
                        timeColumn
                );

        this.featureColumns =
                featureColumns == null
                        ? new ArrayList<>()
                        : new ArrayList<>(featureColumns);

        this.isNumeric =
                isNumeric;

        this.hasMissingValues =
                hasMissingValues;

        this.standardizationStats =
                standardizationStats;

        if (initialTimeCapacity < MINIMUM_INITIAL_TIME_CAPACITY) {
            throw new IllegalArgumentException(
                    "PerFileDelimitedSeriesReader initialTimeCapacity "
                    + "must be as least "
                    + MINIMUM_INITIAL_TIME_CAPACITY
                    + ". Received: "
                    + initialTimeCapacity
                    + "."
            );
        }

        this.initialTimeCapacity = initialTimeCapacity;

        this.missingStrings = snapshotMissingStrings();

        validateStandardizationConfiguration();
    }

    public PerFileDelimitedSeriesReader(
            String entrySeparator,
            boolean hasHeader,
            String timeColumn,
            List<String> featureColumns,
            boolean isNumeric,
            boolean hasMissingValues,
            StandardizationStats standardizationStats
    ) {
        this(
                entrySeparator,
                hasHeader,
                timeColumn,
                featureColumns,
                isNumeric,
                hasMissingValues,
                null,
                DEFAULT_INITIAL_TIME_CAPACITY
        );
    }

    public PerFileDelimitedSeriesReader(
            String entrySeparator,
            boolean hasHeader,
            String timeColumn,
            List<String> featureColumns,
            boolean isNumeric,
            boolean hasMissingValues
    ) {
        this(
                entrySeparator,
                hasHeader,
                timeColumn,
                featureColumns,
                isNumeric,
                hasMissingValues,
                null
        );
    }

    /**
     * Reads and materializes the file identified by a lazy series reference.
     *
     * @param reference lazy reference containing the file to materialize
     * @return materialized dimension-major time series
     */
    @Override
    public Object read(
            LazySeriesRef reference
    ) {
        if (reference == null) {
            throw new IllegalArgumentException(
                    "Cannot read a null LazySeriesRef."
            );
        }

        Path file =
                reference.getFile();

        //try {
        //    return readFile(file);
        //} catch (IOException e) {
        try {
            return readFile(file, false);
        } catch (IOException e) {
            throw new IllegalStateException(
                    "Failed to read delimited time-series file: "
                            + file,
                    e
            );
        }
    }

    /**
     * Reads and materializes one delimited time-series file.
     *
     * <p>The file is processed sequentially. Records are not retained after
     * their selected fields have been appended to the output buffers.</p>
     *
     * @param file file to materialize
     * @return materialized dimension-major time series
     * @throws IOException when the file cannot be read
     */
    public Object readFile(Path file) throws IOException {
        return readFile(file, true);
    }

    private Object readFile(
            Path file, boolean validateFileMetadata
    ) throws IOException {
        if (file == null) {
            throw new IllegalArgumentException(
                    "PerFileDelimitedSeriesReader requires a non-null file."
            );
        }
        if (validateFileMetadata) {
            validateFile(file);
        }


        try (CsvReader<CsvRecord> csvReader =
                     CsvReader.builder()
                             .fieldSeparator(fieldSeparator)
                             .skipEmptyLines(true)
                             .detectBomHeader(true)
                             .ofCsvRecord(file)) {

            var iterator =
                    csvReader.iterator();

            if (!iterator.hasNext()) {
                throw new IOException(
                        "Delimited time-series file is empty: "
                                + file
                );
            }

            CsvRecord firstRecord =
                    iterator.next();

            List<String> firstFields =
                    firstRecord.getFields();

            if (firstFields.isEmpty()) {
                throw new IOException(
                        "Delimited time-series file has no columns: "
                                + file
                );
            }

            List<String> header =
                    hasHeader
                            ? firstFields
                            : null;

            ColumnSelection selection =
                    resolveColumnSelection(
                            file,
                            header,
                            firstFields.size()
                    );

            SeriesAccumulator accumulator =
                    createAccumulator(
                            selection.featureIndices.length
                    );

            int dataRowIndex =
                    0;

            /*
             * When there is no header, the first record is also the first
             * data record and must be materialized.
             */
            if (!hasHeader) {
                appendRecord(
                        file,
                        firstRecord,
                        firstFields,
                        dataRowIndex,
                        selection,
                        accumulator
                );

                dataRowIndex++;
            }

            while (iterator.hasNext()) {
                CsvRecord record =
                        iterator.next();

                List<String> fields =
                        record.getFields();

                appendRecord(
                        file,
                        record,
                        fields,
                        dataRowIndex,
                        selection,
                        accumulator
                );

                dataRowIndex++;
            }

            if (dataRowIndex == 0) {
                throw new IOException(
                        "Delimited time-series file contains no data rows: "
                                + file
                );
            }

            Object series =
                    accumulator.toSeries();

            if (standardizationStats != null) {
                Standardizer.transformInstanceInPlace(
                        series,
                        standardizationStats
                );
            }

            return series;
        } catch (IllegalArgumentException e) {
            throw e;
        } catch (RuntimeException e) {
            throw new IOException(
                    "Failed while parsing delimited time-series file: "
                            + file,
                    e
            );
        }
    }

    private void appendRecord(
            Path file,
            CsvRecord record,
            List<String> fields,
            int dataRowIndex,
            ColumnSelection selection,
            SeriesAccumulator accumulator
    ) {
        validateRecordWidth(
                file,
                record,
                fields.size(),
                selection.columnCount
        );

        if (isNumeric) {
            appendNumericRecord(
                    file,
                    fields,
                    dataRowIndex,
                    selection.featureIndices,
                    accumulator
            );
        } else {
            appendNonnumericRecord(
                    fields,
                    selection.featureIndices,
                    accumulator
            );
        }
    }

    private void appendNumericRecord(
            Path file,
            List<String> fields,
            int dataRowIndex,
            int[] featureIndices,
            SeriesAccumulator accumulator
    ) {
        for (int dimension = 0;
             dimension < featureIndices.length;
             dimension++) {

            int columnIndex =
                    featureIndices[dimension];

            String rawValue =
                    fields.get(columnIndex);

            String token =
                    rawValue == null
                            ? null
                            : rawValue.trim();

            if (isMissingToken(token)) {
                if (!hasMissingValues) {
                    throw new IllegalArgumentException(
                            "Encountered a missing value in file "
                                    + file
                                    + " at data row "
                                    + dataRowIndex
                                    + ", column "
                                    + columnIndex
                                    + ", but hasMissingValues is false."
                    );
                }

                accumulator.addMissing(
                        dimension
                );

                continue;
            }

            try {
                accumulator.addNumeric(
                        dimension,
                        //Double.parseDouble(token)
                        JavaDoubleParser.parseDouble(token)
                );
            } catch (NumberFormatException e) {
                throw new IllegalArgumentException(
                        "Could not parse numeric value '"
                                + token
                                + "' in file "
                                + file
                                + " at data row "
                                + dataRowIndex
                                + ", column "
                                + columnIndex
                                + ".",
                        e
                );
            }
        }
    }

    private void appendNonnumericRecord(
            List<String> fields,
            int[] featureIndices,
            SeriesAccumulator accumulator
    ) {
        for (int dimension = 0;
             dimension < featureIndices.length;
             dimension++) {

            int columnIndex =
                    featureIndices[dimension];

            String value =
                    fields.get(columnIndex);

            accumulator.addObject(
                    dimension,
                    DelimitedFileReader.RowParser.parseValue(
                            value
                    )
            );
        }
    }

    private SeriesAccumulator createAccumulator(
            int dimensionCount
    ) {
        if (isNumeric) {
            return new NumericSeriesAccumulator(
                    dimensionCount,
                    hasMissingValues,
                    initialTimeCapacity
            );
        }

        return new ObjectSeriesAccumulator(
                dimensionCount,
                initialTimeCapacity
        );
    }

    private void validateFile(
            Path file
    ) throws IOException {
        if (!Files.exists(file)) {
            throw new IOException(
                    "Delimited time-series file does not exist: "
                            + file
            );
        }

        if (!Files.isRegularFile(file)) {
            throw new IOException(
                    "Delimited time-series path is not a regular file: "
                            + file
            );
        }

        if (!Files.isReadable(file)) {
            throw new IOException(
                    "Delimited time-series file is not readable: "
                            + file
            );
        }
    }

    private ColumnSelection resolveColumnSelection(
            Path file,
            List<String> header,
            int columnCount
    ) {
        if (columnCount <= 0) {
            throw new IllegalArgumentException(
                    "Delimited file has no columns: "
                            + file
            );
        }

        Map<String, Integer> headerIndices =
                hasHeader
                        ? buildHeaderIndex(
                        file,
                        header
                )
                        : Map.of();

        int timeColumnIndex =
                resolveTimeColumnIndex(
                        file,
                        headerIndices,
                        columnCount
                );

        int[] featureIndices =
                resolveFeatureIndices(
                        file,
                        headerIndices,
                        columnCount,
                        timeColumnIndex
                );

        if (featureIndices.length == 0) {
            throw new IllegalArgumentException(
                    "No feature columns were selected for file: "
                            + file
            );
        }

        return new ColumnSelection(
                columnCount,
                timeColumnIndex,
                featureIndices
        );
    }

    private int resolveTimeColumnIndex(
            Path file,
            Map<String, Integer> headerIndices,
            int columnCount
    ) {
        if (timeColumn == null) {
            return -1;
        }

        if (hasHeader) {
            return findNamedColumn(
                    file,
                    headerIndices,
                    timeColumn,
                    "time"
            );
        }

        int index =
                parseColumnIndex(
                        timeColumn,
                        "time_column",
                        file
                );

        validateColumnIndex(
                index,
                columnCount,
                "time_column",
                file
        );

        return index;
    }

    private int[] resolveFeatureIndices(
            Path file,
            Map<String, Integer> headerIndices,
            int columnCount,
            int timeColumnIndex
    ) {
        if (featureColumns.isEmpty()) {
            return allColumnsExcept(
                    columnCount,
                    timeColumnIndex
            );
        }

        int[] indices =
                new int[featureColumns.size()];

        boolean[] used =
                new boolean[columnCount];

        for (int i = 0;
             i < featureColumns.size();
             i++) {

            String feature =
                    featureColumns.get(i);

            if (feature == null || feature.isBlank()) {
                throw new IllegalArgumentException(
                        "Feature-column names or indices cannot be "
                                + "null or blank."
                );
            }

            int index;

            if (hasHeader) {
                index =
                        findNamedColumn(
                                file,
                                headerIndices,
                                feature,
                                "feature"
                        );
            } else {
                index =
                        parseColumnIndex(
                                feature,
                                "feature column",
                                file
                        );

                validateColumnIndex(
                        index,
                        columnCount,
                        "feature column",
                        file
                );
            }

            if (index == timeColumnIndex) {
                throw new IllegalArgumentException(
                        "Column "
                                + feature
                                + " is configured as both the time column "
                                + "and a feature column in file: "
                                + file
                );
            }

            if (used[index]) {
                throw new IllegalArgumentException(
                        "Feature column was selected more than once: "
                                + feature
                                + " in file "
                                + file
                );
            }

            used[index] =
                    true;

            indices[i] =
                    index;
        }

        return indices;
    }

    private int findNamedColumn(
            Path file,
            Map<String, Integer> headerIndices,
            String requestedColumn,
            String role
    ) {
        Integer index =
                headerIndices.get(
                        requestedColumn
                );

        if (index == null) {
            throw new IllegalArgumentException(
                    "Could not find "
                            + role
                            + " column '"
                            + requestedColumn
                            + "' in file "
                            + file
                            + ". Available columns: "
                            + headerIndices.keySet()
            );
        }

        return index;
    }

    private Map<String, Integer> buildHeaderIndex(
            Path file,
            List<String> header
    ) {
        if (header == null) {
            throw new IllegalArgumentException(
                    "A header was expected but was not available in file: "
                            + file
            );
        }

        Map<String, Integer> indices =
                new HashMap<>(
                        Math.max(
                                16,
                                header.size() * 2
                        )
                );

        for (int i = 0;
             i < header.size();
             i++) {

            String rawName =
                    header.get(i);

            String name =
                    rawName == null
                            ? ""
                            : rawName.trim();

            if (name.isEmpty()) {
                throw new IllegalArgumentException(
                        "Delimited file contains a blank header at column "
                                + i
                                + ": "
                                + file
                );
            }

            Integer previous =
                    indices.put(
                            name,
                            i
                    );

            if (previous != null) {
                throw new IllegalArgumentException(
                        "Delimited file contains duplicate header '"
                                + name
                                + "': "
                                + file
                );
            }
        }

        return indices;
    }

    private int[] allColumnsExcept(
            int columnCount,
            int excludedIndex
    ) {
        int resultLength =
                excludedIndex >= 0
                        ? columnCount - 1
                        : columnCount;

        int[] indices =
                new int[resultLength];

        int outputIndex =
                0;

        for (int columnIndex = 0;
             columnIndex < columnCount;
             columnIndex++) {

            if (columnIndex == excludedIndex) {
                continue;
            }

            indices[outputIndex++] =
                    columnIndex;
        }

        return indices;
    }

    private boolean isMissingToken(
            String token
    ) {
        return token == null
                || token.isEmpty()
                || missingStrings.contains(token);
    }

    private static Set<String> snapshotMissingStrings() {
        if (AppContext.MissingStrings == null
                || AppContext.MissingStrings.isEmpty()) {

            return Set.of();
        }

        Set<String> snapshot =
                new HashSet<>();

        for (String value : AppContext.MissingStrings) {
            if (value == null) {
                continue;
            }

            String trimmed =
                    value.trim();

            if (!trimmed.isEmpty()) {
                snapshot.add(
                        trimmed
                );
            }
        }

        if (snapshot.isEmpty()) {
            return Set.of();
        }

        return Collections.unmodifiableSet(
                snapshot
        );
    }

    private void validateRecordWidth(
            Path file,
            CsvRecord record,
            int actualColumnCount,
            int expectedColumnCount
    ) {
        if (actualColumnCount == expectedColumnCount) {
            return;
        }

        throw new IllegalArgumentException(
                "Inconsistent column count in file "
                        + file
                        + " at CSV record beginning on line "
                        + record.getStartingLineNumber()
                        + ". Expected "
                        + expectedColumnCount
                        + " columns but found "
                        + actualColumnCount
                        + "."
        );
    }

    private int parseColumnIndex(
            String value,
            String argumentName,
            Path file
    ) {
        try {
            return Integer.parseInt(
                    value.trim()
            );
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException(
                    argumentName
                            + " must be a zero-based integer column "
                            + "index when hasHeader is false. Received '"
                            + value
                            + "' for file "
                            + file
                            + ".",
                    e
            );
        }
    }

    private void validateColumnIndex(
            int index,
            int columnCount,
            String argumentName,
            Path file
    ) {
        if (index < 0 || index >= columnCount) {
            throw new IllegalArgumentException(
                    argumentName
                            + " index "
                            + index
                            + " is outside the valid range [0, "
                            + (columnCount - 1)
                            + "] for file "
                            + file
                            + "."
            );
        }
    }

    private static String validateAndNormalizeSeparator(
            String separator
    ) {
        if (separator == null || separator.isEmpty()) {
            throw new IllegalArgumentException(
                    "PerFileDelimitedSeriesReader requires a non-empty "
                            + "entry separator."
            );
        }

        String normalized =
                normalizeSeparator(
                        separator
                );

        if (normalized.length() != 1) {
            throw new IllegalArgumentException(
                    "FastCSV-backed PerFileDelimitedSeriesReader currently "
                            + "requires a single-character entry separator. "
                            + "Received: '"
                            + separator
                            + "'."
            );
        }

        char delimiter =
                normalized.charAt(0);

        if (delimiter == '\n' || delimiter == '\r') {
            throw new IllegalArgumentException(
                    "A line-separator character cannot be used as the "
                            + "entry separator."
            );
        }

        return normalized;
    }

    private static String normalizeSeparator(
            String separator
    ) {
        return switch (separator) {
            case "\\t" -> "\t";
            case "\\n" -> "\n";
            case "\\r" -> "\r";
            default -> separator;
        };
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

    private void validateStandardizationConfiguration() {
        if (standardizationStats == null) {
            return;
        }

        if (!isNumeric) {
            throw new IllegalArgumentException(
                    "Standardization statistics cannot be applied by "
                            + "PerFileDelimitedSeriesReader when "
                            + "isNumeric=false."
            );
        }

        /*
         * Empty feature columns are valid. In that case, dimensions are
         * aligned positionally and the realized dimension count is validated
         * by Standardizer when the series is transformed.
         */
        if (!featureColumns.isEmpty()) {
            standardizationStats.validateFeatureCompatibility(
                    featureColumns
            );
        }
    }

    private interface SeriesAccumulator {

        default void addNumeric(
                int dimension,
                double value
        ) {
            throw new UnsupportedOperationException(
                    "This accumulator does not support numeric values."
            );
        }

        default void addMissing(
                int dimension
        ) {
            throw new UnsupportedOperationException(
                    "This accumulator does not support missing values."
            );
        }

        default void addObject(
                int dimension,
                Object value
        ) {
            throw new UnsupportedOperationException(
                    "This accumulator does not support object values."
            );
        }

        Object toSeries();
    }

    private static final class NumericSeriesAccumulator
            implements SeriesAccumulator {

        private final PrimitiveDoubleBuffer[] dimensions;
        private final MissingPositionBuffer[] missingPositions;
        private final boolean boxedOutput;

        private NumericSeriesAccumulator(
                int dimensionCount,
                boolean boxedOutput,
                int initialTimeCapacity
        ) {
            this.boxedOutput =
                    boxedOutput;

            this.dimensions =
                    new PrimitiveDoubleBuffer[dimensionCount];

            this.missingPositions =
                    boxedOutput
                            ? new MissingPositionBuffer[dimensionCount]
                            : null;

            for (int dimension = 0;
                 dimension < dimensionCount;
                 dimension++) {

                dimensions[dimension] =
                        new PrimitiveDoubleBuffer(
                                initialTimeCapacity
                        );

                if (boxedOutput) {
                    missingPositions[dimension] =
                            new MissingPositionBuffer(
                                    initialTimeCapacity
                            );
                }
            }
        }

        @Override
        public void addNumeric(
                int dimension,
                double value
        ) {
            dimensions[dimension].add(
                    value
            );

            if (boxedOutput) {
                missingPositions[dimension].add(
                        false
                );
            }
        }

        @Override
        public void addMissing(
                int dimension
        ) {
            if (!boxedOutput) {
                throw new IllegalStateException(
                        "Missing numerical values cannot be appended when "
                                + "boxed output is disabled."
                );
            }

            /*
             * The numerical placeholder is irrelevant because the associated
             * missing-position entry determines whether the final value is null.
             */
            dimensions[dimension].add(
                    0.0
            );

            missingPositions[dimension].add(
                    true
            );
        }

        @Override
        public Object toSeries() {
            if (!boxedOutput) {
                double[][] result =
                        new double[dimensions.length][];

                for (int dimension = 0;
                     dimension < dimensions.length;
                     dimension++) {

                    result[dimension] =
                            dimensions[dimension].toArray();
                }

                return result;
            }

            Double[][] result =
                    new Double[dimensions.length][];

            for (int dimension = 0;
                 dimension < dimensions.length;
                 dimension++) {

                result[dimension] =
                        dimensions[dimension].toNullableBoxedArray(
                                missingPositions[dimension]
                        );
            }

            return result;
        }
    }

    private static final class ObjectSeriesAccumulator
            implements SeriesAccumulator {

        private final ObjectBuffer[] dimensions;

        private ObjectSeriesAccumulator(
                int dimensionCount,
                int initialTimeCapacity
        ) {
            dimensions =
                    new ObjectBuffer[dimensionCount];

            for (int dimension = 0;
                 dimension < dimensionCount;
                 dimension++) {

                dimensions[dimension] =
                        new ObjectBuffer(
                                initialTimeCapacity
                        );
            }
        }

        @Override
        public void addObject(
                int dimension,
                Object value
        ) {
            dimensions[dimension].add(
                    value
            );
        }

        @Override
        public Object toSeries() {
            Object[][] result =
                    new Object[dimensions.length][];

            for (int dimension = 0;
                 dimension < dimensions.length;
                 dimension++) {

                result[dimension] =
                        dimensions[dimension].toArray();
            }

            return result;
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

        private double[] toArray() {
            return Arrays.copyOf(
                    values,
                    size
            );
        }

        private Double[] toNullableBoxedArray(
                MissingPositionBuffer missingPositions
        ) {
            if (missingPositions.size() != size) {
                throw new IllegalStateException(
                        "Numeric-value and missing-position buffers have "
                                + "different lengths."
                );
            }

            Double[] result =
                    new Double[size];

            for (int i = 0;
                 i < size;
                 i++) {

                result[i] =
                        missingPositions.isMissing(i)
                                ? null
                                : values[i];
            }

            return result;
        }

        private void ensureCapacity(
                int requiredCapacity
        ) {
            if (requiredCapacity <= values.length) {
                return;
            }

            int expandedCapacity =
                    Math.max(
                            requiredCapacity,
                            //values.length + (values.length >> 1) + 1
                            values.length << 1
                    );

            if (expandedCapacity < 0) {
                expandedCapacity = Integer.MAX_VALUE;
            }

            values =
                    Arrays.copyOf(
                            values,
                            expandedCapacity
                    );
        }
    }

    private static final class MissingPositionBuffer {

        private boolean[] missing;
        private int size;

        private MissingPositionBuffer(
                int initialCapacity
        ) {
            missing =
                    new boolean[
                            Math.max(
                                    1,
                                    initialCapacity
                            )
                            ];
        }

        private void add(
                boolean isMissing
        ) {
            ensureCapacity(
                    size + 1
            );

            missing[size++] =
                    isMissing;
        }

        private boolean isMissing(
                int index
        ) {
            return missing[index];
        }

        private int size() {
            return size;
        }

        private void ensureCapacity(
                int requiredCapacity
        ) {
            if (requiredCapacity <= missing.length) {
                return;
            }

            int expandedCapacity =
                    Math.max(
                            requiredCapacity,
                            //missing.length + (missing.length >> 1) + 1
                            missing.length << 1
                    );

            if (expandedCapacity < 0) {
                expandedCapacity = Integer.MAX_VALUE;
            }

            missing =
                    Arrays.copyOf(
                            missing,
                            expandedCapacity
                    );
        }
    }

    private static final class ObjectBuffer {

        private Object[] values;
        private int size;

        private ObjectBuffer(
                int initialCapacity
        ) {
            values =
                    new Object[
                            Math.max(
                                    1,
                                    initialCapacity
                            )
                            ];
        }

        private void add(
                Object value
        ) {
            ensureCapacity(
                    size + 1
            );

            values[size++] =
                    value;
        }

        private Object[] toArray() {
            return Arrays.copyOf(
                    values,
                    size
            );
        }

        private void ensureCapacity(
                int requiredCapacity
        ) {
            if (requiredCapacity <= values.length) {
                return;
            }

            int expandedCapacity =
                    Math.max(
                            requiredCapacity,
                            //values.length + (values.length >> 1) + 1
                            values.length <<1
                    );

            if (expandedCapacity < 0) {
                expandedCapacity = Integer.MAX_VALUE;
            }

            values =
                    Arrays.copyOf(
                            values,
                            expandedCapacity
                    );
        }
    }

    private static final class ColumnSelection {

        private final int columnCount;
        private final int timeColumnIndex;
        private final int[] featureIndices;

        private ColumnSelection(
                int columnCount,
                int timeColumnIndex,
                int[] featureIndices
        ) {
            this.columnCount =
                    columnCount;

            this.timeColumnIndex =
                    timeColumnIndex;

            this.featureIndices =
                    featureIndices.clone();
        }
    }
}